"""Evaluate model-authored Python edits with hidden tests and a WASI runtime."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
import copy
import difflib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import secrets
import statistics
import sys
import tempfile
import time

from dotenv import dotenv_values
from wasi_python import run

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.agent import Agent
from agents.verification import Verification


TARGET = "visible_reproducer"
DATASET = Path(__file__).with_name("open_repair_cases.json")
PUBLIC = "visible"
RETRY_LIMIT = 1
TOOL_BUDGET = 10

TOOLS = [
    {
        "name": "view_source",
        "description": "Read one editable Python source file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"], "additionalProperties": False},
    },
    {
        "name": "replace_text",
        "description": "Apply a Python edit in one source file. old must match exactly once.",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"},
        }, "required": ["path", "old", "new"], "additionalProperties": False},
    },
]

GUEST = """
import json
import sys
import types
files, entry, function, args_list, marker = json.loads(sys.argv[1])
emit = print
encode = json.dumps
for name in files:
    sys.modules[name] = types.ModuleType(name)
for name in [n for n in files if n != entry] + [entry]:
    module = sys.modules[name]
    exec(compile(files[name], name + ".py", "exec"), module.__dict__)
function = getattr(sys.modules[entry], function)
for index, args in enumerate(args_list):
    try:
        value = function(*args)
        payload = encode([index, True, value], allow_nan=False)
    except Exception as error:
        payload = encode([index, False, type(error).__name__])
    emit(marker + payload)
"""


def equivalent(actual, expected):
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            equivalent(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(expected, (int, float)):
        return type(actual) in (int, float) and actual == expected
    return actual == expected


def probe(case, files, groups):
    """Compare actual outputs on the host. The guest receives inputs but never expected values."""
    tests = [test for group in groups for test in case[group]]
    marker = secrets.token_hex(12) + ":"
    data = json.dumps([files, case["entry"], case["func"], [args for args, _ in tests], marker])
    result = run(["-c", GUEST, data], root=False, env={}, deadline_s=8,
                 fuel=150_000_000, memory_bytes=256 * 1024 * 1024, output_bytes=16_384)
    actuals = {}
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith(marker):
            try:
                index, ok, actual = json.loads(line[len(marker):])
                if type(index) is int and 0 <= index < len(tests) and type(ok) is bool:
                    actuals[index] = (ok, actual)
            except (ValueError, TypeError):
                continue
    statuses, details = {}, {}
    index = 0
    for group in groups:
        statuses[group], details[group] = [], []
        for args, expected in case[group]:
            ok, actual = actuals.get(index, (False, "no result"))
            index += 1
            if result.trapped:
                status = "unverified"
            elif result.exit_code == 0 and ok and equivalent(actual, expected):
                status = "passed"
            else:
                status = "failed"
            statuses[group].append(status)
            if status != "passed" and group == PUBLIC:
                details[group].append(f"Input {args!r}; expected {expected!r}; received {actual!r}.")
    diagnostic = "\n".join(details.get(PUBLIC, []))
    if result.trapped:
        diagnostic += f"\nGuest execution stopped: {result.trapped}."
    if result.exit_code not in (0, None):
        diagnostic += "\n" + result.stderr.decode("utf-8", errors="replace")[-600:]
    return statuses, diagnostic[:1400]


def preflight(cases):
    """Verify isolation and that each scoring test has the intended before/after behavior."""
    with tempfile.TemporaryDirectory() as temp:
        marker = Path(temp) / "outside.txt"
        marker.write_text("outside", encoding="utf-8")
        old_marker = os.environ.get("BEAR_EVAL_HOST_PROBE")
        os.environ["BEAR_EVAL_HOST_PROBE"] = "present"
        try:
            isolation = run(["-c", """
import os, sys
assert os.environ.get("BEAR_EVAL_HOST_PROBE") is None
try:
    with open(sys.argv[1]):
        raise AssertionError("unexpected host file")
except OSError:
    pass
try:
    import socket
    socket.socket()
except (ImportError, OSError):
    pass
else:
    raise AssertionError("unexpected network socket")
print("isolated")
""", str(marker)], root=False, env={}, deadline_s=8)
        finally:
            if old_marker is None:
                del os.environ["BEAR_EVAL_HOST_PROBE"]
            else:
                os.environ["BEAR_EVAL_HOST_PROBE"] = old_marker
        if isolation.exit_code != 0 or isolation.stdout.strip() != b"isolated":
            raise RuntimeError(f"isolation probe failed: {isolation.stderr!r}")
    groups = (PUBLIC, "hidden", "regression")
    for case in cases:
        before, _ = probe(case, case["files"], groups)
        after, _ = probe(case, {**case["files"], **case["reference"]}, groups)
        for group in groups:
            expected_before = "passed" if group == "regression" else "failed"
            if before[group] != [expected_before] * len(case[group]) or after[group] != ["passed"] * len(case[group]):
                raise ValueError(f"{case['id']} {group}: before={before[group]}, after={after[group]}")
    print(f"Isolation and reference solutions checked: {len(cases)} cases", flush=True)


class RepairAgent(Agent):
    def __init__(self, case, files, **config):
        super().__init__(**config, permission_mode="dontAsk", custom_tools=TOOLS,
                         is_sub_agent=True, max_turns=TOOL_BUDGET,
                         custom_system_prompt=(
                             "Repair the requested defect using view_source and replace_text. "
                             "Inspect the named files, preserve existing behaviors, and apply your edit "
                             "before giving a concise summary. Public verification may return diagnostics."
                         ))
        self.case = case
        self.files = files.copy()
        self.replayed_answer = None

    async def _execute_tool_call(self, name, inp):
        if not isinstance(inp, dict):
            return "Invalid arguments."
        path = inp.get("path")
        if not isinstance(path, str) or not path.endswith(".py") or path[:-3] not in self.files:
            return "Unknown source file."
        module = path[:-3]
        source = self.files[module]
        if name == "view_source":
            return source
        if name != "replace_text":
            return "Unknown tool."
        old, new = inp.get("old"), inp.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str) or source.count(old) != 1:
            return "old must match exactly once. Re-read the source before editing."
        updated = source.replace(old, new, 1)
        if len(updated) > 12000:
            return "Replacement exceeds the source size limit."
        self.files[module] = updated
        return f"Updated {path}."

    async def chat(self, user_message, *, runtime_feedback=False):
        if self.replayed_answer is not None and not runtime_feedback:
            answer, self.replayed_answer = self.replayed_answer, None
            return answer
        return await super().chat(user_message, runtime_feedback=runtime_feedback)

    async def _verify_repair_target(self, node_id, *, root, python, timeout):
        start = time.monotonic()
        groups, diagnostic = await asyncio.to_thread(probe, self.case, self.files.copy(), (PUBLIC,))
        statuses = groups[PUBLIC]
        status = "passed" if all(s == "passed" for s in statuses) else (
            "unverified" if "unverified" in statuses else "failed"
        )
        return Verification(node_id, status, f"Public test {status}", time.monotonic() - start, diagnostic)


def snapshot(agent):
    return (
        copy.deepcopy(agent._anthropic_messages),
        agent.current_turns, agent.total_input_tokens, agent.total_output_tokens,
    )


def restore(agent, state):
    messages, agent.current_turns, agent.total_input_tokens, agent.total_output_tokens = state
    agent._anthropic_messages = copy.deepcopy(messages)


def error_label(error):
    status = getattr(error, "status_code", None)
    return f"{type(error).__name__}:{status}" if status else type(error).__name__


def make_row(case, condition, arm, files, duration, tokens, attempts, error):
    groups, _ = probe(case, files, (PUBLIC, "hidden", "regression"))
    source_diff = []
    for name, source in files.items():
        source_diff.extend(difflib.unified_diff(
            case["files"][name].splitlines(), source.splitlines(),
            fromfile=f"a/{name}.py", tofile=f"b/{name}.py", lineterm="",
        ))
    return {
        "id": case["id"], "condition": condition, "arm": arm, "error": error,
        "resolved": error is None and all(s == "passed" for statuses in groups.values() for s in statuses),
        "tests": groups,
        "verification": [{"status": item.status, "seconds": round(item.seconds, 3)}
                         for item in attempts],
        "seconds": round(duration, 3), "tokens": tokens,
        "patch": "\n".join(source_diff),
    }


async def run_case(case, config, conditions):
    public = "\n".join(
        f"{case['entry']}.{case['func']}(*{args!r}) should return {expected!r}"
        for args, expected in case[PUBLIC]
    )
    files = ", ".join(f"{name}.py" for name in case["files"])
    prompt = f"{case['issue']}\nEditable files: {files}\nPublic reproducer:\n{public}"
    first = RepairAgent(case, case["files"], **config)
    error, answer = None, ""
    start = time.monotonic()
    try:
        answer = await asyncio.wait_for(first.chat(prompt), timeout=45)
        if first._aborted:
            error = "interrupted_or_timed_out"
    except Exception as exc:
        error = error_label(exc)
    finally:
        await first.shutdown()
    initial_seconds = time.monotonic() - start
    state = snapshot(first)
    rows = []
    for condition in conditions:
        start_files = case["files"] if condition == "discard_first_patch" else first.files
        baseline = make_row(case, condition, "baseline", start_files, initial_seconds,
                            first.get_token_usage(), [], error)
        if error is not None:
            rows += [baseline, dict(baseline, arm="feedback")]
            continue
        agent = RepairAgent(case, start_files, **config)
        restore(agent, state)
        agent.replayed_answer = answer
        attempts, branch_error = [], None
        start = time.monotonic()
        try:
            _, attempts = await asyncio.wait_for(
                agent.repair_with_verification(prompt, TARGET, retries=RETRY_LIMIT), timeout=45,
            )
            if agent._aborted:
                branch_error = "interrupted_or_timed_out"
        except Exception as exc:
            branch_error = error_label(exc)
        finally:
            await agent.shutdown()
        rows += [baseline, make_row(case, condition, "feedback", agent.files,
                                    initial_seconds + time.monotonic() - start,
                                    agent.get_token_usage(), attempts, branch_error)]
    return rows


def summary(rows):
    data = {}
    for condition in sorted({r["condition"] for r in rows}):
        group = [r for r in rows if r["condition"] == condition]
        for arm in ("baseline", "feedback"):
            selected = [r for r in group if r["arm"] == arm]
            f2p = [s for row in selected for key in (PUBLIC, "hidden") for s in row["tests"][key]]
            p2p = [s for row in selected for s in row["tests"]["regression"]]
            data.setdefault(condition, {})[arm] = {
                "tasks": len(selected), "resolved": sum(row["resolved"] for row in selected),
                "f2p_passed": f2p.count("passed"), "f2p_total": len(f2p),
                "p2p_passed": p2p.count("passed"), "p2p_total": len(p2p),
                "regressions": sum("failed" in row["tests"]["regression"] for row in selected),
                "unverified": sum(any("unverified" in tests for tests in row["tests"].values())
                                  for row in selected),
                "errors": sum(row["error"] is not None for row in selected),
                "median_seconds": round(statistics.median(row["seconds"] for row in selected), 3),
                "total_tokens": sum(sum(row["tokens"].values()) for row in selected),
            }
        pairs = {r["id"]: {} for r in group}
        for row in group:
            pairs[row["id"]][row["arm"]] = row["resolved"]
        data[condition]["paired"] = {
            "recovered": sum(not p["baseline"] and p["feedback"] for p in pairs.values()),
            "lost": sum(p["baseline"] and not p["feedback"] for p in pairs.values()),
        }
    return data


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--case", action="append")
    parser.add_argument("--condition", choices=("normal", "discard_first_patch"), action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        if len(cases) != len(set(args.case)):
            parser.error("unknown case ID")
    preflight(cases)
    if args.check_only:
        return
    if not args.env_file or not args.output or args.output.exists() != args.resume:
        parser.error("provide --env-file and a new --output path, or use --resume with an existing output")
    config = dotenv_values(args.env_file, interpolate=False)
    base = (config.get("API") or "").rstrip("/")
    if base != "https://api.deepseek.com/anthropic" or not config.get("APIKEY"):
        parser.error("official DeepSeek Anthropic endpoint and key are required")
    model = config.get("MODEL") or "deepseek-flash"
    provider = {"anthropic_base_url": base, "api_key": config["APIKEY"], "model": model}
    conditions = args.condition or ["normal", "discard_first_patch"]
    if len(conditions) != len(set(conditions)):
        parser.error("duplicate condition")
    result = {
        "dataset": "10 authored Python issues, two source modules per task",
        "model": model, "wasi_python": importlib.metadata.version("wasi-python"),
        "wasmtime": importlib.metadata.version("wasmtime"),
        "tool_round_budget": TOOL_BUDGET, "feedback_retries": RETRY_LIMIT,
        "conditions": conditions, "case_ids": [case["id"] for case in cases], "rows": [],
    }
    if args.resume:
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in result.items() if key != "rows"):
            parser.error("output metadata does not match this run")
        result["rows"] = previous["rows"]
        result["summary"] = previous["summary"]
    completed = {row["id"] for row in result["rows"]}
    for case in cases:
        if case["id"] in completed:
            continue
        with redirect_stdout(io.StringIO()):
            rows = await run_case(case, provider, conditions)
        result["rows"].extend(rows)
        result["summary"] = summary(result["rows"])
        partial = args.output.with_suffix(args.output.suffix + ".tmp")
        partial.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        partial.replace(args.output)
        print(case["id"], [(r["condition"], r["arm"], r["resolved"], r["error"]) for r in rows], flush=True)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
