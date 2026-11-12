"""Paired ablation of retry opportunity and test diagnostics on Python repair tasks."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

from dotenv import dotenv_values

from run_open_repair import (
    DATASET, PUBLIC, TARGET, TOOL_BUDGET, RepairAgent, error_label, make_row,
    preflight, probe, restore, snapshot,
)


EXTRA = Path(__file__).with_name("ablation_extra_cases.json")
ARMS = ("no_retry", "status_only", "diagnostic")
TIMEOUT = 55
SEED = 20261112
BOOTSTRAP = 10000


class AblationAgent(RepairAgent):
    """Keep the production verification flow and selectively suppress its diagnostic text."""

    def __init__(self, case, files, *, include_diagnostic, **config):
        super().__init__(case, files, **config)
        self.include_diagnostic = include_diagnostic

    async def chat(self, user_message, *, runtime_feedback=False):
        if runtime_feedback and not self.include_diagnostic:
            user_message = user_message.partition("\n\n<pytest-output>")[0]
        return await super().chat(user_message, runtime_feedback=runtime_feedback)


def load_cases():
    original = json.loads(DATASET.read_text(encoding="utf-8"))
    extra = json.loads(EXTRA.read_text(encoding="utf-8"))
    cases = original + extra
    if len(original) != 10 or len(extra) != 20 or len({case["id"] for case in cases}) != 30:
        raise ValueError("expected 30 distinct tasks, including 20 with incomplete patches")
    return cases


def validate(cases):
    preflight(cases)
    issues = []
    for case in cases:
        if "fault" not in case:
            continue
        statuses, _ = probe(case, {**case["files"], **case["fault"]}, (PUBLIC, "regression"))
        if sorted(statuses[PUBLIC]) != ["failed", "passed"] or set(statuses["regression"]) != {"passed"}:
            issues.append(f"{case['id']}: {statuses}")
    if issues:
        raise ValueError("invalid incomplete patches\n" + "\n".join(issues))
    print("Incomplete patches checked: each fixes one public check and leaves another failing.", flush=True)


def instruction(case):
    files = ", ".join(f"{name}.py" for name in case["files"])
    return f"{case['issue']}\nEditable files: {files}\nImplement the requested behavior."


def stamp(row, initial, eligible):
    row["public_before_retry"] = initial
    row["eligible_for_retry"] = eligible
    row["retried"] = len(row["verification"]) > 1
    return row


async def branches(case, condition, config, prompt, files, state, answer, seconds, tokens, error):
    initial, _ = probe(case, files, (PUBLIC,))
    initial = initial[PUBLIC]
    eligible = error is None and "failed" in initial and "unverified" not in initial and state[1] < TOOL_BUDGET
    baseline = make_row(case, condition, "no_retry", files, seconds, tokens, [], error)
    stamp(baseline, initial, eligible)
    if not eligible:
        return [baseline] + [dict(baseline, arm=arm) for arm in ARMS[1:]]

    async def retry(arm):
        agent = AblationAgent(case, files, include_diagnostic=arm == "diagnostic", **config)
        restore(agent, state)
        agent.replayed_answer = answer
        attempts, retry_error = [], None
        started = time.monotonic()
        try:
            _, attempts = await asyncio.wait_for(
                agent.repair_with_verification(prompt, TARGET, retries=1), timeout=TIMEOUT,
            )
            if agent._aborted:
                retry_error = "interrupted_or_timed_out"
        except Exception as exc:
            retry_error = error_label(exc)
        finally:
            await agent.shutdown()
        row = make_row(case, condition, arm, agent.files, seconds + time.monotonic() - started,
                       agent.get_token_usage(), attempts, retry_error)
        return stamp(row, initial, eligible)

    # Start both otherwise-identical retries together to reduce time-of-run effects.
    return [baseline, *(await asyncio.gather(*(retry(arm) for arm in ARMS[1:])))]


async def run_case(case, config):
    prompt = instruction(case)
    first = RepairAgent(case, case["files"], **config)
    answer, error = "", None
    started = time.monotonic()
    try:
        answer = await asyncio.wait_for(first.chat(prompt), timeout=TIMEOUT)
        if first._aborted:
            error = "interrupted_or_timed_out"
    except Exception as exc:
        error = error_label(exc)
    finally:
        await first.shutdown()
    seconds = time.monotonic() - started
    observed = await branches(case, "normal", config, prompt, first.files, snapshot(first),
                              answer, seconds, first.get_token_usage(), error)

    if "fault" not in case:
        return observed
    # The controlled workload starts from an authored incomplete patch and one fixed message history.
    initial_answer = "I applied an initial patch for the requested behavior."
    fixed_history = [{"role": "user", "content": prompt}, {"role": "assistant", "content": initial_answer}]
    controlled = await branches(case, "incomplete_patch", config, prompt,
                                {**case["files"], **case["fault"]}, (fixed_history, 0, 0, 0),
                                initial_answer, 0, {"input": 0, "output": 0}, None)
    return observed + controlled


def paired_effect(left, right):
    deltas = [int(right[key]["resolved"]) - int(left[key]["resolved"]) for key in sorted(left)]
    wins, losses = deltas.count(1), deltas.count(-1)
    n = len(deltas)
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
            / 2 ** discordant)
    rng = random.Random(SEED)
    samples = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(BOOTSTRAP))
    return {
        "wins": wins, "losses": losses, "delta_pp": round(100 * (wins - losses) / n, 2),
        "bootstrap_95_pp": [round(100 * samples[int(BOOTSTRAP * q)], 2) for q in (0.025, 0.975)],
        "mcnemar_exact_p": round(p, 6),
    }


def summarize(rows):
    report = {}
    for condition in ("normal", "incomplete_patch"):
        selected = [row for row in rows if row["condition"] == condition]
        if not selected:
            continue
        arms = {arm: {row["id"]: row for row in selected if row["arm"] == arm} for arm in ARMS}
        metrics = {}
        for arm, cases in arms.items():
            values = list(cases.values())
            repair = [s for row in values for group in (PUBLIC, "hidden") for s in row["tests"][group]]
            keep = [s for row in values for s in row["tests"]["regression"]]
            metrics[arm] = {
                "n": len(values), "resolved": sum(row["resolved"] for row in values),
                "repair_tests": [repair.count("passed"), len(repair)],
                "regression_tests": [keep.count("passed"), len(keep)],
                "regression_tasks": sum("failed" in row["tests"]["regression"] for row in values),
                "errors": sum(row["error"] is not None for row in values),
                "unverified_tasks": sum(
                    any("unverified" in tests for tests in row["tests"].values()) for row in values),
                "eligible": sum(row["eligible_for_retry"] for row in values),
                "retried": sum(row["retried"] for row in values),
                "median_seconds": round(statistics.median(row["seconds"] for row in values), 3),
                "tokens": sum(sum(row["tokens"].values()) for row in values),
            }
        comparisons = [
            ("status_vs_no_retry", "no_retry", "status_only"),
            ("diagnostic_vs_status", "status_only", "diagnostic"),
            ("diagnostic_vs_no_retry", "no_retry", "diagnostic"),
        ]
        report[condition] = {
            "arms": metrics,
            "paired": {label: paired_effect(arms[left], arms[right]) for label, left, right in comparisons},
        }
    return report


def checkpoint(path, result):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


async def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--env-file", type=Path)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--check-only", action="store_true")
    args = cli.parse_args()
    cases = load_cases()
    validate(cases)
    if args.check_only:
        return
    if not args.env_file or not args.output or args.output.exists() != args.resume:
        cli.error("provide --env-file and a new --output, or use --resume with an existing output")
    env = dotenv_values(args.env_file, interpolate=False)
    base = (env.get("API") or "").rstrip("/")
    if base != "https://api.deepseek.com/anthropic" or not env.get("APIKEY"):
        cli.error("the official DeepSeek Anthropic endpoint and key are required")
    model = env.get("MODEL") or "deepseek-flash"
    config = {"anthropic_base_url": base, "model": model, "api_key": env["APIKEY"]}
    result = {
        "protocol": 1, "model": model, "tool_budget": TOOL_BUDGET, "timeout_seconds": TIMEOUT,
        "workers": args.workers, "seed": SEED, "bootstrap_samples": BOOTSTRAP,
        "case_ids": [case["id"] for case in cases],
        "fixture_hashes": {p.name: sha256(p.read_bytes()).hexdigest() for p in (DATASET, EXTRA)},
        "rows": [],
    }
    if args.resume:
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in result.items() if key != "rows"):
            cli.error("saved metadata does not match")
        result["rows"] = previous["rows"]
    completed = {row["id"] for row in result["rows"]}
    slots = asyncio.Semaphore(args.workers)

    async def job(case):
        async with slots:
            return case["id"], await run_case(case, config)

    jobs = [asyncio.create_task(job(case)) for case in cases if case["id"] not in completed]
    try:
        with redirect_stdout(io.StringIO()):
            for future in asyncio.as_completed(jobs):
                key, rows = await future
                result["rows"].extend(rows)
                result["summary"] = summarize(result["rows"])
                checkpoint(args.output, result)
                print(key, [(r["condition"], r["arm"], r["resolved"], r["error"]) for r in rows],
                      file=sys.__stdout__, flush=True)
    finally:
        for job_task in jobs:
            if not job_task.done():
                job_task.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
    if len(result["rows"]) != 150:
        raise RuntimeError("expected 150 task-condition-arm results")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
