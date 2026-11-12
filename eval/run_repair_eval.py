"""Paired DeepSeek patch-selection evaluation. This runner never executes model output."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import dotenv_values
from agents.agent import Agent
from agents.verification import Verification

TARGET = "test_target.py::test_target"
TOOL = {
    "name": "submit_patch",
    "description": "Submit a patch option from the displayed choices. Its test result is not known until verification.",
    "input_schema": {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": ["A", "B", "C"]}},
        "required": ["choice"],
        "additionalProperties": False,
    },
}


class EvalAgent(Agent):
    def __init__(self, choices, oracle, *, fault_injection, **kwargs):
        super().__init__(**kwargs)
        self.choices = choices
        self.oracle = oracle
        self.patch = "implementation"
        self.submissions = []
        self.fault_injection = fault_injection
        self.dropped_first_submission = False

    async def _execute_tool_call(self, name, inp):
        choice = inp.get("choice") if isinstance(inp, dict) else None
        if name != TOOL["name"] or not isinstance(choice, str) or choice not in self.choices:
            return "Error: submit exactly one of A, B, C."
        self.submissions.append(choice)
        if self.fault_injection and len(self.submissions) == 1:
            self.dropped_first_submission = True
        else:
            self.patch = self.choices[choice]
        return f"Option {choice} submitted."

    async def _verify_repair_target(self, node_id, *, root, python, timeout):
        record = self.oracle[self.patch]
        return Verification(node_id, record["target"], record["reason"], 0, record["diagnostic"])


async def run_case(case, choices, mode, *, oracle, model_config, budget, fault_injection):
    attempts = []
    answer = ""
    error = None
    start = time.monotonic()
    alternatives = "\n\n".join(f"{label}:\n{case[patch]}" for label, patch in choices.items())
    prompt = (
        f"Fix this bug in {case['module']}: {case['issue']}\n"
        f"The reproducing case is {TARGET}:\n{case['target']}\n"
        f"Review the patch options and call submit_patch with one choice per repair attempt.\n{alternatives}"
    )
    with redirect_stdout(io.StringIO()):
        agent = EvalAgent(
            choices, oracle, fault_injection=fault_injection, **model_config, permission_mode="dontAsk",
            custom_tools=[TOOL], is_sub_agent=True, max_turns=budget,
            custom_system_prompt="You are evaluating synthetic Python repairs. Submit one available patch "
            "option for the current issue. If the runtime reports a failed test, select a better option.",
        )
        try:
            if mode == "baseline":
                answer = await asyncio.wait_for(agent.chat(prompt), timeout=55)
            else:
                answer, attempts = await asyncio.wait_for(
                    agent.repair_with_verification(prompt, TARGET, retries=1), timeout=55,
                )
            if agent._aborted:
                error = "interrupted_or_timed_out"
        except Exception as exc:
            error = type(exc).__name__
            if getattr(exc, "status_code", None):
                error += f":HTTP {exc.status_code}"
        finally:
            await agent.shutdown()
    record = oracle[agent.patch]
    return {
        "case": case["id"], "mode": mode, "choices": choices,
        "resolved": error is None and record["target"] == record["regression"] == "passed",
        "target": record["target"], "regression": record["regression"],
        "verification": [attempt.status for attempt in attempts],
        "verification_misses_regression": bool(attempts and attempts[-1].status == "passed"
                                               and record["regression"] != "passed"),
        "submissions": agent.submissions, "dropped_first_submission": agent.dropped_first_submission,
        "tokens": agent.get_token_usage(),
        "seconds": round(time.monotonic() - start, 3), "error": error,
        "assistant_summary": answer,
    }


def summarize(rows):
    return {
        "tasks": len(rows),
        "resolved": sum(row["resolved"] for row in rows),
        "target_passed": sum(row["target"] == "passed" for row in rows),
        "regressions_failed": sum(row["regression"] == "failed" for row in rows),
        "errors": sum(row["error"] is not None for row in rows),
        "feedback_retries": sum(len(row["verification"]) > 1 for row in rows),
        "retry_recoveries": sum(row["resolved"] and row["verification"] == ["failed", "passed"] for row in rows),
        "target_passed_but_regression_not_passed": sum(row["verification_misses_regression"] for row in rows),
        "median_seconds": round(statistics.median(row["seconds"] for row in rows), 3),
        "input_tokens": sum(row["tokens"]["input"] for row in rows),
        "output_tokens": sum(row["tokens"]["output"] for row in rows),
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", help="Evaluate only this case ID (repeatable)")
    parser.add_argument("--fault-injection", action="store_true",
                        help="Simulate the first submitted patch failing to apply in both arms")
    args = parser.parse_args()

    config = dotenv_values(args.env_file, interpolate=False)
    base = (config.get("API") or "").rstrip("/")
    if base != "https://api.deepseek.com/anthropic" or not config.get("APIKEY"):
        parser.error("the evaluation needs the official DeepSeek Anthropic-compatible endpoint and key")
    model = config.get("MODEL") or "deepseek-flash"
    model_config = {"anthropic_base_url": base, "model": model, "api_key": config["APIKEY"]}
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists():
        parser.error("choose a fresh output path to keep prior results")
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(__file__).with_name("repair_cases.json").read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        if len(cases) != len(set(args.case)):
            parser.error("unknown case ID")
    budget = 6
    rng = random.Random(14)
    report = {
        "model": model, "max_tool_rounds": budget, "feedback_retries": 1,
        "dataset": "local synthetic Python repair fixtures, fixed-choice, offline pytest oracle",
        "fault_injection": args.fault_injection,
        "case_ids": [case["id"] for case in cases], "results": [], "summary": {},
    }
    for index, case in enumerate(cases):
        patches = ["implementation", "correct", "shortcut"]
        rng.shuffle(patches)
        choices = dict(zip("ABC", patches))
        modes = ("baseline", "feedback") if index % 2 == 0 else ("feedback", "baseline")
        for mode in modes:
            row = await run_case(case, choices, mode, oracle=oracle[case["id"]],
                                 model_config=model_config, budget=budget, fault_injection=args.fault_injection)
            report["results"].append(row)
            report["summary"] = {arm: summarize([r for r in report["results"] if r["mode"] == arm])
                                 for arm in modes if any(r["mode"] == arm for r in report["results"])}
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output)
            print(f"{case['id']} {mode}: resolved={row['resolved']} "
                  f"verification={row['verification']} error={row['error']}", flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
