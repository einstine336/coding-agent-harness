"""Integration checks for the verifier and a paired, scripted retry ablation."""

import asyncio
from contextlib import chdir
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from agents.agent import Agent
from agents.verification import verify_pytest

TEST_PYTHON = os.environ.get("BEAR_TEST_PYTHON", sys.executable)
NODE = "test_target.py::test_target"


class ScriptedAgent(Agent):
    def __init__(self, initial=None, repair=None, *, mode="dontAsk", interrupted=False):
        super().__init__(api_base="http://127.0.0.1:1/v1", model="test-stub", api_key="offline",
                         permission_mode=mode, custom_system_prompt="offline test", is_sub_agent=True)
        self.initial = initial
        self.repair = repair
        self.interrupted = interrupted
        self.requests = []

    async def chat(self, message, *, runtime_feedback=False):
        self.requests.append((message, runtime_feedback))
        self._aborted = self.interrupted
        patch = self.repair if runtime_feedback else self.initial
        if patch:
            path, source = patch
            Path(path).write_text(source, encoding="utf-8")
        return "Done (scripted)"


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bear-verify-check-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def verify(self, content, node=NODE, **overrides):
        (self.root / "test_target.py").write_text(content, encoding="utf-8")
        return verify_pytest(node, cwd=self.root, python=overrides.pop("python", TEST_PYTHON), **overrides)

    def test_assertion_pass(self):
        self.assertEqual(self.verify("def test_target(): assert 1 == 1\n").status, "passed")

    def test_assertion_failure_includes_diagnostics(self):
        result = self.verify("def test_target(): assert 1 == 2\n")
        self.assertEqual(result.status, "failed")
        self.assertIn("assert 1 == 2", result.output)

    def test_missing_case_is_unverified(self):
        self.assertEqual(self.verify("def test_other(): pass\n").status, "unverified")

    def test_skipped_and_xfail_are_unverified(self):
        for marker in ("skip", "xfail"):
            with self.subTest(marker=marker):
                result = self.verify(f"import pytest\n@pytest.mark.{marker}\ndef test_target(): assert False\n")
                self.assertEqual(result.status, "unverified")
                self.assertIn("skipped or marked xfail", result.reason)

    def test_timeout_is_unverified(self):
        result = self.verify("import time\ndef test_target(): time.sleep(10)\n", timeout=1.5)
        self.assertEqual(result.status, "unverified")
        self.assertIn("timed out", result.reason)

    def test_broken_setup_is_unverified(self):
        result = self.verify("import pytest\n@pytest.fixture\ndef bad(): raise RuntimeError()\n"
                             "def test_target(bad): assert True\n")
        self.assertEqual(result.status, "unverified")
        self.assertIn("did not execute", result.reason)

    def test_interpreter_start_failure_is_unverified(self):
        self.assertEqual(self.verify("def test_target(): pass\n",
                                     python=str(self.root / "missing-python")).status, "unverified")

    def test_selector_must_be_within_working_directory(self):
        self.assertEqual(self.verify("def test_target(): pass\n", "../other.py::test_target").status, "unverified")

    def test_exact_parameterized_id(self):
        source = 'import pytest\n@pytest.mark.parametrize("x", [1, 2])\ndef test_target(x): assert x == 1\n'
        self.assertEqual(self.verify(source, NODE + "[1]").status, "passed")
        self.assertEqual(self.verify(source, NODE + "[2]").status, "failed")
        self.assertEqual(self.verify(source).status, "unverified")

    def test_plan_mode_and_interruption_never_run_tests(self):
        async def run():
            for mode, interrupted in (("plan", False), ("dontAsk", True)):
                agent = ScriptedAgent(mode=mode, interrupted=interrupted)
                try:
                    _, outcomes = await agent.repair_with_verification("fix", NODE, python=TEST_PYTHON)
                    self.assertEqual([item.status for item in outcomes], ["unverified"])
                    self.assertIn("Plan Mode" if mode == "plan" else "interrupted", outcomes[0].reason)
                    self.assertEqual(len(agent.requests), int(interrupted))
                finally:
                    await agent.shutdown()
        with chdir(self.root):
            asyncio.run(run())

    def test_skipped_target_does_not_trigger_retry(self):
        (self.root / "test_target.py").write_text(
            "import pytest\n@pytest.mark.skip\ndef test_target(): pass\n", encoding="utf-8")
        async def run():
            agent = ScriptedAgent()
            try:
                _, outcomes = await agent.repair_with_verification("fix", NODE, python=TEST_PYTHON)
                self.assertEqual([item.status for item in outcomes], ["unverified"])
                self.assertEqual(len(agent.requests), 1)
            finally:
                await agent.shutdown()
        with chdir(self.root):
            asyncio.run(run())

    def test_target_pass_ends_the_workflow(self):
        (self.root / "test_target.py").write_text("def test_target(): pass\n", encoding="utf-8")
        async def run():
            agent = ScriptedAgent()
            try:
                _, outcomes = await agent.repair_with_verification("fix", NODE, python=TEST_PYTHON)
                self.assertEqual([item.status for item in outcomes], ["passed"])
                self.assertEqual(len(agent.requests), 1)
            finally:
                await agent.shutdown()
        with chdir(self.root):
            asyncio.run(run())

    def test_paired_retry_ablation_on_six_fixtures(self):
        cases = json.loads(Path(__file__).with_name("repair_cases.json").read_text(encoding="utf-8"))
        rows = []
        async def run(case, retries):
            module = case["module"]
            Path(module).write_text(case["implementation"], encoding="utf-8")
            Path("test_target.py").write_text(case["target"], encoding="utf-8")
            Path("test_regression.py").write_text(case["regression"], encoding="utf-8")
            agent = ScriptedAgent(repair=(module, case["correct"]))
            try:
                _, outcomes = await agent.repair_with_verification("fix", NODE, python=TEST_PYTHON,
                                                                   retries=retries, timeout=8)
                statuses = [item.status for item in outcomes]
                self.assertEqual(statuses, ["failed", "passed"] if retries else ["failed"])
                if retries:
                    self.assertTrue(agent.requests[1][1])
                    self.assertIn("target test failed", agent.requests[1][0])
                    self.assertIn("FAILED", agent.requests[1][0])
                regression = verify_pytest("test_regression.py::test_regression", python=TEST_PYTHON, timeout=8)
                self.assertEqual(regression.status, "passed")
                return {"case": case["id"], "retry_enabled": bool(retries), "target_attempts": statuses,
                        "regression": regression.status, "resolved": statuses[-1] == regression.status == "passed"}
            finally:
                await agent.shutdown()
        with chdir(self.root):
            for case in cases:
                for retries in (0, 1):
                    with self.subTest(case=case["id"], retries=retries):
                        rows.append(asyncio.run(run(case, retries)))
        self.assertEqual(len(rows), len(cases) * 2)
        if output := os.environ.get("BEAR_CONTROLLED_RESULTS"):
            with Path(output).open("x", encoding="utf-8") as out:
                json.dump({"type": "scripted_retry_ablation", "results": rows}, out, indent=2)


if __name__ == "__main__":
    unittest.main()
