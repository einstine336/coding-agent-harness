"""Run one exact pytest case and classify its observed result."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


@dataclass(frozen=True)
class Verification:
    node_id: str
    status: str
    reason: str
    seconds: float
    output: str = ""

    def summary(self) -> str:
        return f"Verification: {self.status.upper()} | {self.node_id} | {self.seconds:.2f}s | {self.reason}"


def verify_pytest(node_id: str, *, cwd: Path | None = None,
                  python: str | None = None, timeout: float = 30) -> Verification:
    start = time.monotonic()
    root = (cwd or Path.cwd()).resolve()
    node_id = node_id.strip()

    def result(status: str, reason: str, output: str = "") -> Verification:
        return Verification(node_id, status, reason, time.monotonic() - start, output)

    filename, separator, test_name = node_id.partition("::")
    if not separator or not test_name or not filename or filename.startswith("-"):
        return result("unverified", "provide one pytest case as path.py::test_name")
    try:
        path = (root / filename).resolve()
        relative = path.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return result("unverified", "test file must be within the working directory")
    if relative.suffix != ".py" or not path.is_file():
        return result("unverified", "test file not found")
    node_id = f"{relative.as_posix()}::{test_name}"

    with tempfile.TemporaryDirectory(prefix="bear-verify-") as directory:
        report = Path(directory) / "report.json"
        command = [python or sys.executable, str(Path(__file__).resolve()), str(report), node_id]
        try:
            with (Path(directory) / "pytest.log").open("w+b") as log:
                run = subprocess.run(command, cwd=root, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=timeout)
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 4000))
                output = log.read().decode("utf-8", errors="replace").strip()
        except subprocess.TimeoutExpired:
            return result("unverified", f"pytest timed out after {timeout:g}s")
        except OSError as error:
            return result("unverified", f"could not start the test interpreter: {error}")
        try:
            phases = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return result("unverified", "pytest produced no usable report; check the test interpreter", output)

    if not isinstance(phases, dict) or any(
        not isinstance(phase, dict) or phase.get("outcome") not in ("passed", "failed", "skipped")
        or not isinstance(phase.get("xfail"), bool)
        for phase in phases.values()
    ):
        return result("unverified", "pytest produced a malformed report", output)

    if any(phase["outcome"] == "skipped" or phase["xfail"] for phase in phases.values()):
        return result("unverified", "target test skipped or marked xfail", output)
    call = phases.get("call")
    if not call:
        return result("unverified", "target test did not execute its test body", output)
    lifecycle_ok = all(phases.get(phase, {}).get("outcome") == "passed"
                       for phase in ("setup", "teardown"))
    if not lifecycle_ok:
        return result("unverified", "target test did not complete setup and teardown", output)
    if call["outcome"] == "passed" and run.returncode == 0:
        return result("passed", "target test executed and passed", output)
    if call["outcome"] == "failed" and run.returncode == 1:
        return result("failed", "target test executed and failed", output)
    return result("unverified", f"pytest ended with an unexpected outcome (exit {run.returncode})", output)


def _probe(report_path: str, node_id: str) -> int:
    # Executed by the target project's Python, which owns the pytest dependency.
    import pytest

    phases = {}

    class Probe:
        def pytest_runtest_logreport(self, report):
            if report.nodeid == node_id:
                phases[report.when] = {
                    "outcome": report.outcome,
                    "xfail": hasattr(report, "wasxfail"),
                }

    sys.path.insert(0, str(Path.cwd()))
    status = pytest.main(["-q", "--rootdir", str(Path.cwd()), "--", node_id], plugins=[Probe()])
    Path(report_path).write_text(json.dumps(phases), encoding="utf-8")
    return int(status)


if __name__ == "__main__":
    raise SystemExit(_probe(*sys.argv[1:]))
