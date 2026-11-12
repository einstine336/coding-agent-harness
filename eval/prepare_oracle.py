"""Prepare the repair evaluation oracle by running pytest on fixed synthetic fixtures."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.verification import verify_pytest

TARGET = "test_target.py::test_target"
REGRESSION = "test_regression.py::test_regression"
EXPECTED = {
    "implementation": ("failed", "passed"),
    "correct": ("passed", "passed"),
    "shortcut": ("passed", "failed"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-python", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    cases = json.loads(Path(__file__).with_name("repair_cases.json").read_text(encoding="utf-8"))
    records = {}
    for case in cases:
        records[case["id"]] = {}
        for patch, expected in EXPECTED.items():
            with tempfile.TemporaryDirectory(prefix="bear-oracle-") as directory:
                root = Path(directory)
                (root / case["module"]).write_text(case[patch], encoding="utf-8")
                (root / "test_target.py").write_text(case["target"], encoding="utf-8")
                (root / "test_regression.py").write_text(case["regression"], encoding="utf-8")
                target = verify_pytest(TARGET, cwd=root, python=args.test_python, timeout=8)
                regression = verify_pytest(REGRESSION, cwd=root, python=args.test_python, timeout=8)
                observed = (target.status, regression.status)
                if observed != expected:
                    raise SystemExit(f"{case['id']} {patch}: expected {expected}; got {observed}\n{target.output}\n{regression.output}")
                records[case["id"]][patch] = {
                    "target": target.status, "regression": regression.status,
                    "reason": target.reason,
                    "diagnostic": target.output.replace(str(root), "<fixture>") if target.status == "failed" else "",
                }
        print(f"{case['id']}: three patches, six outcomes checked", flush=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(records, out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
