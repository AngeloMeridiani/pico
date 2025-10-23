#!/usr/bin/env python3

"""
Quick smoke test for the dynamic collective selector integrated in schedgen.

The test replays a tiny 4-rank trace twice:
  1. With the built-in defaults (no rules file) to ensure we see the fallback
     algorithms (binomial bcast, recursive_doubling allreduce).
  2. With an explicit rules file that switches small broadcasts to the
     'binary' tree and medium allreduces to 'ring', proving the selector honours
     user overrides.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run_schedgen(args: list[str]) -> str:
    """Run schedgen with the given CLI arguments and return stdout."""
    result = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    schedgen_bin = repo_root / "schedgen" / "bin" / "schedgen"
    trace_file = (
        repo_root
        / "schedgen"
        / "liballprof-samples"
        / "sweep3d-2x2"
        / "pmpi-trace-rank-0.txt"
    )

    if not schedgen_bin.exists():
        print(f"Missing schedgen binary: {schedgen_bin}", file=sys.stderr)
        return 1
    if not trace_file.exists():
        print(f"Missing sample trace: {trace_file}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        default_stdout = run_schedgen(
            [
                str(schedgen_bin),
                "--ptrn=trace",
                "--traces",
                str(trace_file),
                "--filename",
                str(tmpdir / "default.goal"),
                "--traces-nops",
                "16",
                "--selector-debug",
            ]
        )

        if "[selector] bcast -> binomial" not in default_stdout:
            raise AssertionError("Expected binomial broadcast in default run.")
        if "[selector] allreduce -> recursive_doubling" not in default_stdout:
            raise AssertionError(
                "Expected recursive_doubling allreduce in default run."
            )
        if "[selector] allreduce -> ring" in default_stdout:
            raise AssertionError(
                "Unexpected ring allreduce without selector overrides."
            )

        rules_path = tmpdir / "selector.rules"
        rules_path.write_text(
            "\n".join(
                [
                    "bcast binary 0 * 0 32",
                    "bcast binomial 0 * 33 *",
                    "allreduce recursive_doubling 0 * 0 4",
                    "allreduce ring 0 * 5 *",
                ]
            )
            + "\n",
            encoding="ascii",
        )

        custom_stdout = run_schedgen(
            [
                str(schedgen_bin),
                "--ptrn=trace",
                "--traces",
                str(trace_file),
                "--filename",
                str(tmpdir / "custom.goal"),
                "--traces-nops",
                "16",
                "--selector-debug",
                "--selector-rules",
                str(rules_path),
            ]
        )

        if "[selector] bcast -> binary" not in custom_stdout:
            raise AssertionError(
                "Expected binary broadcast with selector overrides."
            )
        if "[selector] allreduce -> ring" not in custom_stdout:
            raise AssertionError(
                "Expected ring allreduce with selector overrides."
            )
        if "[selector] bcast -> binomial" not in custom_stdout:
            raise AssertionError(
                "Expected binomial broadcast for large messages even with overrides."
            )

    print("Dynamic selector test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
