#!/usr/bin/env python3
"""Compile the pinned, dependency-free proof file and audit every theorem's axioms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
MODULES = {"Appendix.lean": "PEID", "Counterexamples.lean": "PEIDCounterexample"}
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def main() -> int:
    version = subprocess.run(
        ["lean", "--version"], cwd=HERE, check=True, text=True, capture_output=True
    ).stdout.strip()
    if "version 4.29.0," not in version:
        raise RuntimeError(f"Unexpected Lean version: {version}")
    modules = {}
    logs = []
    for filename, namespace in MODULES.items():
        path = HERE / filename
        source = path.read_text()
        theorems = re.findall(r"^theorem\s+(\w+)", source, re.MULTILINE)
        requested = re.findall(rf"^#print axioms {namespace}\.(\w+)", source, re.MULTILINE)
        if sorted(theorems) != sorted(requested):
            raise RuntimeError("Every theorem must have exactly one axiom audit")
        if re.search(r"\b(?:sorry|admit)\b|^\s*(?:private\s+)?axiom\b", source, re.MULTILINE):
            raise RuntimeError("Unfinished proof or custom axiom declaration")

        run = subprocess.run(["lean", filename], cwd=HERE, text=True, capture_output=True)
        log = run.stdout + run.stderr
        logs.append(f"{filename}\n{log}")
        (HERE / "verification.log").write_text("\n".join(logs))
        if run.returncode or re.search(r"\b(?:error|warning):", log):
            print(log, end="")
            raise RuntimeError("Compilation must finish without errors or warnings")

        audited = {}
        for name, names in re.findall(rf"'{namespace}\.(\w+)' depends on axioms: \[([^\]]*)\]", log):
            axioms = {x.strip() for x in names.split(",") if x.strip()}
            if not axioms <= ALLOWED_AXIOMS:
                raise RuntimeError(f"Unexpected axiom dependencies for {name}: {axioms}")
            audited[name] = sorted(axioms)
        for name in re.findall(rf"'{namespace}\.(\w+)' does not depend on any axioms", log):
            audited[name] = []
        if set(audited) != set(theorems):
            raise RuntimeError("Incomplete theorem audit output")
        modules[filename] = {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "theorem_count_including_helpers": len(theorems),
            "axiom_audit": audited,
        }

    status = {
        "checked_date": "2026-10-27",
        "lean_version": version,
        "theorem_count_including_helpers": sum(m["theorem_count_including_helpers"] for m in modules.values()),
        "errors": 0,
        "warnings": 0,
        "modules": modules,
        "scope": "Algebraic identities and conditional information-theoretic inference steps; not a complete probability/entropy formalization.",
    }
    (HERE / "verification.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"PASS: {status['theorem_count_including_helpers']} theorems, 0 errors, 0 warnings; standard Lean axioms only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
