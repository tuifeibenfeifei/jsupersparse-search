# -*- coding: utf-8 -*-
"""check_runtime.py - runtime check: can the code actually start?

Division of labour with audit_repo.py (the static check): this script really
executes the code, so it catches problems that only appear at run time rather
than at compile time.

Three kinds of checks, each with the correct protocol so that nothing is
reported as a false positive:

A. Package modules (src/**): verified through a real package import
   (import src.xxx). They must not be loaded as top-level modules, because
   relative imports then necessarily raise ImportError.

B. CLI scripts (source contains argparse): run --help to verify the argument
   definitions are complete.

C. Experiment scripts (no argparse): these call the full experiment directly
   under __main__, so --help is ignored and the script starts hours of
   computation. They are therefore not given --help; instead the module is
   imported, which only verifies that the dependencies are present and that
   no module-level code raises.

Whether a script belongs to B or C is decided by the presence of argparse in
its source, not by guessing.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERBOSE = "-v" in sys.argv

# Importing these starts long computations or reads large files, so they are
# skipped to keep the check itself from running for hours.
SLOW_IMPORT = {
    "supplement_stats/peerj_wallclock.py",
    "supplement_stats/peerj_resume_mc.py",
    "supplement_stats/peerj_proxy_only.py",
    "supplement_stats/mc_paired_audit.py",
    "supplement_stats/chl_sensitivity_scan.py",
    "supplement_stats/perm_control_rerun.py",
    "results/ablation_cross/run_cross_ablation.py",
    "species-search/run_framework.py",
    "species-search/run_framework_v3.py",
    "species-search/run_framework_sparse.py",
    "species-search/run_framework_features.py",
    "species-search/run_adaptive_species.py",
    "species-search/run_cluster_ablation.py",
    "species-search/run_ablation_species.py",
}

PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")


def run(args, timeout=90):
    try:
        p = subprocess.run(
            args, cwd=str(ROOT), capture_output=True, text=True,
            timeout=timeout, env=ENV, encoding="utf-8", errors="replace",
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "", ""


def why(stderr):
    for key, label in [
        ("ModuleNotFoundError", "missing module"),
        ("ImportError", "import error"),
        ("SyntaxError", "syntax error"),
        ("FileNotFoundError", "missing data file"),
        ("AttributeError", "attribute error"),
        ("TypeError", "type error"),
        ("ValueError", "value error"),
        ("KeyError", "key error"),
    ]:
        if key in stderr:
            return label
    return "other"


def last_line(s):
    lines = [l for l in s.strip().split("\n") if l.strip()]
    return lines[-1][:160] if lines else ""


def main():
    allpy = [
        p for p in ROOT.rglob("*.py")
        if ".git" not in p.parts and "__pycache__" not in p.parts
        and p.name not in ("check_runtime.py", "audit_repo.py")
    ]
    rels = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in allpy)

    pkg = [r for r in rels if r.startswith("src/")]
    cli, runner = [], []
    for r in rels:
        if r.startswith("src/"):
            continue
        src = io.open(ROOT / r, encoding="utf-8-sig").read()
        (cli if "argparse" in src else runner).append(r)

    print(f"python {sys.version.split()[0]} | total .py = {len(rels)}")
    print(f"  package modules (src/..) : {len(pkg)}")
    print(f"  CLI scripts (argparse)   : {len(cli)}")
    print(f"  experiment runners       : {len(runner)}")
    print()

    fails = []

    # ---------- A. package modules: real package import ----------
    print("=" * 78)
    print("A) import package modules as real packages (src.xxx)")
    print("=" * 78)
    bad_a = 0
    for r in pkg:
        mod = r[:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        rc, out, err = run([PY, "-c", f"import {mod}"], timeout=60)
        if rc != 0:
            bad_a += 1
            print(f"  [FAIL] {mod}: {why(err)}: {last_line(err)}")
            fails.append(f"pkg import: {mod}")
    print(f"failures: {bad_a}/{len(pkg)}")
    if not bad_a:
        print("  all package modules import cleanly")

    # ---------- B. CLI scripts: --help ----------
    print()
    print("=" * 78)
    print("B) --help on CLI scripts (argparse definitions)")
    print("=" * 78)
    bad_b = 0
    for r in cli:
        rc, out, err = run([PY, r, "--help"], timeout=90)
        if rc != 0:
            bad_b += 1
            print(f"  [FAIL] {r} (rc={rc}): {why(err)}: {last_line(err)}")
            fails.append(f"--help: {r}")
    print(f"failures: {bad_b}/{len(cli)}")
    if not bad_b:
        print("  all CLI scripts accept --help")

    # ---------- C. experiment scripts: import check ----------
    print()
    print("=" * 78)
    print("C) import experiment runners (deps installed? module-level code sane?)")
    print("=" * 78)
    bad_c, skipped = 0, 0
    for r in runner:
        if r in SLOW_IMPORT:
            skipped += 1
            continue
        code = (
            "import importlib.util,sys,os\n"
            f"p={str(ROOT / r)!r}\n"
            "d=os.path.dirname(p); sys.path.insert(0,d)\n"
            "sys.path.insert(0,os.getcwd())\n"
            "spec=importlib.util.spec_from_file_location('_probe',p)\n"
            "m=importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
        )
        rc, out, err = run([PY, "-c", code], timeout=90)
        if rc != 0:
            bad_c += 1
            print(f"  [FAIL] {r}: {why(err)}")
            print(f"         {last_line(err)}")
            fails.append(f"runner import: {r}")
    print(f"failures: {bad_c}/{len(runner) - skipped} (skipped slow: {skipped})")

    # ---------- summary ----------
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"package import failures : {bad_a}")
    print(f"--help failures         : {bad_b}")
    print(f"runner import failures  : {bad_c}")
    print()
    for f in fails:
        print(f"  [FAIL] {f}")
    print()
    print("RESULT:", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
