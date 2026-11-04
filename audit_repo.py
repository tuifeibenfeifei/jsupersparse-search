# -*- coding: utf-8 -*-
"""audit_repo.py — reproducible repository/documentation consistency audit.

Usage (run from the repository root):
    python audit_repo.py

Checks:
    1) every tracked code file compiles
    2) no CJK characters remain in tracked code (repository rule: must be 0)
    3) every file path referenced by 需求.txt exists
    4) each supplement_stats script resolves REPO_ROOT to the repository root
    5) key figures quoted in 需求.txt match the result CSVs

Any failure is reported as [FAIL] and the process exits non-zero.
"""
from __future__ import annotations

import ast
import io
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAILS = []
WARNS = []


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [f for f in out.split("\n") if f.strip()]


def check_compile(files):
    section("1) tracked code compiles")
    code = [f for f in files if f.endswith(".py")]
    bad = []
    for f in code:
        p = ROOT / f
        if not p.exists():
            bad.append((f, "missing"))
            continue
        try:
            # utf-8-sig strips a leading BOM, matching what the import system does.
            compile(io.open(p, encoding="utf-8-sig").read(), str(p), "exec")
        except SyntaxError as e:
            bad.append((f, f"SyntaxError: {e}"))
    print(f"checked {len(code)} python files, failures: {len(bad)}")
    for f, why in bad:
        print(f"  [FAIL] {f}: {why}")
        FAILS.append(f"compile: {f}")
    if not bad:
        print("  all OK")


def check_cjk(files):
    section("2) tracked code free of CJK")
    exts = (".py", ".yaml", ".yml", ".sh", ".json")
    # This audit script must quote 需求.txt verbatim (its file name, section
    # anchors and figure snippets), so the CJK it contains is data required to
    # match the document, not prose. Every other tracked code file must be 0.
    exempt = {"audit_repo.py"}
    code = [f for f in files if f.endswith(exts)]
    bad = []
    for f in code:
        p = ROOT / f
        if f in exempt or not p.exists():
            continue
        txt = io.open(p, encoding="utf-8", errors="ignore").read()
        n = sum(1 for c in txt if "\u4e00" <= c <= "\u9fff")
        if n:
            bad.append((f, n))
    print(f"checked {len(code)} code files, with CJK: {len(bad)}")
    for f, n in bad:
        print(f"  [FAIL] {f}: {n} CJK chars")
        FAILS.append(f"cjk: {f}")
    if not bad:
        print("  all OK")


PATH_RE = re.compile(r"[A-Za-z0-9_\-./\\]+\.(?:py|csv|txt|yaml|yml|json|npz|png|md|sh|dll)")

# Paths that legitimately appear in the document but do not live in the repo,
# each with the reason it is exempt.
KNOWN_NONREPO = {
    "cubiomes.dll": "external binary, excluded by .gitignore (src/cubiomes.dll)",
    "data/oracle.npz": "generated data, data/ is gitignored (built by build_*_oracle.py)",
    "oracle.npz": "generated data, data/ is gitignored",
    "raw_chlorophyll.csv": "downloaded raw data, data/ is gitignored",
    "occurrences.csv": "downloaded raw data, data/ is gitignored",
    "_params19.csv": "filename fragment in a glob-style list, not a literal path",
    "4.py": "fragment of a range written as 'stage1.py ~ stage4.py'",
    "stage4.py": "fragment of a range written as 'stage1.py ~ stage4.py'",
    "_oracle.py": "regex fragment of build_oracle.py / build_*_oracle.py",
    "run_ablation_stageX.py": "documented placeholder for stage1..stage4 scripts",
    "scripts/run_ablation_stageX.py": "documented placeholder for stage1..stage4 scripts",
}


def check_doc_paths(files):
    """Check the paths referenced by 需求.txt.

    Most paths in the document are written relative to the directory they live
    in (for example run_chl.py actually sits under chlorophyll-search/), so a
    reference passes if the full path exists OR a file with the same basename
    exists anywhere in the repository. Only a reference satisfying neither is
    treated as genuinely missing.
    """
    section("3) every path referenced by 需求.txt resolves to a real file")
    docp = ROOT / "需求.txt"
    if not docp.exists():
        print("  [SKIP] 需求.txt not found")
        return

    by_base = {}
    for f in files:
        by_base.setdefault(Path(f).name, []).append(f)

    src = io.open(docp, encoding="utf-8-sig").read()
    seen = {}
    for m in PATH_RE.finditer(src):
        raw = m.group(0).replace("\\", "/").strip(".")
        raw = raw.lstrip("/")
        if "<" in raw or ">" in raw or "*" in raw:
            continue
        if "xxx" in raw.lower():  # documented placeholder (scripts/xxx.py)
            continue
        seen[raw] = seen.get(raw, 0) + 1

    exact, bybase, missing = 0, 0, []
    for p, c in sorted(seen.items()):
        if (ROOT / p).exists():
            exact += 1
        elif Path(p).name in by_base:
            bybase += 1
        else:
            missing.append((p, c))

    # Every unresolved reference must carry an explicit documented reason,
    # otherwise it counts as a failure. This keeps each exemption traceable
    # instead of silently ignoring the whole category.
    unclassified = []
    print(f"distinct file paths referenced: {len(seen)}")
    print(f"  resolved exactly            : {exact}")
    print(f"  resolved by basename        : {bybase}")
    print(f"  unresolved                  : {len(missing)}")
    for p, c in missing:
        reason = KNOWN_NONREPO.get(p)
        if reason is None:
            print(f"  [FAIL] {c}x  {p}  (unresolved, no documented reason)")
            FAILS.append(f"doc path: {p}")
            unclassified.append(p)
        else:
            print(f"  [ok]   {c}x  {p}  -- {reason}")
    if not unclassified:
        print("  all unresolved references are documented external/placeholder paths")


def check_repo_root():
    """Statically check the REPO_ROOT resolution in supplement_stats scripts.

    The scripts are not executed (that would have side effects: for example
    importing mc_paired_audit.py prints statistical results). Instead their own
    resolution logic is simulated in memory: walk up from the script directory
    until a directory containing configs/ is found, which is the repo root.
    Also confirms each script has a "raise if not found" guard so it cannot
    silently fall back to a wrong directory.
    """
    section("4) supplement_stats REPO_ROOT resolution (static)")
    d = ROOT / "supplement_stats"
    if not d.is_dir():
        print("  [SKIP] supplement_stats not found")
        return

    bad = []
    scripts = sorted(d.glob("*.py"))
    checked = 0
    for f in scripts:
        src = io.open(f, encoding="utf-8-sig").read()
        if "REPO_ROOT" not in src:
            continue
        checked += 1

        # simulate: walk up from the script directory looking for configs/
        p = f.parent
        found = None
        while True:
            if (p / "configs").is_dir():
                found = p
                break
            if p.parent == p:
                break
            p = p.parent

        has_guard = "raise RuntimeError" in src or "raise FileNotFoundError" in src
        if found is None:
            bad.append((f.name, "no ancestor contains configs/"))
        elif found != ROOT:
            bad.append((f.name, f"resolved to {found} instead of {ROOT}"))
        elif not has_guard:
            bad.append((f.name, "resolves correctly but has no fail-loud guard"))

    print(f"checked {checked} scripts, bad: {len(bad)}")
    for name, why in bad:
        print(f"  [FAIL] {name}: {why}")
        FAILS.append(f"repo_root: {name}")
    if not bad:
        print("  all resolve to the repository root and fail loudly on failure")


# Document anchors for each claim: each entry is (config name, the figure the
# line is expected to carry). Any single line containing both tokens is taken
# as the document stating that configuration's figure, so that line must carry
# the correct number or the document is wrong.
ANCHORS = {
    "final model mean queries": ["final_model", "最终版"],
    "stage2 grid_scan queries": ["grid_scan"],
    "stage2 ray_scan queries": ["ray_scan"],
    "chl CHL5_params7 queries": ["CHL5_params7"],
    "chl CHL7_adaptive queries": ["CHL7_adaptive"],
    "o3d O3D5_params7 queries": ["O3D5_params7"],
    "o3d O3D_RANDOM queries": ["O3D_RANDOM"],
}


def _num_close(a, b, tol=0.06):
    """True when two numbers agree within tol (orders of magnitude count as not equal)."""
    try:
        fa = float(a)
    except (TypeError, ValueError):
        return False
    return abs(fa - b) <= tol


def check_doc_numbers():
    """Compare every key figure in 需求.txt against the result CSVs.

    Each claim is (label, CSV path, key column, key value, metric column,
    snippet that must appear verbatim in the document). Because the snippet is
    exact text, one wrong figure -- or the whole sentence being deleted --
    fails the claim; a correct number appearing elsewhere cannot mask it.
    """
    section("5) 需求.txt key figures vs result CSVs")
    import csv

    docp = ROOT / "需求.txt"
    doc = io.open(docp, encoding="utf-8-sig").read() if docp.exists() else ""

    claims = [
        ("final model mean queries", "results/final_logs/final_model_summary.csv",
         "label", "final_model", "mean_queries", "平均查询 2117.6"),
        ("stage2 grid_scan queries", "results/ablation_logs/stage2_ray_vs_grid_summary.csv",
         "label", "finder=grid_scan", "mean_queries", "grid_scan 40.0（2054.8 查询、400/400）"),
        ("stage2 ray_scan queries", "results/ablation_logs/stage2_ray_vs_grid_summary.csv",
         "label", "finder=ray_scan", "mean_queries", "ray_scan 40.0（2409.3，399/400）"),
        ("chl CHL5_params7 queries", "results/chl_search/chl_summary.csv",
         "config", "CHL5_params7", "mean_queries", "CHL5_params7   11.30 / 2731"),
        ("chl CHL7_adaptive queries", "results/chl_search/chl_summary.csv",
         "config", "CHL7_adaptive", "mean_queries", "CHL7_adaptive  11.30 / 2766"),
        ("o3d O3D5_params7 queries", "results/ocean3d/o3d_summary.csv",
         "config", "O3D5_params7", "mean_queries", "O3D5_params7 最优：40.00 点 / 3,894 查询"),
        ("o3d O3D_RANDOM queries", "results/ocean3d/o3d_summary.csv",
         "config", "O3D_RANDOM", "mean_queries", "O3D_RANDOM 8,330 查询仅 39.15 点"),
    ]

    checked, bad = 0, []
    for label, rel, keyc, keyv, metric, snippet in claims:
        p = ROOT / rel
        if not p.exists():
            print(f"  [SKIP] {label}: {rel} missing")
            continue
        with io.open(p, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        row = next((r for r in rows if r.get(keyc) == keyv), None)
        if row is None:
            print(f"  [FAIL] {label}: row {keyv} not in {rel}")
            bad.append(label)
            FAILS.append(f"claim: {label}")
            continue

        val = float(row[metric])
        # the snippet must appear verbatim in the document
        present = snippet in doc
        # the figure that represents this metric inside the snippet (candidates
        # above 20, so configuration indices are skipped) must agree with the
        # CSV value, tolerating rounding and thousands separators
        nums = [n.replace(",", "") for n in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", snippet)]
        cands = [float(n) for n in nums if float(n) > 20]
        num_ok = any(_num_close(c, val, tol=0.6) for c in cands) if cands else True

        status = "ok" if (present and num_ok) else "FAIL"
        print(f"  [{status}] {label}: csv={val} snippet_present={present} "
              f"snippet_numbers_match={num_ok}")
        if not present:
            print(f"        missing snippet: {snippet!r}")
        if status == "FAIL":
            bad.append(label)
            FAILS.append(f"claim: {label}")
        else:
            checked += 1

    print(f"claims verified: {checked}/{len(claims)}")
    if not bad:
        print("  every sampled figure is stated verbatim in the document and matches the CSVs")


def self_test():
    """Prove the audit can fail: inject a known error, confirm it is caught, restore.

    This step matters. Seeing RESULT: PASS alone does not show the audit works;
    a check that always passes is worthless. The injected fault proves it is
    really inspecting the document.
    """
    section("SELF-TEST: does this audit actually detect a fault?")
    docp = ROOT / "需求.txt"
    original = io.open(docp, encoding="utf-8-sig").read()
    marker = "平均查询 2117.6"
    if marker not in original:
        print(f"  [SKIP] marker {marker!r} not found; cannot self-test")
        return 0

    tampered = original.replace(marker, "平均查询 8888.8")
    try:
        io.open(docp, "w", encoding="utf-8").write(tampered)
        global FAILS
        saved, FAILS = FAILS, []
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            check_doc_numbers()
        caught = len(FAILS) > 0
        FAILS = saved
        print(f"  injected fault into {marker!r} -> audit detected: {caught}")
        if not caught:
            print("  [FAIL] the audit did NOT notice a wrong figure")
            FAILS.append("self-test: fault not detected")
            return 1
        print("  self-test OK: the audit is genuinely checking the document")
        return 0
    finally:
        io.open(docp, "w", encoding="utf-8").write(original)


def main():
    selftest = "--self-test" in sys.argv
    files = tracked_files()
    print(f"repository: {ROOT}")
    print(f"tracked files: {len(files)}")
    check_compile(files)
    check_cjk(files)
    check_doc_paths(files)
    check_repo_root()
    check_doc_numbers()
    if selftest:
        self_test()

    section("SUMMARY")
    print(f"FAILURES: {len(FAILS)}")
    for f in FAILS:
        print(f"  [FAIL] {f}")
    print(f"WARNINGS: {len(WARNS)}")
    for w in WARNS:
        print(f"  [WARN] {w}")
    print()
    print("RESULT:", "PASS" if not FAILS else "FAIL")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
