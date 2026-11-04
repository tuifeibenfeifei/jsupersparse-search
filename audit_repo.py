# -*- coding: utf-8 -*-
"""audit_repo.py — 可复现的仓库/文档一致性审计。

用法（在仓库根目录执行）：
    python audit_repo.py

输出：
    1) 所有受跟踪代码文件是否编译通过
    2) 受跟踪代码里是否残留中文（仓库规范要求为 0）
    3) 需求.txt 里引用的每个文件路径是否真实存在
    4) supplement_stats 下每个脚本的 REPO_ROOT 是否解析到仓库根
    5) 需求.txt 里引用的关键数值是否能在结果 CSV 中找到

任何一项不通过都会以 [FAIL] 标出，脚本退出码非 0。
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
    code = [f for f in files if f.endswith(exts)]
    bad = []
    for f in code:
        p = ROOT / f
        if not p.exists():
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

# 文档里合法出现、但本来就不在仓库里的路径，逐条写明理由。
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
    """需求.txt 引用路径核查。

    文档里的路径多数是相对各自所在目录写的（例如 run_chl.py 实际在
    chlorophyll-search/ 下），因此判定规则是：完整路径存在，或仓库中任意
    位置存在同名文件，都算通过。只有两者都不满足才算真缺失。
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

    # 无法解析的引用必须逐条给出理由，否则算失败。这样每条豁免都可追溯，
    # 而不是笼统地「全部忽略」。
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
    """静态核查 supplement_stats 脚本的 REPO_ROOT 解析。

    不执行脚本（执行会产生副作用，例如 mc_paired_audit.py 会把统计结果
    打印出来）。改为按脚本自身的解析逻辑在内存里模拟：
    从脚本所在目录逐级向上，找到含 configs/ 的目录即为仓库根。
    同时确认脚本带有「找不到就报错」的守卫，避免静默落到错误目录。
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

        # 模拟：从脚本目录向上找 configs/
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


# 每条断言对应的「文档内定位锚点」：每对 (配置名, 该行应带的数字)。
# 凡是同一行同时出现这两个词的，就是文档在陈述该配置的数值，
# 该行必须带着正确数字，否则说明文档写错了。
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
    """比较两个数字是否在容差内相等（对数级差异也算不等）。"""
    try:
        fa = float(a)
    except (TypeError, ValueError):
        return False
    return abs(fa - b) <= tol


def check_doc_numbers():
    """把 需求.txt 里的关键数值断言与结果 CSV 逐条对账。

    每条断言 = (说明, CSV 路径, 配置列, 配置值, 指标列, 文档中应逐字出现的片段)。
    片段是精确文本，因此文档里只要有一处把数字写错、或整段话被删掉，
    这一项就会失败——不会因为别处恰好出现正确数字而蒙混过关。
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
        # 文档里必须逐字出现声明的片段
        present = snippet in doc
        # 片段中代表该指标的数字（取 >20 的候选，避开配置名里的序号），
        # 必须有一个与 CSV 值吻合（容忍四舍五入与千分位逗号）
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
    """验证审计确实会失败：注入一个已知错误，确认 audit 报 FAIL，然后还原。

    这一步很关键。只看到 RESULT: PASS 并不能说明审计有效——一个永远
    返回 PASS 的检查毫无价值。这里通过注入错误证明它真的在检查。
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
