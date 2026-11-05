# -*- coding: utf-8 -*-
"""fetch_chlorophyll.py - download ESA CCI monthly mean chlorophyll for the Bay of Bengal (real data).
Region: 10-20N x 85-95E, 2024-06, 0.04 deg grid (about 200x200 = 40000 points).
Output: chlorophyll-search/data/raw_chlorophyll.csv

The download runs only when the file is executed as a script, so importing it (for
example from a repository-wide health check) never touches the network.
"""
import csv
import sys
import urllib.error
import urllib.request
from pathlib import Path

# --- repository root: walk up until the directory that holds configs/ ---
REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "configs").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("repository root (containing configs/) not found")
    REPO_ROOT = REPO_ROOT.parent
# -----------------------------------------------------------------------

URL = ("https://oceanwatch.pifsc.noaa.gov/erddap/griddap/esa-cci-chla-monthly-v6-0.csv"
       "?chlor_a%5B(2024-06-01T00:00:00Z)%5D%5B(10.0):(20.0)%5D%5B(85.0):(95.0)%5D")


def main() -> int:
    """Download the gridded product and write data/raw_chlorophyll.csv."""
    out = REPO_ROOT / "chlorophyll-search" / "data"
    out.mkdir(parents=True, exist_ok=True)

    print("downloading", URL[:80], "...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        raw = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        print(f"[error] download failed: {exc}")
        print("        the server needs to be reachable; retry later or supply the CSV yourself")
        return 1

    lines = raw.splitlines()
    print("lines:", len(lines), "first:", lines[0] if lines else "")
    # drop the leading header comment lines (time/UTC rows) and the column header
    data_lines = [ln for ln in lines
                  if ln and not ln.startswith("UTC")
                  and ln != "time,latitude,longitude,chlor_a"]

    n = 0
    with open(out / "raw_chlorophyll.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["latitude", "longitude", "chlor_a"])
        for ln in data_lines:
            parts = ln.split(",")
            if len(parts) < 4:
                continue
            try:
                w.writerow([float(parts[1]), float(parts[2]), float(parts[3])])
                n += 1
            except ValueError:
                continue
    print("rows written:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
