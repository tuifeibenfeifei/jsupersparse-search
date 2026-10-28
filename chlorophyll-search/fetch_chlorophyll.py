# -*- coding: utf-8 -*-
# --- repository root (auto-located) ---
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
# --------------------------------------
"""fetch_chlorophyll.py - download ESA CCI monthly mean chlorophyll for the Bay of Bengal (real data).
Region: 10-20N x 85-95E, 2024-06, 0.04 deg grid (about 200x200 = 40000 points).
Output: data/raw_chlorophyll.csv
"""
import csv
import urllib.request
from pathlib import Path

OUT = Path(REPO_ROOT / "chlorophyll-search/data")
OUT.mkdir(parents=True, exist_ok=True)

url = ("https://oceanwatch.pifsc.noaa.gov/erddap/griddap/esa-cci-chla-monthly-v6-0.csv"
       "?chlor_a%5B(2024-06-01T00:00:00Z)%5D%5B(10.0):(20.0)%5D%5B(85.0):(95.0)%5D")
print("downloading", url[:80], "...")
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
raw = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")
lines = raw.splitlines()
print("lines:", len(lines), "first:", lines[0] if lines else "")
# drop the leading header comment lines (time/UTC rows)
data_lines = [l for l in lines if l and not l.startswith("UTC") and l != "time,latitude,longitude,chlor_a"]
with open(OUT / "raw_chlorophyll.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["latitude", "longitude", "chlor_a"])
    n = 0
    for l in data_lines:
        parts = l.split(",")
        if len(parts) < 4:
            continue
        lat, lon, chl = parts[1], parts[2], parts[3]
        try:
            w.writerow([float(lat), float(lon), float(chl)])
            n += 1
        except ValueError:
            continue
print("rows written:", n)
