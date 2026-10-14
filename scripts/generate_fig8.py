#!/usr/bin/env python
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
"""Generate Figure 8: Sample efficiency comparison"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11

# Data from Table 4
scenes = ['Chlorophyll', '3D hypoxic core', 'Clouded leopard']
framework_efficiency = [4.19, 1.59, 1.22]
fullscan_efficiency = [0.71, 0.38, 0.77]

x = np.arange(len(scenes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, framework_efficiency, width, label='Proposed framework', 
                color='#1976D2', alpha=0.8)
rects2 = ax.bar(x + width/2, fullscan_efficiency, width, label='Best full-scan method', 
                color='#F57C00', alpha=0.8)

# Add labels, title and legend
ax.set_ylabel('Points per 1000 covered cells', fontsize=12)
ax.set_title('Sample Efficiency Comparison: Framework vs Full-Scan', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenes, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

# Add ratio annotations
ratios = [5.9, 4.2, 1.6]
for i, ratio in enumerate(ratios):
    ax.annotate(f'{ratio:.1f}x',
                xy=(i, max(framework_efficiency[i], fullscan_efficiency[i]) + 0.3),
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#388E3C')

plt.tight_layout()
plt.savefig(REPO_ROOT / "figures/Fig8_sample_efficiency.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Generated: Fig8_sample_efficiency.png")
