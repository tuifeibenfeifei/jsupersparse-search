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
"""Generate Graphical Abstract"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

# Graphical abstract is vertical: 531 x 1328 pixels (h x w)
# Create figure with appropriate aspect ratio
fig, ax = plt.subplots(figsize=(5.31, 13.28), dpi=100)
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)
ax.axis('off')

# Title
ax.text(5, 24, 'Black-Box Online Search', ha='center', va='center', 
        fontsize=14, fontweight='bold')
ax.text(5, 23.2, 'Framework', ha='center', va='center', 
        fontsize=14, fontweight='bold')

# Subtitle
ax.text(5, 22, 'Model-free | Cross-domain | Budget-efficient', 
        ha='center', va='center', fontsize=9, style='italic', color='#666')

# Three stages visualization
# Stage 1
stage1 = FancyBboxPatch((1, 19.5), 8, 1.5, boxstyle="round,pad=0.1", 
                        facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
ax.add_patch(stage1)
ax.text(5, 20.5, 'Stage 1: Blind Sweep', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#1976D2')
ax.text(5, 19.9, 'Cold start coverage', ha='center', va='center', fontsize=9)

# Arrow down
ax.annotate('', xy=(5, 19.2), xytext=(5, 18.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

# Stage 2
stage2 = FancyBboxPatch((1, 17), 8, 1.5, boxstyle="round,pad=0.1", 
                        facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
ax.add_patch(stage2)
ax.text(5, 18, 'Stage 2: AUC Sparsification', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#388E3C')
ax.text(5, 17.4, 'Online feature selection', ha='center', va='center', fontsize=9)

# Arrow down
ax.annotate('', xy=(5, 16.7), xytext=(5, 16.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

# Stage 3
stage3 = FancyBboxPatch((1, 14.5), 8, 1.5, boxstyle="round,pad=0.1", 
                        facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
ax.add_patch(stage3)
ax.text(5, 15.5, 'Stage 3: Q-Learning + Far-Jump', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#F57C00')
ax.text(5, 14.9, 'Exploit + Explore', ha='center', va='center', fontsize=9)

# Key results box
results_box = FancyBboxPatch((1, 11.5), 8, 2.2, boxstyle="round,pad=0.1", 
                             facecolor='#FFF8E1', edgecolor='#FFA000', linewidth=2)
ax.add_patch(results_box)
ax.text(5, 13.2, 'Key Results', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#FFA000')
ax.text(5, 12.5, '4-6x sample efficiency', ha='center', va='center', fontsize=10)
ax.text(5, 11.9, 'vs full-scan baselines', ha='center', va='center', fontsize=9)

# Application scenarios
ax.text(5, 10.5, 'Validated Across Domains', ha='center', va='center', 
        fontsize=11, fontweight='bold')

# Scenario icons (simple boxes)
scenarios = [
    (1.5, 9, 'Game\n(MC)'),
    (4, 9, 'Species\n(species)'),
    (6.5, 9, 'Ocean\n(chlorophyll)'),
    (8.5, 9, '3D\n(geology)'),
]

for x, y, label in scenarios:
    box = FancyBboxPatch((x-0.9, y-0.6), 1.8, 1.2, boxstyle="round,pad=0.05", 
                         facecolor='#F5F5F5', edgecolor='#424242', linewidth=1)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=8)

# Bottom message
ax.text(5, 7, 'No surrogate model needed', ha='center', va='center', 
        fontsize=10, fontweight='bold')
ax.text(5, 6.3, 'Just binary presence feedback', ha='center', va='center', fontsize=9)
ax.text(5, 5.6, '+ covariate vector', ha='center', va='center', fontsize=9)

# Value proposition
value_box = FancyBboxPatch((1, 3.5), 8, 1.5, boxstyle="round,pad=0.1", 
                            facecolor='#E0F7FA', edgecolor='#0097A7', linewidth=2)
ax.add_patch(value_box)
ax.text(5, 4.5, 'When to use?', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='#0097A7')
ax.text(5, 3.9, 'Budget-limited | Scene unknown', ha='center', va='center', fontsize=9)

# Author info
ax.text(5, 2, 'H. Xie', ha='center', va='center', fontsize=9)
ax.text(5, 1.3, 'Putian University', ha='center', va='center', fontsize=8, color='#666')

# Save
plt.tight_layout()
plt.savefig(REPO_ROOT / "graphical_abstract.png", 
            dpi=100, bbox_inches='tight', facecolor='white')
plt.close()
print("Generated: graphical_abstract.png")
