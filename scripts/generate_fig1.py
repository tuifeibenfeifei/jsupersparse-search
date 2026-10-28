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
"""Generate Figure 1: Framework architecture diagram"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(6, 7.7, 'Three-Stage Black-Box Online Search Framework', 
        ha='center', va='center', fontsize=14, fontweight='bold')

# Stage 1 box
stage1 = FancyBboxPatch((1, 5.5), 3, 1.5, boxstyle="round,pad=0.1", 
                         facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
ax.add_patch(stage1)
ax.text(2.5, 6.5, 'Stage 1\nBlind Sweep\nCoverage', ha='center', va='center', 
        fontsize=11, fontweight='bold')
ax.text(2.5, 5.8, '12% budget\nCold start', ha='center', va='center', fontsize=9)

# Stage 2 box
stage2 = FancyBboxPatch((4.5, 5.5), 3, 1.5, boxstyle="round,pad=0.1", 
                         facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
ax.add_patch(stage2)
ax.text(6, 6.5, 'Stage 2\nCovariate Binning\n+ AUC Sparsification', 
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(6, 5.8, 'Online feature selection\nGuidance map', ha='center', va='center', fontsize=9)

# Stage 3 box
stage3 = FancyBboxPatch((8, 5.5), 3, 1.5, boxstyle="round,pad=0.1", 
                        facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
ax.add_patch(stage3)
ax.text(9.5, 6.5, 'Stage 3\nReward Shaping\n+ Q-Table', 
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(9.5, 5.8, 'Exploit + Explore\nFar-jump mechanism', ha='center', va='center', fontsize=9)

# Arrows between stages
arrow1 = FancyArrowPatch((4, 6.25), (4.5, 6.25), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='#333')
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((7.5, 6.25), (8, 6.25), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='#333')
ax.add_patch(arrow2)

# Trigger labels
ax.text(4.25, 6.7, '5+ hits', ha='center', va='center', fontsize=8, style='italic')
ax.text(7.75, 6.7, 'Dry spell', ha='center', va='center', fontsize=8, style='italic')

# Lower modules
# Mutation detection
mutation = FancyBboxPatch((1, 3.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                          facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=1.5)
ax.add_patch(mutation)
ax.text(2.25, 4.3, 'Mutation Detection', ha='center', va='center', 
        fontsize=10, fontweight='bold')
ax.text(2.25, 3.8, 'Zero-hit streak > 40\nScatter jumps', ha='center', va='center', fontsize=8)

# Follow mode
follow = FancyBboxPatch((4, 3.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                        facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=1.5)
ax.add_patch(follow)
ax.text(5.25, 4.3, 'Slow-Drift Follow', ha='center', va='center', 
        fontsize=10, fontweight='bold')
ax.text(5.25, 3.8, 'Annular exploration\nRing radius 8', ha='center', va='center', fontsize=8)

# Multi-cluster adaptation
multicluster = FancyBboxPatch((7, 3.5), 3, 1.2, boxstyle="round,pad=0.1", 
                              facecolor='#E0F7FA', edgecolor='#0097A7', linewidth=1.5)
ax.add_patch(multicluster)
ax.text(8.5, 4.3, 'Multi-Cluster Adaptation', ha='center', va='center', 
        fontsize=10, fontweight='bold')
ax.text(8.5, 3.8, 'BFS clustering diagnosis\nSingle vs multi cluster', ha='center', va='center', fontsize=8)

# Arrows from lower modules to stages
arrow_follow = FancyArrowPatch((5.25, 4.7), (5.25, 5.5), arrowstyle='->', 
                               mutation_scale=15, linewidth=1.5, color='#7B1FA2', linestyle='--')
ax.add_patch(arrow_follow)

arrow_multicluster = FancyArrowPatch((8.5, 4.7), (8.5, 5.5), arrowstyle='->', 
                                     mutation_scale=15, linewidth=1.5, color='#0097A7', linestyle='--')
ax.add_patch(arrow_multicluster)

# Input/Output boxes
# Input: Presence feedback
input_box = FancyBboxPatch((1, 1.5), 3, 1.2, boxstyle="round,pad=0.1", 
                            facecolor='#F5F5F5', edgecolor='#424242', linewidth=1.5)
ax.add_patch(input_box)
ax.text(2.5, 2.3, 'Input: Binary Presence', ha='center', va='center', 
        fontsize=10, fontweight='bold')
ax.text(2.5, 1.8, 'presence in {0,1}\n+ covariate vector x', ha='center', va='center', fontsize=8)

# Output: Valid points
output_box = FancyBboxPatch((8, 1.5), 3, 1.2, boxstyle="round,pad=0.1", 
                            facecolor='#F5F5F5', edgecolor='#424242', linewidth=1.5)
ax.add_patch(output_box)
ax.text(9.5, 2.3, 'Output: Valid Points', ha='center', va='center', 
        fontsize=10, fontweight='bold')
ax.text(9.5, 1.8, 'Spacing constraint\nmax points / budget', ha='center', va='center', fontsize=8)

# Arrow from input to framework
arrow_input = FancyArrowPatch((2.5, 2.7), (2.5, 5.5), arrowstyle='->', 
                              mutation_scale=15, linewidth=2, color='#333')
ax.add_patch(arrow_input)

# Arrow from framework to output
arrow_output = FancyArrowPatch((9.5, 5.5), (9.5, 2.7), arrowstyle='->', 
                                mutation_scale=15, linewidth=2, color='#333')
ax.add_patch(arrow_output)

# Baseline comparison box
baseline_box = FancyBboxPatch((4.5, 1.5), 3, 1.2, boxstyle="round,pad=0.1", 
                              facecolor='#FFF8E1', edgecolor='#FFA000', linewidth=1.5)
ax.add_patch(baseline_box)
ax.text(6, 2.3, 'Compared Baselines', ha='center', va='center', 
        fontsize=10, fontweight='bold')
ax.text(6, 1.8, 'Random, Grid, Hill Climb\nGP-UCB, Go-Explore, etc.', ha='center', va='center', fontsize=8)

# Save figure
plt.tight_layout()
(REPO_ROOT / "figures").mkdir(parents=True, exist_ok=True)
plt.savefig(REPO_ROOT / "figures/Fig1_framework_architecture.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Generated: Fig1_framework_architecture.png")
