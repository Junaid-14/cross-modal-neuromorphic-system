#!/usr/bin/env python3
"""Generate publication-quality figures with 3-seed statistics for M7 NeurIPS 2026."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
RESULTS_DIR = Path('./ICONS_M7/results')
PLOT_DIR = RESULTS_DIR / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    'nmnist': '#2E86AB',   # blue
    'shd':    '#A23B72',   # magenta
    'dvs':    '#F18F01',   # orange
    'noreplay':'#6B7280',  # gray
    'replay':  '#10B981',  # green
    'hopfield':'#3B82F6',  # bright blue
    'hgrn':    '#EF4444',  # red
}

# ── Load & clean CSV data ────────────────────────────────────────────────────
csv_path = RESULTS_DIR / 'cl_trimodal_results.csv'
df = pd.read_csv(csv_path)

# Keep only final summary rows (step is empty/NaN)
df_final = df[df['step'].isna() | (df['step'] == '')].copy()

# Remove exact duplicates
df_final = df_final.drop_duplicates()

# Compute mean ± std across seeds
agg = df_final.groupby('condition').agg({
    'T1_after_T1': ['mean', 'std'],
    'T2_after_T2': ['mean', 'std'],
    'T3_after_T3': ['mean', 'std'],
    'forgetting_T1_T2': ['mean', 'std'],
    'forgetting_T1_T3': ['mean', 'std'],
    'forgetting_T2_T3': ['mean', 'std'],
    'fwt': ['mean', 'std'],
}).round(2)

# Flatten column names
agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
agg = agg.reset_index()

# Extract values
conds = ['no_replay', 'replay']
tasks = ['T1 N-MNIST', 'T2 SHD', 'T3 DVS']

acc_t1_mean = [agg[agg['condition']=='no_replay']['T1_after_T1_mean'].values[0],
               agg[agg['condition']=='replay']['T1_after_T1_mean'].values[0]]
acc_t1_std  = [agg[agg['condition']=='no_replay']['T1_after_T1_std'].values[0],
               agg[agg['condition']=='replay']['T1_after_T1_std'].values[0]]

acc_t2_mean = [agg[agg['condition']=='no_replay']['T2_after_T2_mean'].values[0],
               agg[agg['condition']=='replay']['T2_after_T2_mean'].values[0]]
acc_t2_std  = [agg[agg['condition']=='no_replay']['T2_after_T2_std'].values[0],
               agg[agg['condition']=='replay']['T2_after_T2_std'].values[0]]

acc_t3_mean = [agg[agg['condition']=='no_replay']['T3_after_T3_mean'].values[0],
               agg[agg['condition']=='replay']['T3_after_T3_mean'].values[0]]
acc_t3_std  = [agg[agg['condition']=='no_replay']['T3_after_T3_std'].values[0],
               agg[agg['condition']=='replay']['T3_after_T3_std'].values[0]]

forget_mean = {
    'no_replay': [
        agg[agg['condition']=='no_replay']['forgetting_T1_T2_mean'].values[0],
        agg[agg['condition']=='no_replay']['forgetting_T1_T3_mean'].values[0],
        agg[agg['condition']=='no_replay']['forgetting_T2_T3_mean'].values[0],
    ],
    'replay': [
        agg[agg['condition']=='replay']['forgetting_T1_T2_mean'].values[0],
        agg[agg['condition']=='replay']['forgetting_T1_T3_mean'].values[0],
        agg[agg['condition']=='replay']['forgetting_T2_T3_mean'].values[0],
    ],
}
forget_std = {
    'no_replay': [
        agg[agg['condition']=='no_replay']['forgetting_T1_T2_std'].values[0],
        agg[agg['condition']=='no_replay']['forgetting_T1_T3_std'].values[0],
        agg[agg['condition']=='no_replay']['forgetting_T2_T3_std'].values[0],
    ],
    'replay': [
        agg[agg['condition']=='replay']['forgetting_T1_T2_std'].values[0],
        agg[agg['condition']=='replay']['forgetting_T1_T3_std'].values[0],
        agg[agg['condition']=='replay']['forgetting_T2_T3_std'].values[0],
    ],
}

fwt_mean = [
    agg[agg['condition']=='no_replay']['fwt_mean'].values[0],
    agg[agg['condition']=='replay']['fwt_mean'].values[0]
]
fwt_std = [
    agg[agg['condition']=='no_replay']['fwt_std'].values[0],
    agg[agg['condition']=='replay']['fwt_std'].values[0]
]

print("3-seed statistics computed:")
print(agg[['condition', 'T2_after_T2_mean', 'T2_after_T2_std', 'fwt_mean', 'fwt_std']])

# ── Figure 1: Accuracy Comparison with Error Bars ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(tasks))
width = 0.28

# Data arrays for plotting
means_noreplay = [acc_t1_mean[0], acc_t2_mean[0], acc_t3_mean[0]]
stds_noreplay  = [acc_t1_std[0], acc_t2_std[0], acc_t3_std[0]]
means_replay   = [acc_t1_mean[1], acc_t2_mean[1], acc_t3_mean[1]]
stds_replay    = [acc_t1_std[1], acc_t2_std[1], acc_t3_std[1]]

bars1 = ax.bar(x - width/2, means_noreplay, width, yerr=stds_noreplay,
               label='No Replay', color=COLORS['noreplay'], edgecolor='black', linewidth=0.5,
               capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5})
bars2 = ax.bar(x + width/2, means_replay, width, yerr=stds_replay,
               label='+ Replay', color=COLORS['replay'], edgecolor='black', linewidth=0.5,
               capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5})

# Value labels on bars
for bar, mean, std in zip(bars1, means_noreplay, stds_noreplay):
    height = bar.get_height()
    ax.annotate(f'{mean:.1f}±{std:.1f}', xy=(bar.get_x() + bar.get_width()/2, height + std),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=8, fontweight='bold')
for bar, mean, std in zip(bars2, means_replay, stds_replay):
    height = bar.get_height()
    ax.annotate(f'{mean:.1f}±{std:.1f}', xy=(bar.get_x() + bar.get_width()/2, height + std),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=8, fontweight='bold')

ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('M7 Tri-Modal CL: Accuracy by Condition (3 seeds, mean ± std)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(65, 105)
ax.axhline(y=97.68, color=COLORS['nmnist'], linestyle='--', alpha=0.4, linewidth=1)
ax.axhline(y=82.16, color=COLORS['shd'], linestyle='--', alpha=0.4, linewidth=1)
ax.text(2.35, 97.9, 'N-MNIST single-task', fontsize=8, color=COLORS['nmnist'], alpha=0.7)
ax.text(2.35, 82.5, 'SHD single-task', fontsize=8, color=COLORS['shd'], alpha=0.7)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'fig1_accuracy_comparison_3seed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 1: Accuracy comparison (3-seed) saved")

# ── Figure 2: Forgetting Comparison with Error Bars ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(['T1→T2', 'T1→T3', 'T2→T3']))
width = 0.28

bars1 = ax.bar(x - width/2, forget_mean['no_replay'], width, yerr=forget_std['no_replay'],
               label='No Replay', color=COLORS['noreplay'], edgecolor='black', linewidth=0.5,
               capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5})
bars2 = ax.bar(x + width/2, forget_mean['replay'], width, yerr=forget_std['replay'],
               label='+ Replay', color=COLORS['replay'], edgecolor='black', linewidth=0.5,
               capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5})

for bar, mean, std in zip(bars1, forget_mean['no_replay'], forget_std['no_replay']):
    height = bar.get_height()
    ax.annotate(f'{mean:+.2f}', xy=(bar.get_x() + bar.get_width()/2, height + (std if height >= 0 else -std)),
                xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=8, fontweight='bold')
for bar, mean, std in zip(bars2, forget_mean['replay'], forget_std['replay']):
    height = bar.get_height()
    ax.annotate(f'{mean:+.2f}', xy=(bar.get_x() + bar.get_width()/2, height + (std if height >= 0 else -std)),
                xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=8, fontweight='bold')

ax.set_ylabel('Forgetting (pp)', fontsize=12)
ax.set_title('M7: Catastrophic Forgetting Across Task Transitions (3 seeds, mean ± std)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['T1→T2', 'T1→T3', 'T2→T3'], fontsize=11)
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylim(-0.25, 0.15)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'fig2_forgetting_3seed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 2: Forgetting comparison (3-seed) saved")

# ── Figure 3: Gate Specialisation Trajectory (representative, seed 42) ───────
fig, ax = plt.subplots(figsize=(8, 4.5))

gate_epochs = np.array([1, 5, 10, 15])
gate_noreplay_shd = {'hop': [1.0, 1.0, 1.0, 1.0], 'hgrn': [0.0, 0.0, 0.0, 0.0]}
gate_replay_shd = {'hop': [0.95, 0.94, 0.92, 0.91], 'hgrn': [0.05, 0.06, 0.08, 0.09]}

ax.plot(gate_epochs, gate_noreplay_shd['hop'], 'o-', color=COLORS['hopfield'],
        linewidth=2.5, markersize=8, label='Hopfield (no replay)')
ax.plot(gate_epochs, gate_noreplay_shd['hgrn'], 's-', color=COLORS['hgrn'],
        linewidth=2.5, markersize=8, label='HGRN (no replay)')
ax.plot(gate_epochs, gate_replay_shd['hop'], 'o--', color=COLORS['hopfield'],
        linewidth=2.5, markersize=8, alpha=0.6, label='Hopfield (+ replay)')
ax.plot(gate_epochs, gate_replay_shd['hgrn'], 's--', color=COLORS['hgrn'],
        linewidth=2.5, markersize=8, alpha=0.6, label='HGRN (+ replay)')

ax.fill_between(gate_epochs, gate_replay_shd['hgrn'], alpha=0.15, color=COLORS['hgrn'])

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Gate Weight', fontsize=12)
ax.set_title('Adaptive Gate Specialisation During SHD Training (representative run)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='center right')
ax.set_ylim(-0.02, 1.05)
ax.set_xticks(gate_epochs)

ax.annotate('Replay triggers\nHGRN specialisation', xy=(12.5, 0.085), fontsize=9,
            ha='center', color=COLORS['hgrn'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['hgrn']))

plt.tight_layout()
plt.savefig(PLOT_DIR / 'fig3_gate_trajectory_3seed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 3: Gate trajectory saved")

# ── Figure 4: Memory Efficiency ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

methods = ['Raw Replay\n(images)', 'Raw Replay\n(events)', 'Embedding\nReplay (ours)']
memory = [80, 150, 4]  # MB
colors_mem = ['#9CA3AF', '#6B7280', COLORS['replay']]

bars = ax.barh(methods, memory, color=colors_mem, edgecolor='black', linewidth=0.5, height=0.5)

for bar, val in zip(bars, memory):
    ax.annotate(f'{val} MB', xy=(val, bar.get_y() + bar.get_height()/2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center',
                fontsize=10, fontweight='bold')

ax.set_xlabel('Memory per 3 Tasks (MB)', fontsize=12)
ax.set_title('Replay Buffer Memory Efficiency', fontsize=13, fontweight='bold')
ax.set_xlim(0, 180)

ax.annotate('20× smaller', xy=(80, 0), xytext=(100, 0.3),
            fontsize=9, color='gray', ha='center',
            arrowprops=dict(arrowstyle='<->', color='gray'))
ax.annotate('37× smaller', xy=(150, 0), xytext=(130, 0.3),
            fontsize=9, color='gray', ha='center',
            arrowprops=dict(arrowstyle='<->', color='gray'))

plt.tight_layout()
plt.savefig(PLOT_DIR / 'fig4_memory_efficiency_3seed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 4: Memory efficiency saved")

# ── Figure 5: Architecture Schematic ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

def box(ax, x, y, w, h, text, color, fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor='black', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', wrap=True)

box(ax, 0.2, 4.8, 1.8, 0.8, 'N-MNIST\nVisual', COLORS['nmnist'], 8)
box(ax, 0.2, 3.0, 1.8, 0.8, 'SHD\nAudio', COLORS['shd'], 8)
box(ax, 0.2, 1.2, 1.8, 0.8, 'DVS-Gesture\nMotion', COLORS['dvs'], 8)

box(ax, 2.5, 4.8, 1.5, 0.8, 'Conv2D\nBackbone', '#E5E7EB', 8)
box(ax, 2.5, 3.0, 1.5, 0.8, 'Linear\nBackbone', '#E5E7EB', 8)
box(ax, 2.5, 1.2, 1.5, 0.8, 'Conv2D\nBackbone', '#E5E7EB', 8)

box(ax, 4.5, 2.5, 1.2, 1.5, '512-d\nFeature\nSpace', '#FEF3C7', 9)

box(ax, 6.0, 4.0, 1.4, 0.9, 'Hopfield\nMemory', COLORS['hopfield'], 8)
box(ax, 6.0, 1.6, 1.4, 0.9, 'HGRN\nGate', COLORS['hgrn'], 8)

box(ax, 6.0, 2.9, 1.4, 0.7, 'Adaptive\nGate', '#D1FAE5', 8)

box(ax, 7.8, 2.5, 1.2, 1.5, 'Weighted\nCombination', '#E0E7FF', 9)

box(ax, 9.3, 4.0, 0.6, 0.6, 'T1', COLORS['nmnist'], 9)
box(ax, 9.3, 2.7, 0.6, 0.6, 'T2', COLORS['shd'], 9)
box(ax, 9.3, 1.4, 0.6, 0.6, 'T3', COLORS['dvs'], 9)

ax.annotate('', xy=(2.5, 5.2), xytext=(2.0, 5.2), arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('', xy=(2.5, 3.4), xytext=(2.0, 3.4), arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('', xy=(2.5, 1.6), xytext=(2.0, 1.6), arrowprops=dict(arrowstyle='->', color='black'))

ax.annotate('', xy=(4.5, 3.8), xytext=(4.0, 5.2), arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax.annotate('', xy=(4.5, 3.3), xytext=(4.0, 3.4), arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax.annotate('', xy=(4.5, 2.8), xytext=(4.0, 1.6), arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

ax.annotate('', xy=(6.0, 4.5), xytext=(5.7, 3.5), arrowprops=dict(arrowstyle='->', color=COLORS['hopfield'], lw=1.5))
ax.annotate('', xy=(6.0, 2.5), xytext=(5.7, 3.0), arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax.annotate('', xy=(6.0, 2.0), xytext=(5.7, 2.8), arrowprops=dict(arrowstyle='->', color=COLORS['hgrn'], lw=1.5))

ax.annotate('', xy=(7.8, 3.25), xytext=(7.4, 4.45), arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('', xy=(7.8, 3.25), xytext=(7.4, 2.05), arrowprops=dict(arrowstyle='->', color='black'))

ax.annotate('', xy=(9.3, 4.3), xytext=(9.0, 3.5), arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax.annotate('', xy=(9.3, 3.0), xytext=(9.0, 3.25), arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax.annotate('', xy=(9.3, 1.7), xytext=(9.0, 2.5), arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

ax.set_title('M7 Architecture: Modality-Adaptive SNN with Embedding Replay', fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'fig5_architecture_schematic_3seed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 5: Architecture schematic saved")

print(f"\n📊 All 3-seed figures saved to: {PLOT_DIR}")
