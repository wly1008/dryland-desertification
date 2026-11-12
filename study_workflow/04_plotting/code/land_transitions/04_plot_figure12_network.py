"""Plot the land-cover transition network (Figure 12)."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
from pathlib import Path
import json
import runpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
ROOT = PROJECT_ROOT / '03_analysis/06_land_transitions'
OUT = FIGURES_ROOT / 'jpg'
s = json.loads((PROJECT_ROOT / '03_analysis/06_land_transitions/data/analysis_summary.json').read_text(encoding='utf-8'))
z = np.load(PROJECT_ROOT / '03_analysis/06_land_transitions/data/lucc_dsi_statistics.npz')
selected = z['display_transitions']
combos = z['top_combinations']
short = ['Forest', 'Grassland', 'Cropland', 'Wetlands', 'Settlements', 'Other land']
labels = [f'{short[t // 6]} → {short[t % 6]}' for t in selected]
cell_area = z['display_area_km2'][selected]
cell_count = z['display_count'][selected]
medians = z['display_quantiles'][selected, :, 2] * 1000
within = cell_area / z['area_km2'][selected, None] * 100
assert s['schema_version'] == 4
limit = s['color_scale']['symmetric_limit']
norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
cmap = LinearSegmentedColormap.from_list('DSI', ['#2166AC', '#92C5DE', '#F7F7F7', '#F4A582', '#B2182B'])
cmap.set_bad('#F0F1F2')
plt.rcParams.update({'font.family': 'Arial', 'font.sans-serif': ['Arial'], 'font.size': 6, 'font.weight': 'normal', 'axes.linewidth': 0.5, 'axes.edgecolor': '#9AA5AC', 'text.color': '#30363D', 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial', 'mathtext.bf': 'Arial:bold', 'mathtext.default': 'regular', 'axes.titleweight': 'normal'})
fig = plt.figure(figsize=(17 / 2.54, 9.5 / 2.54))
bottom, height, heat_left, heat_width = (0.32, 0.59, 0.51, 0.45)
fig.text(0.015, 0.955, '(a) Within-transition composition', fontsize=8, fontweight='normal')
fig.text(heat_left, 0.955, '(b) Median DSI trend', fontsize=8, fontweight='normal')
network = fig.add_axes([0, 0, 1, 1])
network.set_xlim(0, 1)
network.set_ylim(0, 1)
network.axis('off')
source_x, target_x = (0.35, 0.145)
source_y = bottom + height * (1 - (np.arange(len(selected)) + 0.5) / len(selected))
target_y = bottom + height * (1 - (np.arange(within.shape[1]) + 0.5) / within.shape[1])
highlight = np.zeros_like(cell_count, dtype=bool)
share_rank = np.zeros_like(cell_count, dtype=int)
rank_colors = ['#D0D0D0', '#4DAF4A', '#756BB1', '#A6AD35']
for i in range(len(selected)):
    candidates = np.flatnonzero(cell_count[i] > 0)
    ranked = candidates[np.argsort(-within[i, candidates], kind='stable')]
    highlight[i, ranked[:3]] = True
    share_rank[i, ranked[:3]] = np.arange(1, len(ranked[:3]) + 1)
for prominent in [False, True]:
    for i in range(len(selected)):
        for j in range(within.shape[1]):
            if cell_count[i, j] == 0 or highlight[i, j] != prominent:
                continue
            curve = MplPath([(source_x, source_y[i]), (source_x - 0.1, source_y[i]), (target_x + 0.08, target_y[j]), (target_x, target_y[j])], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
            network.add_patch(PathPatch(curve, facecolor='none', edgecolor=rank_colors[share_rank[i, j]], lw=max(0.18, 0.055 * within[i, j]), alpha=0.78 if prominent else 0.42, capstyle='round', zorder=2 if prominent else 1))
for i, t in enumerate(selected):
    network.text(source_x + 0.009, source_y[i], labels[i], ha='left', va='center', fontsize=6)
    network.scatter(source_x, source_y[i], s=12, facecolor='#C3C6C6' if t // 6 == t % 6 else '#737C80', edgecolor='white', linewidth=0.35, zorder=4)
arrows = ['↓', 'NS', '↑']
for j, ypos in enumerate(target_y):
    network.scatter(target_x, ypos, s=16, marker='^', facecolor='#3B528B', edgecolor='white', linewidth=0.35, zorder=4)
    if j < len(combos):
        c = int(combos[j])
        states = [(c - 1) // 9, (c - 1) % 9 // 3, (c - 1) % 3]
        network.text(0.032, ypos, f'C{c:02d}', ha='left', va='center', fontsize=6)
        state_centers = [0.079, 0.102, 0.125]
        for k, (xpos, state) in enumerate(zip(state_centers, states)):
            network.text(xpos, ypos, arrows[state], ha='center', va='center', fontsize=6, gid=f'FVC_state_C{c:02d}_{k}')
        for k in range(2):
            xpos = (state_centers[k] + state_centers[k + 1]) / 2
            network.text(xpos, ypos, '/', ha='center', va='center', fontsize=6, gid=f'FVC_separator_C{c:02d}_{k}')
    else:
        label = 'Other 19 pooled'
        network.text(target_x - 0.009, ypos, label, ha='right', va='center', fontsize=6)
ax = fig.add_axes([heat_left, bottom, heat_width, height])
ax.imshow(np.ma.masked_where(cell_count == 0, medians), cmap=cmap, norm=norm, aspect='auto', interpolation='none')
ax.set_xticks(range(within.shape[1]), [f'C{int(c):02d}' for c in combos] + ['Other 19'], rotation=0, ha='center', fontsize=6)
ax.set_yticks([])
ax.set_xticks(np.arange(-0.5, within.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(selected), 1), minor=True)
ax.grid(which='minor', color='#E0E5E7', lw=0.4)
ax.tick_params(which='both', length=0, pad=3)
ax.axvline(within.shape[1] - 1.5, color='#8B9BA3', ls='--', lw=0.7)
for i, t in enumerate(selected):
    if i < len(selected) - 1 and selected[i + 1] // 6 != t // 6:
        ax.axhline(i + 0.5, color='#8B9BA3', lw=0.7)
    for j in range(within.shape[1]):
        value = medians[i, j]
        if cell_count[i, j] == 0:
            label, color = ('—', '#AEB9BF')
        else:
            label = ('0.00' if round(value, 2) == 0 else f'{value:+.2f}') + ('†' if cell_count[i, j] < s['small_cell_flag_n'] else '')
            color = 'white' if abs(value) > limit * 0.72 else '#30363D'
        ax.text(j, i, label, ha='center', va='center', fontsize=6, color=color)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([heat_left + 0.025, 0.245, heat_width - 0.05, 0.015]), orientation='horizontal')
cb.set_label('Median DSI trend ($10^{-3}$ yr$^{-1}$)', fontsize=6, labelpad=2)
cb.ax.xaxis.label.set_gid('colorbar_title')
cb.ax.tick_params(labelsize=6, length=2)
network.text(0.045, 0.255, 'Within-group area (%)', fontsize=6, va='center').set_gid('legend_title')
for x, share in zip([0.17, 0.275, 0.38], [5, 25, 50]):
    network.plot([x, x + 0.055], [0.255, 0.255], color='#909595', lw=max(0.18, 0.055 * share), solid_capstyle='round')
    network.text(x + 0.028, 0.225, str(share), ha='center', va='center', fontsize=6)
network.text(0.045, 0.185, 'Share rank:', va='center', fontsize=6).set_gid('legend_title')
for x, rank, label in zip([0.17, 0.255, 0.34, 0.425], [1, 2, 3, 0], ['1st', '2nd', '3rd', 'Other']):
    network.plot([x, x + 0.028], [0.185, 0.185], color=rank_colors[rank], lw=1.2)
    network.text(x + 0.035, 0.185, label, va='center', fontsize=6)
assert np.allclose(within.sum(axis=1), 100), 'Within-transition shares must sum to 100%.'
assert cell_area.shape == (10, 9), 'Review layout if the displayed matrix changes.'
fig._lucc_alignment = (network, ax, source_y)
runpy.run_path(str(CODE_ROOT / '04_plotting/code/land_transitions/05_export_figure.py'), init_globals={'fig': fig, 'base': 'Figure12_LUCC_FVC_DSI', 'OUT': OUT, 'target_width_cm': 17})
plt.close(fig)
print(f'Saved v4 Figure12; {np.count_nonzero(cell_count)} links, {np.count_nonzero(highlight)} highlighted; color limits ±{limit:g} x 10^-3 per year.')
