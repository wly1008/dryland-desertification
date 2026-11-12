"""Generate the land-cover transition figures (Figures 11 and 12)."""
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
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle
ROOT = PROJECT_ROOT / '03_analysis/06_land_transitions'
DATA, OUT = (PROJECT_ROOT / '03_analysis/06_land_transitions/data', FIGURES_ROOT / 'jpg')
OUT.mkdir(parents=True, exist_ok=True)
s = json.loads((PROJECT_ROOT / '03_analysis/06_land_transitions/data/analysis_summary.json').read_text(encoding='utf-8'))
z = np.load(PROJECT_ROOT / '03_analysis/06_land_transitions/data/lucc_dsi_statistics.npz')
short = ['Forest', 'Grassland', 'Cropland', 'Wetlands', 'Settlements', 'Other land']
areas, counts, quantiles = (z['area_km2'], z['counts'], z['summary_quantiles'])
percent = areas / areas.sum() * 100
selected = z['display_transitions']
labels = [f'{short[t // 6]} → {short[t % 6]}' for t in selected]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 6, 'font.weight': 'normal', 'axes.titlesize': 8, 'axes.titleweight': 'normal', 'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 6, 'axes.linewidth': 0.6, 'axes.edgecolor': '#8B959C', 'text.color': '#30363D', 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial', 'mathtext.bf': 'Arial:bold', 'mathtext.default': 'regular'})
cmap = LinearSegmentedColormap.from_list('DSI', ['#2166AC', '#92C5DE', '#F7F7F7', '#F4A582', '#B2182B'])
cmap.set_bad('#F0F1F2')
area_cmap = LinearSegmentedColormap.from_list('area', ['#FAFCFC', '#C8DADD', '#628D99', '#284F61'])
area_cmap.set_bad('#F0F1F2')
medians = quantiles[:, 0, 2] * 1000
assert s['schema_version'] == 4
limit = s['color_scale']['symmetric_limit']
norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
fig = plt.figure(figsize=(17 / 2.54, 11 / 2.54))
for panel in range(2):
    ax = fig.add_axes([0.115 if panel == 0 else 0.51, 0.585, 0.39, 0.355])
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(5.5, -0.5)
    ax.set_xticks(range(6), short, fontsize=6)
    ax.set_yticks(range(6), short if panel == 0 else [], fontsize=6)
    ax.set_xlabel('Final class', fontsize=7, labelpad=4)
    if panel == 0:
        ax.set_ylabel('Initial class', fontsize=7, labelpad=6)
    ax.set_title('(a) Transition area' if panel == 0 else '(b) Median DSI trend', loc='left', fontsize=8, fontweight='normal', pad=6)
    for k in np.arange(-0.5, 6, 1):
        ax.axvline(k, color='#DEE3E6', lw=0.6, zorder=0)
        ax.axhline(k, color='#DEE3E6', lw=0.6, zorder=0)
    for t in range(36):
        y, x = divmod(t, 6)
        if panel == 0:
            if x == y:
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='#EDF1F2', edgecolor='none', zorder=-1))
            if counts[t]:
                ax.scatter(x, y, s=440 * np.sqrt(percent[t] / percent.max()), color='#78A0AB', alpha=0.75, edgecolors='none')
                label = f'{percent[t]:.2f}' if percent[t] >= 0.01 else '<0.01'
                ax.text(x, y, label, ha='center', va='center', fontsize=6)
            else:
                ax.text(x, y, '—', color='#B7BEC3', ha='center', va='center', fontsize=6)
        else:
            color = cmap(norm(medians[t])) if counts[t] else '#F0F1F2'
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor='#DEE3E6', lw=0.5))
            value = ('0.00' if round(medians[t], 2) == 0 else f'{medians[t]:+.2f}') + ('†' if counts[t] < s['small_cell_flag_n'] else '') if counts[t] else '—'
            ax.text(x, y, value, ha='center', va='center', fontsize=6, color='white' if counts[t] and abs(medians[t]) > limit * 0.7 else '#30363D')
    ax.tick_params(length=0)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.91, 0.585, 0.012, 0.355]), orientation='vertical', extend='both', extendrect=True)
cb.set_label('Median DSI trend ($10^{-3}$ yr$^{-1}$)', fontsize=6, labelpad=3)
cb.ax.yaxis.label.set_gid('colorbar_title')
cb.ax.tick_params(labelsize=6, length=2)
ax = fig.add_axes([0.17, 0.11, 0.73, 0.355])
fig.text(0.17, 0.49, '(c) Major transitions and same-origin unchanged-endpoint references', fontsize=8, fontweight='normal')
for j, t in enumerate(selected):
    q = quantiles[t, 0] * 1000
    diagonal = t // 6 == t % 6
    color = '#B5BDBE' if diagonal else '#789BA9'
    if diagonal:
        ax.axhspan(j - 0.45, j + 0.45, color='#F0F3F3', zorder=0)
    ax.plot([q[0], q[4]], [j, j], color=color, lw=1.5)
    ax.vlines([q[0], q[4]], j - 0.09, j + 0.09, color=color, lw=0.9)
    ax.add_patch(Rectangle((q[1], j - 0.25), q[3] - q[1], 0.5, facecolor=color, edgecolor='none'))
    ax.vlines(q[2], j - 0.25, j + 0.25, color='#344852', lw=1.2)
    if j < len(selected) - 1 and selected[j + 1] // 6 != t // 6:
        ax.axhline(j + 0.5, color='#D5DBDE', lw=0.7)
    ax.text(1.03, j, f'{percent[t]:.2f}%', transform=ax.get_yaxis_transform(), va='center', fontsize=6)
ax.text(1.03, 1.045, 'Area', transform=ax.transAxes, fontsize=6, color='#65737B')
ax.set_yticks(range(len(selected)), labels, fontsize=6)
ax.set_ylim(len(selected) - 0.5, -0.5)
ax.axvline(0, color='#687A86', lw=0.8, ls='--')
ax.grid(axis='x', color='#E0E5E8', lw=0.5)
ax.set_axisbelow(True)
ax.set_xlabel('DSI trend ($10^{-3}$ yr$^{-1}$)', fontsize=7)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='x', labelsize=7)
ax.tick_params(axis='y', length=0)
runpy.run_path(str(CODE_ROOT / '04_plotting/code/land_transitions/05_export_figure.py'), init_globals={'fig': fig, 'base': 'Figure11_LUCC_DSI', 'OUT': OUT, 'target_width_cm': 17})
plt.close(fig)
runpy.run_path(str(CODE_ROOT / '04_plotting/code/land_transitions/04_plot_figure12_network.py'))
print('Figures 11 and 12 exported.', flush=True)
