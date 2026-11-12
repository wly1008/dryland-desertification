"""Plot PV-NPV-bare soil trend combinations and DSI responses (Figure 10)."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, save_main_figure, project_file
from pathlib import Path
import json
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
ROOT = PROJECT_ROOT / '03_analysis/05_cover_combinations'
DATA, OUT = (PROJECT_ROOT / '03_analysis/05_cover_combinations/data', FIGURES_ROOT / 'jpg')
OUT.mkdir(parents=True, exist_ok=True)
s = json.loads((PROJECT_ROOT / '03_analysis/05_cover_combinations/data/main_dsi_response_summary.json').read_text(encoding='utf-8'))
f = json.loads((PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_combination_summary.json').read_text(encoding='utf-8'))
z = np.load(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/main_response_series.npz')
top = s['top_ids']
symbols = ['↓', 'NS', '↑']
palette = ['#B2B9B3', '#669F96', '#C89C69', '#8FAB73', '#AB7B9D', '#718FB3', '#C88281', '#A89C76']
up_color, ns_color, down_color = ('#C88780', '#D6DADD', '#759FAF')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'axes.titleweight': 'normal', 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.5, 'axes.edgecolor': '#8B959C', 'text.color': '#30363D', 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
with rasterio.open(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/paired_fvc_combination.tif') as src:
    labels, bounds, crs = (src.read(1), src.bounds, src.crs)
study = gpd.read_file(project_file(s['study_path'])).to_crs(crs)
fig = plt.figure(figsize=(17.3 / 2.54, 10.1 / 2.54))
projection = ccrs.PlateCarree()
axis = fig.add_axes([0.055, 0.685, 0.93, 0.3], projection=projection)
display = np.full(labels.shape, 255, np.uint8)
display[labels != 255] = len(top)
for j, g in enumerate(top):
    display[labels == g] = j
cmap = ListedColormap([*palette[:len(top)], '#E4E5E7'])
cmap.set_bad((1, 1, 1, 1))
axis.imshow(np.ma.masked_equal(display, 255), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], origin='upper', transform=projection, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, len(top) + 1.5), len(top) + 1), interpolation='nearest', rasterized=True)
axis.set_facecolor('white')
axis.add_geometries(list(study.geometry), crs=projection, facecolor='none', edgecolor='#7D868C', linewidth=0.15, alpha=0.5)
axis.set_extent([-180, 180, -1, 68], crs=projection)
axis.set_xticks([-150, -100, -50, 0, 50, 100, 150], crs=projection)
axis.set_yticks([0, 20, 40, 60], crs=projection)
axis.xaxis.set_major_formatter(LongitudeFormatter())
axis.yaxis.set_major_formatter(LatitudeFormatter())
axis.tick_params(length=2, width=0.4, pad=2)
axis.gridlines(xlocs=[-150, -100, -50, 0, 50, 100, 150], ylocs=[0, 20, 40, 60], linewidth=0.3, color='#D4DADF', zorder=0)
axis.text(0.012, 0.965, '(a)  Cover-change pathways', transform=axis.transAxes, ha='left', va='top', fontsize=8, fontweight='bold')
handles = []
for j, g in enumerate(top):
    rec = s['classes'][g - 1]
    desc = '/'.join((symbols[state] for state in rec['states']))
    handles.append(Patch(facecolor=palette[j], label=f"C{g:02d}  {desc}  {rec['paired_area_percent']:.1f}%"))
rest = 100 - sum((s['classes'][g - 1]['paired_area_percent'] for g in top))
handles.append(Patch(facecolor='#E4E5E7', label=f'Other  {rest:.1f}%'))
fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.055, 0.57, 0.93, 0.08), mode='expand', ncol=5, frameon=False, fontsize=6, columnspacing=1.0, handlelength=1.1, labelspacing=0.7)
fig.text(0.07, 0.54, '(b) Joint component trends', fontsize=8, fontweight='bold')
axis = fig.add_axes([0.08, 0.14, 0.36, 0.365])
edges = z['plane_edges_pp_per_year']
centers = (edges[:-1] + edges[1:]) / 2
xx, yy = np.meshgrid(centers, centers)
bin_area, bin_bs = (z['plane_area_km2'], z['plane_bs_trend_pp_per_year'])
nonempty = bin_area > 0
bs_limit = max(0.4, float(np.ceil(np.nanpercentile(np.abs(bin_bs), 98) / 0.2) * 0.2))
sizes = 5 + 100 * np.sqrt(bin_area[nonempty] / bin_area.max())
im = axis.scatter(xx[nonempty], yy[nonempty], c=bin_bs[nonempty], s=sizes, marker='s', edgecolors='none', cmap='BrBG_r', norm=TwoSlopeNorm(vmin=-bs_limit, vcenter=0, vmax=bs_limit), rasterized=True)
axis.axhline(0, color='#8D989F', lw=0.55, zorder=0)
axis.axvline(0, color='#8D989F', lw=0.55, zorder=0)
axis.set_xlim(edges[0], edges[-1])
axis.set_ylim(edges[0], edges[-1])
axis.set_xlabel('PV trend (percentage points yr$^{-1}$)')
axis.set_ylabel('NPV trend (percentage points yr$^{-1}$)')
axis.spines[['top', 'right']].set_visible(False)
cb_ticks = [-bs_limit, -bs_limit / 2, 0, bs_limit / 2, bs_limit]
cb = fig.colorbar(im, cax=fig.add_axes([0.452, 0.125, 0.01, 0.38]), orientation='vertical', ticks=cb_ticks, extend='both')
cb.set_label('Mean BS trend (percentage points yr$^{-1}$)', fontsize=6, labelpad=5)
cb.ax.yaxis.set_label_position('right')
cb.ax.yaxis.set_ticks_position('right')
cb.ax.tick_params(length=2, pad=2, labelsize=6)
cb.outline.set_linewidth(0.4)
fig.text(0.548, 0.54, '(c) DSI trends', fontsize=8, fontweight='bold')
axis = fig.add_axes([0.58, 0.14, 0.235, 0.365])
tick_labels = []
for j, g in enumerate(top):
    quant = z['dsi_trend_quantiles_per_year'][g - 1] * 1000
    axis.plot([quant[0], quant[4]], [j, j], color=palette[j], lw=1.1)
    axis.plot([quant[0], quant[0]], [j - 0.08, j + 0.08], color=palette[j], lw=0.8)
    axis.plot([quant[4], quant[4]], [j - 0.08, j + 0.08], color=palette[j], lw=0.8)
    axis.add_patch(Rectangle((quant[1], j - 0.22), quant[3] - quant[1], 0.44, facecolor=palette[j], edgecolor='white', linewidth=0.4, alpha=0.9))
    axis.plot([quant[2], quant[2]], [j - 0.22, j + 0.22], color='#35434C', lw=1.0)
    tick_labels.append(f'C{g:02d}')
axis.axvline(0, color='#7A858D', lw=0.6, ls='--')
axis.set_yticks(range(len(top)), tick_labels)
axis.tick_params(axis='y', labelsize=6)
axis.set_ylim(len(top) - 0.5, -0.5)
axis.set_xlabel('DSI trend ($10^{-3}$ yr$^{-1}$)')
axis.grid(axis='x', color='#E2E6E8', lw=0.35)
axis.spines[['top', 'right', 'left']].set_visible(False)
axis.tick_params(length=2, width=0.4)
fig.text(0.85, 0.54, '(d) Direction', fontsize=8, fontweight='bold')
axis = fig.add_axes([0.85, 0.14, 0.13, 0.365])
for j, g in enumerate(top):
    lower, upper = s['classes'][g - 1]['dsi_direction_percent_decrease_increase']
    axis.barh(j, lower, color=down_color, height=0.44)
    axis.barh(j, upper, left=lower, color=up_color, height=0.44)
    axis.text(101.5, j, f'{upper:.0f}%', va='center', fontsize=6, color='#865149')
axis.set_xlim(0, 119)
axis.set_xticks([0, 50, 100])
axis.set_ylim(len(top) - 0.5, -0.5)
axis.set_yticks([])
axis.set_xlabel('Area (%)')
axis.spines[['top', 'right', 'left']].set_visible(False)
axis.tick_params(length=2, width=0.4)
legend_axis = fig.add_axes([0.818, 0.14, 0.025, 0.365])
legend_axis.set_axis_off()
legend_axis.text(0.5, 0.82, 'DSI ↑', rotation=90, ha='center', va='center', fontsize=6, color='#66727B')
legend_axis.add_patch(Rectangle((0.28, 0.61), 0.44, 0.11, transform=legend_axis.transAxes, facecolor=up_color, edgecolor='none'))
legend_axis.text(0.5, 0.34, 'DSI ↓', rotation=90, ha='center', va='center', fontsize=6, color='#66727B')
legend_axis.add_patch(Rectangle((0.28, 0.13), 0.44, 0.11, transform=legend_axis.transAxes, facecolor=down_color, edgecolor='none'))
destination = FIGURES_ROOT / 'jpg/Fig10.jpg'
save_main_figure(fig, 10, bbox_inches='tight', pad_inches=0.01, pil_kwargs={'quality': 95, 'subsampling': 0})
print(destination, flush=True)
plt.close(fig)
