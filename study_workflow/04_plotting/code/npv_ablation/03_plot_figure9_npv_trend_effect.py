"""Plot paired NPV trend differences and external agreement (Figure 9)."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, study_boundary, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, save_main_figure
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd
import numpy as np
from pyproj import Geod
import rasterio
ROOT = PROJECT_ROOT / '03_analysis/04_npv_ablation'
DATA_DIR = PROJECT_ROOT / '03_analysis/04_npv_ablation/data/figure9'
FIGURE_DIR = FIGURES_ROOT / 'jpg'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
START_YEAR, END_YEAR = (2004, 2019)
PERIOD = f'{START_YEAR}_{END_YEAR}'
FIGURE_WIDTH_CM = 17.0
FIGURE_HEIGHT_CM = 12.5
FINAL_DPI, JPG_DPI = (400, 400)
DELTA_SLOPE_COLOR_LIMIT = 0.0012
STUDY_AREA_PATH = study_boundary()
TEXT_COLOR = '#30363D'
WITHOUT_COLOR, WITH_COLOR = ('#92999F', '#85648E')
TRANSITION_COLORS = ['#D9DCDE', '#CB7971', '#E7B8A6', '#922F37', '#A6C4D7', '#618EAE', '#25577C']
summary = json.loads((DATA_DIR / f'figure9_statistics_{PERIOD}.json').read_text(encoding='utf-8'))
with rasterio.open(DATA_DIR / f'DSI_trend_difference_with_minus_without_NPV_{PERIOD}.tif') as source:
    delta_slope = source.read(1, masked=True)
    transform, bounds, shape, crs = (source.transform, source.bounds, source.shape, source.crs)
    valid = ~np.ma.getmaskarray(delta_slope) & np.isfinite(delta_slope.data)
with rasterio.open(DATA_DIR / f'DSI_trend_category_transition_{PERIOD}.tif') as source:
    transitions = source.read(1, masked=True)
    if not np.array_equal(np.ma.getmaskarray(transitions), ~valid):
        raise ValueError('The two maps have different common-valid masks.')
study_area = gpd.read_file(STUDY_AREA_PATH).to_crs(crs)
study_geometries = list(study_area.geometry)
matrix = np.asarray(summary['transition_matrix_area_percent'])
pearson = np.asarray(summary['pearson_r_direction_aligned'])
delta_r = np.asarray(summary['delta_r'])
delta_r_ci = np.asarray(summary['delta_r_ci95'])
delta_ba = np.asarray(summary['delta_balanced_agreement_pp'])
delta_ba_ci = np.asarray(summary['delta_balanced_agreement_pp_ci95'])
geod = Geod(ellps='WGS84')
area_by_row = np.empty(shape[0])
for row in range(shape[0]):
    west, east = (transform.c, transform.c + transform.a)
    north, south = (transform.f + row * transform.e, transform.f + (row + 1) * transform.e)
    area, _ = geod.polygon_area_perimeter([west, east, east, west], [north, north, south, south])
    area_by_row[row] = abs(area) / 1000000.0
area_grid = np.broadcast_to(area_by_row[:, None], shape)
saturated_low = float(area_grid[valid & (delta_slope.data < -DELTA_SLOPE_COLOR_LIMIT)].sum() / area_grid[valid].sum() * 100)
saturated_high = float(area_grid[valid & (delta_slope.data > DELTA_SLOPE_COLOR_LIMIT)].sum() / area_grid[valid].sum() * 100)
plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6, 'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR, 'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR, 'axes.edgecolor': '#303030', 'axes.linewidth': 0.35, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
delta_cmap = LinearSegmentedColormap.from_list('NPV_trend_difference', ['#35628D', '#9DB8CD', '#F9F7F2', '#D9A59A', '#A74740'], N=256)
delta_cmap.set_bad('#FFFFFF')
transition_cmap = ListedColormap(TRANSITION_COLORS)
transition_cmap.set_bad('#FFFFFF')
delta_norm = TwoSlopeNorm(vmin=-DELTA_SLOPE_COLOR_LIMIT * 1000, vcenter=0, vmax=DELTA_SLOPE_COLOR_LIMIT * 1000)
transition_norm = BoundaryNorm(np.arange(-0.5, 7.5), transition_cmap.N)
projection = ccrs.PlateCarree()
figure = plt.figure(figsize=(FIGURE_WIDTH_CM / 2.54, FIGURE_HEIGHT_CM / 2.54), facecolor='white')
image_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
map_images = []
map_axes = []
for panel_index, (position, data, title) in enumerate([([0.055, 0.745, 0.91, 0.235], delta_slope * 1000, '(a) Trend difference, $\\Delta\\beta$'), ([0.055, 0.49, 0.91, 0.235], transitions, '(b) Changes in trend classification')]):
    axis = figure.add_axes(position, projection=projection)
    axis.set_facecolor('#FFFFFF')
    map_images.append(axis.imshow(data, origin='upper', extent=image_extent, transform=projection, interpolation='nearest', rasterized=True, cmap=delta_cmap if panel_index == 0 else transition_cmap, norm=delta_norm if panel_index == 0 else transition_norm))
    axis.add_geometries(study_geometries, crs=projection, facecolor='none', edgecolor='#303030', linewidth=0.175, alpha=0.9, zorder=3)
    axis.set_extent([-180, 180, -1, 68], crs=projection)
    axis.set_xticks([-150, -100, -50, 0, 50, 100, 150], crs=projection)
    axis.set_yticks([0, 20, 40, 60], crs=projection)
    axis.xaxis.set_major_formatter(LongitudeFormatter())
    axis.yaxis.set_major_formatter(LatitudeFormatter())
    axis.tick_params(length=2, width=0.35, pad=2, labelbottom=panel_index == 1)
    axis.gridlines(xlocs=[-150, -100, -50, 0, 50, 100, 150], ylocs=[0, 20, 40, 60], linewidth=0.125, color='#777777', linestyle=':', alpha=0.3, zorder=0)
    axis.text(0.012, 0.955, title, transform=axis.transAxes, ha='left', va='top', fontsize=8, fontweight='bold', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1.0}, zorder=5)
    if panel_index == 0:
        axis.text(0.02, 0.075, f'{START_YEAR}–{END_YEAR}', transform=axis.transAxes, ha='left', va='bottom', fontsize=6, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.82, 'pad': 1.5}, zorder=5)
    else:
        axis.text(0.985, 0.955, f"Changed area: {summary['changed_area_percent']:.2f}%", transform=axis.transAxes, ha='right', va='top', fontsize=6, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1.0}, zorder=5)
    map_axes.append(axis)
map_axes[0].add_patch(Rectangle((0.775, 0.02), 0.215, 0.205, transform=map_axes[0].transAxes, facecolor='white', edgecolor='none', alpha=0.86, zorder=5))
colorbar_axis = map_axes[0].inset_axes([0.82, 0.105, 0.155, 0.05], zorder=6)
colorbar = figure.colorbar(map_images[0], cax=colorbar_axis, orientation='horizontal', ticks=[-1.2, 1.2], extend='both', extendrect=True)
colorbar.ax.set_title('$\\Delta\\beta$  ($10^{-3}$ DSI yr$^{-1}$)', fontsize=6, pad=1.5)
colorbar.outline.set_linewidth(0.4)
colorbar.ax.tick_params(length=2, width=0.4, pad=1.0, labelsize=6)
transition_labels = []
short_transitions = ['Unchanged', 'NS → D', 'R → NS', 'R → D', 'D → NS', 'NS → R', 'D → R']
for label, value in zip(short_transitions, summary['transition_area_percent']):
    precision = 3 if 0 < value < 0.05 else 2
    transition_labels.append(f'{label} ({value:.{precision}f}%)')
figure.legend(handles=[Patch(facecolor=color, edgecolor='none', label=label) for color, label in zip(TRANSITION_COLORS, transition_labels)], loc='upper center', bbox_to_anchor=(0.5, 0.455), ncol=7, frameon=False, fontsize=6, handlelength=1.0, handleheight=0.8, columnspacing=0.75, handletextpad=0.35, borderaxespad=0)
figure.text(0.055, 0.375, '(c) Category transitions', fontweight='bold', fontsize=8, va='bottom')
matrix_axis = figure.add_axes([0.065, 0.105, 0.24, 0.25])
matrix_axis.imshow(matrix, cmap='Blues', vmin=0, vmax=80, interpolation='nearest')
matrix_axis.set_xticks([0, 1, 2], ['D', 'NS', 'R'])
matrix_axis.set_yticks([0, 1, 2], ['D', 'NS', 'R'])
matrix_axis.set_xlabel('With NPV', labelpad=4)
matrix_axis.set_ylabel('Without NPV', labelpad=5)
matrix_axis.tick_params(length=0, pad=3, labelsize=6)
matrix_axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
matrix_axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
matrix_axis.grid(which='minor', color='white', linewidth=1.5)
matrix_axis.tick_params(which='minor', bottom=False, left=False)
for row in range(3):
    for column in range(3):
        value = matrix[row, column]
        precision = 3 if 0 < value < 0.05 else 2
        matrix_axis.text(column, row, f'{value:.{precision}f}%', ha='center', va='center', fontsize=6, color='white' if value > 40 else TEXT_COLOR)
    matrix_axis.add_patch(Rectangle((row - 0.5, row - 0.5), 1, 1, facecolor='none', edgecolor='#8195A3', linewidth=0.7))
figure.text(0.375, 0.375, '(d) Continuous agreement', fontweight='bold', fontsize=8, va='bottom')
r_axis = figure.add_axes([0.39, 0.11, 0.245, 0.23])
product_positions = np.array([2, 1, 0])
for product_index, y in enumerate(product_positions):
    r_axis.plot(pearson[product_index], [y, y], color='#B8BEC3', linewidth=1.2, zorder=1)
    r_axis.scatter(pearson[product_index, 0], y, color=WITHOUT_COLOR, s=22, marker='o', zorder=3)
    r_axis.scatter(pearson[product_index, 1], y, color=WITH_COLOR, s=26, marker='D', zorder=4)
    r_axis.text(0.5, y - 0.28, f'Δr* {delta_r[product_index]:+.3f} [{delta_r_ci[product_index, 0]:+.3f}, {delta_r_ci[product_index, 1]:+.3f}]', transform=r_axis.get_yaxis_transform(), ha='center', va='top', fontsize=6, color='#626A72')
r_axis.set_ylim(-0.62, 2.45)
r_axis.set_xlim(0.35, 0.55)
r_axis.set_yticks(product_positions, summary['products'])
r_axis.xaxis.set_major_locator(MultipleLocator(0.05))
r_axis.set_xlabel('Direction-aligned Pearson r*', labelpad=4)
r_axis.spines[['top', 'right', 'left']].set_visible(False)
r_axis.tick_params(axis='y', length=0, pad=8, labelsize=6)
r_axis.tick_params(axis='x', length=2, width=0.45, pad=2, labelsize=7)
r_axis.grid(axis='x', color='#E0E4E7', linewidth=0.4, zorder=0)
r_axis.legend(handles=[Line2D([], [], color=WITHOUT_COLOR, marker='o', linestyle='none', markersize=4, label='Without NPV'), Line2D([], [], color=WITH_COLOR, marker='D', linestyle='none', markersize=4, label='With NPV')], loc='lower center', bbox_to_anchor=(0.42, 1.015), ncol=2, frameon=False, fontsize=6, handlelength=0.8, handletextpad=0.35, columnspacing=1.0, borderaxespad=0)
figure.text(0.65, 0.375, '(e) Class-agreement change', fontweight='bold', fontsize=8, va='bottom')
ba_axis = figure.add_axes([0.66, 0.11, 0.305, 0.23])
ba_axis.axvline(0, color='#6B747B', linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
for product_index, y in enumerate(product_positions):
    ba_axis.plot(delta_ba_ci[product_index], [y, y], color=WITH_COLOR, linewidth=1.1, zorder=2)
    ba_axis.plot(delta_ba_ci[product_index], [y, y], color=WITH_COLOR, linestyle='none', marker='|', markersize=5)
    ba_axis.scatter(delta_ba[product_index], y, color=WITH_COLOR, s=23, marker='D', zorder=3)
    ba_axis.text(delta_ba[product_index], y + 0.22, f'{delta_ba[product_index]:+.2f}', ha='center', va='bottom', fontsize=6)
ba_axis.set_ylim(-0.62, 2.45)
ba_axis.set_xlim(-1.0, 2.2)
ba_axis.set_yticks([])
ba_axis.xaxis.set_major_locator(MultipleLocator(1))
ba_axis.set_xlabel('Δ balanced agreement (pp)', labelpad=4, fontsize=7)
ba_axis.spines[['top', 'right', 'left']].set_visible(False)
ba_axis.tick_params(axis='y', length=0, pad=4)
ba_axis.tick_params(axis='x', length=2, width=0.45, pad=2, labelsize=7)
ba_axis.grid(axis='x', color='#E0E4E7', linewidth=0.4, linestyle=':', zorder=0)
stem = f'Fig9_NPV_trend_effect_{PERIOD}'
save_main_figure(figure, 9, bbox_inches='tight', pad_inches=0, facecolor='white', pil_kwargs={'quality': 95, 'subsampling': 0})
plt.close(figure)
(DATA_DIR / f'figure9_display_audit_{PERIOD}.json').write_text(json.dumps({'symmetric_color_limit_DSI_per_year': DELTA_SLOPE_COLOR_LIMIT, 'saturated_low_area_percent': saturated_low, 'saturated_high_area_percent': saturated_high, 'all_values_retained_in_statistics': True}, indent=2), encoding='utf-8')
print(f'Figure 9 saved: {FIGURE_DIR}')
print(f'Color saturation area: low={saturated_low:.4f}%, high={saturated_high:.4f}%')
