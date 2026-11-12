"""Plot DSI trends and significance (Figure 7)."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, study_boundary, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, save_main_figure
from pathlib import Path
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
DATA_DIR = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'
TREND_PATH = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_2001_2021.tif'
PVALUE_PATH = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_significance_pvalue_2001_2021.tif'
SIGNIFICANT_TREND_PATH = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_p_lt_0.05_2001_2021.tif'
STUDY_AREA_PATH = study_boundary()
FIGURE_PATH = FIGURES_ROOT / 'jpg/Fig7.jpg'
STRETCH_MIN = -0.001
STRETCH_MAX = 0.001
PVALUE_THRESHOLD = 0.05
HISTOGRAM_PERCENTILES = (1.0, 99.0)
HISTOGRAM_BIN_COUNT = 100
FIGURE_DPI = 400
BASE_FIGURE_WIDTH_INCHES = 14.0
BASE_FIGURE_HEIGHT_INCHES = 8.8
TARGET_FIGURE_WIDTH_CM = 17.0
CENTIMETERS_PER_INCH = 2.54
FIGURE_WIDTH = TARGET_FIGURE_WIDTH_CM / CENTIMETERS_PER_INCH
SIZE_SCALE = FIGURE_WIDTH / BASE_FIGURE_WIDTH_INCHES
FIGURE_HEIGHT = BASE_FIGURE_HEIGHT_INCHES * SIZE_SCALE

def scaled_size(value):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    return value * SIZE_SCALE
SOFT_GREEN = '#72A98F'
PALE_NEUTRAL = '#F4F0E6'
SOFT_RED = '#D9827C'
SOFT_GRAY = '#D4D4D0'
OUTLINE_COLOR = '#303030'
OUTLINE_WIDTH = 0.175
AXIS_COLOR = '#5B6570'
TEXT_COLOR = '#343432'
SIGNIFICANCE_LINE_COLOR = '#425D78'
BASE_FONT_SIZE = 7
PANEL_TITLE_FONT_SIZE = 8
SECONDARY_FONT_SIZE = 6
for input_path in (PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_2001_2021.tif', PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_significance_pvalue_2001_2021.tif', PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_p_lt_0.05_2001_2021.tif'):
    if not input_path.is_file():
        raise FileNotFoundError(f'Input raster does not exist: {input_path}')
if not STUDY_AREA_PATH.is_file():
    raise FileNotFoundError(f'Study-area vector does not exist: {STUDY_AREA_PATH}')
with rasterio.open(TREND_PATH) as trend_source:
    if trend_source.count != 1:
        raise RuntimeError(f'Trend raster must have one band: {TREND_PATH}')
    trend = trend_source.read(1, masked=True)
    reference_width = trend_source.width
    reference_height = trend_source.height
    reference_crs = trend_source.crs
    reference_transform = trend_source.transform
    reference_bounds = trend_source.bounds
with rasterio.open(PVALUE_PATH) as pvalue_source:
    if pvalue_source.count != 1:
        raise RuntimeError(f'P-value raster must have one band: {PVALUE_PATH}')
    if pvalue_source.width != reference_width or pvalue_source.height != reference_height or pvalue_source.crs != reference_crs or (not pvalue_source.transform.almost_equals(reference_transform)):
        raise RuntimeError('P-value and trend rasters are not spatially aligned.')
    pvalue = pvalue_source.read(1, masked=True)
with rasterio.open(SIGNIFICANT_TREND_PATH) as significant_source:
    if significant_source.count != 1:
        raise RuntimeError(f'Significant-trend raster must have one band: {SIGNIFICANT_TREND_PATH}')
    if significant_source.width != reference_width or significant_source.height != reference_height or significant_source.crs != reference_crs or (not significant_source.transform.almost_equals(reference_transform)):
        raise RuntimeError('Significant-trend and trend rasters are not spatially aligned.')
    significant_trend = significant_source.read(1, masked=True)
study_area = gpd.read_file(STUDY_AREA_PATH)
if study_area.empty or study_area.geometry.isna().all():
    raise RuntimeError(f'Study-area vector contains no valid geometry: {STUDY_AREA_PATH}')
if study_area.crs is None:
    raise RuntimeError(f'Study-area vector has no CRS: {STUDY_AREA_PATH}')
if reference_crs is None:
    raise RuntimeError(f'Trend raster has no CRS: {TREND_PATH}')
if study_area.crs != reference_crs:
    study_area = study_area.to_crs(reference_crs)
trend_values = np.asarray(trend.data, dtype=np.float64)
pvalue_values = np.asarray(pvalue.data, dtype=np.float64)
common_valid = ~np.ma.getmaskarray(trend) & ~np.ma.getmaskarray(pvalue) & np.isfinite(trend_values) & np.isfinite(pvalue_values)
row_indices = np.arange(reference_height, dtype=np.int32)
column_indices = np.zeros(reference_height, dtype=np.int32)
_, row_latitudes = rasterio.transform.xy(reference_transform, row_indices, column_indices, offset='center')
row_area_weights = np.cos(np.deg2rad(np.asarray(row_latitudes)))
row_area_weights = np.clip(row_area_weights, 0.0, None)
area_weights = np.broadcast_to(row_area_weights[:, None], trend_values.shape)
significant_decrease = common_valid & (trend_values < 0.0) & (pvalue_values < PVALUE_THRESHOLD)
significant_increase = common_valid & (trend_values > 0.0) & (pvalue_values < PVALUE_THRESHOLD)
not_significant = common_valid & ~(significant_decrease | significant_increase)
total_weight = np.sum(area_weights[common_valid])
area_percentages = np.array([np.sum(area_weights[significant_decrease]), np.sum(area_weights[not_significant]), np.sum(area_weights[significant_increase])], dtype=np.float64)
area_percentages = area_percentages / total_weight * 100.0
trend_percentiles = np.percentile(trend_values[common_valid], HISTOGRAM_PERCENTILES)
histogram_limit = float(np.max(np.abs(trend_percentiles)))
histogram_edges = np.linspace(-histogram_limit, histogram_limit, HISTOGRAM_BIN_COUNT + 1)
histogram_display = common_valid & (trend_values >= -histogram_limit) & (trend_values <= histogram_limit)
histogram_total_weight = np.sum(area_weights[histogram_display])
all_histogram = np.histogram(trend_values[histogram_display], bins=histogram_edges, weights=area_weights[histogram_display])[0]
decrease_histogram = np.histogram(trend_values[significant_decrease & histogram_display], bins=histogram_edges, weights=area_weights[significant_decrease & histogram_display])[0]
increase_histogram = np.histogram(trend_values[significant_increase & histogram_display], bins=histogram_edges, weights=area_weights[significant_increase & histogram_display])[0]
all_bin_percentages = all_histogram / histogram_total_weight * 100.0
decrease_bin_percentages = decrease_histogram / histogram_total_weight * 100.0
increase_bin_percentages = increase_histogram / histogram_total_weight * 100.0
histogram_centers = (histogram_edges[:-1] + histogram_edges[1:]) / 2.0
significant_histogram = decrease_histogram + increase_histogram
significant_proportion = np.full(all_histogram.shape, np.nan, dtype=np.float64)
np.divide(significant_histogram * 100.0, all_histogram, out=significant_proportion, where=all_histogram > 0.0)
trend_cmap = LinearSegmentedColormap.from_list('soft_green_neutral_red', [SOFT_GREEN, PALE_NEUTRAL, SOFT_RED], N=256)
trend_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
extent = (reference_bounds.left, reference_bounds.right, reference_bounds.bottom, reference_bounds.top)
data_crs = ccrs.PlateCarree()
map_crs = ccrs.PlateCarree()
longitude_ticks = [-150, -100, -50, 0, 50, 100, 150]
latitude_ticks = [0, 20, 40, 60]
plt.rcParams.update({'font.family': 'Arial', 'font.size': BASE_FONT_SIZE, 'axes.titlesize': PANEL_TITLE_FONT_SIZE, 'axes.labelsize': BASE_FONT_SIZE, 'axes.labelpad': scaled_size(4), 'axes.titlepad': scaled_size(6), 'xtick.labelsize': BASE_FONT_SIZE, 'ytick.labelsize': BASE_FONT_SIZE, 'axes.linewidth': 0.35, 'xtick.major.size': scaled_size(3.5), 'ytick.major.size': scaled_size(3.5), 'xtick.major.width': scaled_size(0.8), 'ytick.major.width': scaled_size(0.8), 'xtick.major.pad': scaled_size(3.5), 'ytick.major.pad': scaled_size(3.5), 'lines.linewidth': scaled_size(1.5), 'patch.linewidth': scaled_size(1.0), 'xtick.direction': 'out', 'ytick.direction': 'out', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42})
figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
outer_grid = figure.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 0.78], left=0.06, right=0.99, top=0.995, bottom=0.06, hspace=0.17)
map_grid = outer_grid[0].subgridspec(2, 1, hspace=-0.08)
statistics_grid = outer_grid[1].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.23)
map_axes = [figure.add_subplot(map_grid[0, 0], projection=map_crs), figure.add_subplot(map_grid[1, 0], projection=map_crs)]
histogram_axis = figure.add_subplot(statistics_grid[0, 0])
area_axis = figure.add_subplot(statistics_grid[0, 1])
histogram_original_position = histogram_axis.get_position()
map_row_grid_position = outer_grid[0].get_position(figure)
histogram_title_x = map_row_grid_position.x0 + 0.012 * map_row_grid_position.width
area_title_x = area_axis.get_position().x0
histogram_title_y = histogram_original_position.y1 + scaled_size(8) / 72.0 / FIGURE_HEIGHT
histogram_frame_extension = 0.042
histogram_expanded_position = [histogram_original_position.x0, histogram_original_position.y0, histogram_original_position.width, histogram_original_position.height + histogram_frame_extension]
map_titles = ['(a) DSI trend, 2001–2021', f'(b) DSI trend where p < {PVALUE_THRESHOLD}']
map_images = []
for axis, map_data, title in zip(map_axes, (trend, significant_trend), map_titles):
    axis.set_extent(extent, crs=data_crs)
    map_image = axis.imshow(map_data, cmap=trend_cmap, vmin=STRETCH_MIN, vmax=STRETCH_MAX, extent=extent, origin='upper', interpolation='nearest', transform=data_crs, zorder=1)
    axis.add_geometries(study_area.geometry, crs=data_crs, facecolor='none', edgecolor=OUTLINE_COLOR, linewidth=OUTLINE_WIDTH, alpha=0.9, zorder=3)
    axis.set_xticks(longitude_ticks, crs=data_crs)
    axis.set_yticks(latitude_ticks, crs=data_crs)
    axis.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='°', number_format='.0f'))
    axis.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='°', number_format='.0f'))
    axis.gridlines(crs=data_crs, draw_labels=False, xlocs=longitude_ticks, ylocs=latitude_ticks, color='#777777', linewidth=0.125, linestyle=':', alpha=0.3, zorder=2)
    axis.spines['geo'].set_edgecolor(OUTLINE_COLOR)
    axis.spines['geo'].set_linewidth(0.35)
    axis.text(0.012, 0.965, title, transform=axis.transAxes, ha='left', va='top', fontsize=PANEL_TITLE_FONT_SIZE, fontweight='bold', color=TEXT_COLOR, zorder=5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'pad': 1.0})
    map_images.append(map_image)
first_image = map_images[0]
map_axes[0].tick_params(labelbottom=False)
map_axes[1].add_patch(Rectangle((0.845, 0.045), 0.14, 0.22, transform=map_axes[1].transAxes, facecolor='white', edgecolor='none', alpha=0.8, zorder=4))
colorbar_axis = map_axes[1].inset_axes([0.865, 0.175, 0.105, 0.038], zorder=5)
colorbar = figure.colorbar(first_image, cax=colorbar_axis, orientation='horizontal', ticks=[STRETCH_MIN, STRETCH_MAX])
colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=SECONDARY_FONT_SIZE, direction='out', length=scaled_size(2.5), width=scaled_size(0.6), pad=scaled_size(1.5))
colorbar.outline.set_edgecolor(AXIS_COLOR)
colorbar.outline.set_linewidth(scaled_size(0.6))
colorbar.set_label('Trend (DSI yr$^{-1}$)', fontsize=SECONDARY_FONT_SIZE, color=TEXT_COLOR, labelpad=scaled_size(3))
colorbar.ax.xaxis.set_label_position('top')
histogram_axis.stairs(all_bin_percentages, histogram_edges, fill=True, color=SOFT_GRAY, alpha=0.72, linewidth=scaled_size(0.7), label='All pixels')
histogram_axis.stairs(decrease_bin_percentages, histogram_edges, fill=True, color=SOFT_GREEN, alpha=0.78, linewidth=scaled_size(0.7), label='Significant decrease')
histogram_axis.stairs(increase_bin_percentages, histogram_edges, fill=True, color=SOFT_RED, alpha=0.78, linewidth=scaled_size(0.7), label='Significant increase')
histogram_axis.axvline(0.0, color=OUTLINE_COLOR, linewidth=scaled_size(0.8), linestyle='--', zorder=4)
histogram_axis.set_xlim(-histogram_limit, histogram_limit)
histogram_axis.set_ylim(0.0, max(float(np.max(all_bin_percentages)), float(np.max(decrease_bin_percentages)), float(np.max(increase_bin_percentages))) * 1.48)
histogram_axis.set_xlabel('Trend (DSI yr$^{-1}$)')
histogram_axis.set_ylabel('Area proportion per bin (%)')
histogram_axis.ticklabel_format(axis='x', style='sci', scilimits=(-3, -3), useMathText=True)
histogram_axis.grid(axis='y', color='#C9C9C6', linewidth=scaled_size(0.4), linestyle='--', alpha=0.55)
histogram_axis.set_axisbelow(True)
histogram_axis.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.88, fontsize=SECONDARY_FONT_SIZE)
significance_axis = histogram_axis.twinx()
histogram_axis.set_position(histogram_expanded_position)
significance_axis.set_position(histogram_expanded_position)
significance_axis.plot(histogram_centers, significant_proportion, color=SIGNIFICANCE_LINE_COLOR, linewidth=scaled_size(1.6), zorder=6)
significance_axis.set_ylim(0.0, 130.0)
significance_axis.set_yticks([0, 25, 50, 75, 100])
significance_axis.set_ylabel('Significant proportion (%)', color=SIGNIFICANCE_LINE_COLOR)
significance_axis.tick_params(axis='y', colors=SIGNIFICANCE_LINE_COLOR)
significance_axis.spines['right'].set_color(SIGNIFICANCE_LINE_COLOR)
significance_axis.grid(False)
figure.text(histogram_title_x, histogram_title_y, '(c) Area-weighted trend distribution', ha='left', va='bottom', fontsize=PANEL_TITLE_FONT_SIZE, fontweight='bold', color=TEXT_COLOR)
figure.text(area_title_x, histogram_title_y, '(d) Area proportions', ha='left', va='bottom', fontsize=PANEL_TITLE_FONT_SIZE, fontweight='bold', color=TEXT_COLOR)
area_labels = ['Significant\ndecrease', 'Not\nsignificant', 'Significant\nincrease']
area_colors = [SOFT_GREEN, SOFT_GRAY, SOFT_RED]
x_positions = np.arange(len(area_labels))
area_bars = area_axis.bar(x_positions, area_percentages, width=0.62, color=area_colors, edgecolor='white', linewidth=scaled_size(0.8))
area_ymax = max(80.0, float(np.ceil(np.max(area_percentages) / 10.0) * 10.0 + 10.0))
for bar, percentage in zip(area_bars, area_percentages):
    area_axis.text(bar.get_x() + bar.get_width() / 2.0, percentage + area_ymax * 0.02, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=SECONDARY_FONT_SIZE, color=TEXT_COLOR)
area_axis.set_xlim(-0.55, len(area_labels) - 0.45)
area_axis.set_ylim(0.0, area_ymax)
area_axis.set_xticks(x_positions, labels=area_labels)
area_axis.tick_params(axis='x', labelsize=SECONDARY_FONT_SIZE)
area_axis.set_ylabel('Area proportion (%)')
area_axis.grid(axis='y', color='#C9C9C6', linewidth=scaled_size(0.4), linestyle='--', alpha=0.55)
area_axis.set_axisbelow(True)
area_axis.spines['right'].set_visible(False)
area_axis.spines['top'].set_visible(False)
save_main_figure(figure, 7, bbox_inches='tight', pad_inches=0.02, facecolor='white', pil_kwargs={'quality': 95, 'subsampling': 0})
plt.close(figure)
print(f'Area-weighted percentages (significant decrease, not significant, significant increase): {area_percentages}')
print(f'Histogram display limits: {-histogram_limit:.8f} to {histogram_limit:.8f} (based on the {HISTOGRAM_PERCENTILES[0]:.0f}%—{HISTOGRAM_PERCENTILES[1]:.0f}% percentile range)')
print(f'Figure saved: {FIGURE_PATH}')
