"""Plot spatial DSI comparisons (Figure 6)."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, study_boundary, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, save_main_figure
from pathlib import Path
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from cartopy.mpl.ticker import LatitudeFormatter
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INPUT_DIR = PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'
INPUT_PATHS = {2001: PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_2001.tif', 2011: PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_2011.tif', 2021: PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_2021.tif'}
VECTOR_PATH = study_boundary()
OUTPUT_DIR = FIGURES_ROOT / 'jpg'
OUTPUT_PATH = FIGURES_ROOT / 'jpg/Fig6.jpg'
MEAN_START_YEAR = 2001
MEAN_END_YEAR = 2021
INPUT_NAME_TEMPLATE = 'DI_{year}.tif'
BAND_INDEX = 1
EXPECTED_RESOLUTION_DEG = 0.1
PIXELS_PER_GROUP = 10
RESOLUTION_TOLERANCE = 1e-09
OVERWRITE = True
STATIC_MIN = 0.4
STATIC_MAX = 0.8
DIFFERENCE_MIN = -0.05
DIFFERENCE_MAX = 0.05
DIFFERENCE_PROFILE_MIN = -0.03
DIFFERENCE_PROFILE_MAX = 0.03
BASE_FIGURE_WIDTH_INCHES = 14.0
BASE_FIGURE_HEIGHT_INCHES = 16.0
TARGET_FIGURE_WIDTH_CM = 17.0
CENTIMETERS_PER_INCH = 2.54
FIGURE_WIDTH = TARGET_FIGURE_WIDTH_CM / CENTIMETERS_PER_INCH
SIZE_SCALE = FIGURE_WIDTH / BASE_FIGURE_WIDTH_INCHES
FIGURE_HEIGHT = BASE_FIGURE_HEIGHT_INCHES * SIZE_SCALE
DPI = 400

def scaled_size(value):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    return value * SIZE_SCALE
STATIC_COLORS = ['#F3EFE3', '#D7DEC4', '#B7CCB4', '#8FB4A6', '#5C6678']
STATIC_CLASS_BOUNDARIES = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0]
STATIC_CLASS_TICKS = [0.5, 0.6, 0.7, 0.8]
DIFFERENCE_COLORS = ['#567FA6', '#9DB8CD', '#F4F1EA', '#D9A59A', '#B45F55']
STATIC_COLORMAP = ListedColormap(STATIC_COLORS, name='soft_static_classes')
DIFFERENCE_COLORMAP = LinearSegmentedColormap.from_list('soft_difference', DIFFERENCE_COLORS, N=256)
STATIC_COLORMAP.set_bad(color='#F8F8F5', alpha=1.0)
DIFFERENCE_COLORMAP.set_bad(color='#F8F8F5', alpha=1.0)
STATIC_NORM = BoundaryNorm(STATIC_CLASS_BOUNDARIES, ncolors=STATIC_COLORMAP.N, clip=True)
DIFFERENCE_NORM = TwoSlopeNorm(vmin=DIFFERENCE_MIN, vcenter=0.0, vmax=DIFFERENCE_MAX)
YEAR_LINE_COLORS = {2001: '#557FA3', 2011: '#C8895C', 2021: '#6F9876'}
YEAR_FILL_COLORS = {2001: '#B8CAD9', 2011: '#E2C3A9', 2021: '#BDD1C0'}
NEGATIVE_COLOR = '#6F9876'
POSITIVE_COLOR = '#BC7165'
GRID_COLOR = '#D9DEE3'
AXIS_COLOR = '#5B6570'
TEXT_COLOR = '#333333'
PANEL_LABEL_FONT_SIZE = 8
BASE_FONT_SIZE = 7
SECONDARY_FONT_SIZE = 6
LINE_WIDTH = scaled_size(1.35)
FILL_ALPHA = 0.34
GRID_ALPHA = 0.62
VECTOR_LINE_COLOR = '#4E565D'
VECTOR_LINE_WIDTH = scaled_size(0.52)
VECTOR_LINE_ALPHA = 0.88
PANEL_TITLES = ['(d)', '(f)', '(h)', '(j)', '(l)']
LATITUDE_PANEL_LABELS = ['(e)', '(g)', '(i)', '(k)', '(m)']
LONGITUDE_TICKS = np.arange(-150, 181, 50)
LONGITUDE_TICK_LABELS = ['150°W', '100°W', '50°W', '0°', '50°E', '100°E', '150°E']
if PIXELS_PER_GROUP <= 0:
    raise ValueError('PIXELS_PER_GROUP must be a positive integer.')
if DPI <= 0:
    raise ValueError('DPI must be positive.')
if STATIC_MIN >= STATIC_MAX:
    raise ValueError('STATIC_MIN must be less than STATIC_MAX.')
if DIFFERENCE_MIN >= 0.0 or DIFFERENCE_MAX <= 0.0:
    raise ValueError('Difference-map limits must span zero.')
if DIFFERENCE_PROFILE_MIN >= 0.0 or DIFFERENCE_PROFILE_MAX <= 0.0:
    raise ValueError('Difference-profile limits must span zero.')
missing_paths = [path for path in INPUT_PATHS.values() if not path.is_file()]
if missing_paths:
    raise FileNotFoundError('Missing input rasters:\n' + '\n'.join((str(path) for path in missing_paths)))
if not VECTOR_PATH.is_file():
    raise FileNotFoundError(f'Boundary vector file does not exist: {VECTOR_PATH}')
vector_data = gpd.read_file(VECTOR_PATH)
if vector_data.crs is None:
    raise RuntimeError(f'Boundary vector has no CRS: {VECTOR_PATH}')
vector_data = vector_data.to_crs('EPSG:4326')
vector_geometries = [geometry for geometry in vector_data.geometry if geometry is not None and (not geometry.is_empty)]
if not vector_geometries:
    raise RuntimeError(f'Boundary vector has no valid geometry: {VECTOR_PATH}')
annual_data = {}
reference_width = None
reference_height = None
reference_crs = None
reference_transform = None
for year in (2001, 2011, 2021):
    input_path = INPUT_PATHS[year]
    print(f'Read {year}：{input_path}')
    with rasterio.open(input_path) as source:
        if BAND_INDEX < 1 or BAND_INDEX > source.count:
            raise ValueError(f'Band index out of range: {input_path} has {source.count} bands; current BAND_INDEX={BAND_INDEX}。')
        if abs(source.transform.b) > RESOLUTION_TOLERANCE or abs(source.transform.d) > RESOLUTION_TOLERANCE:
            raise ValueError(f'Rotated or sheared raster transforms are unsupported: {input_path}')
        x_resolution = abs(float(source.transform.a))
        y_resolution = abs(float(source.transform.e))
        if abs(x_resolution - EXPECTED_RESOLUTION_DEG) > RESOLUTION_TOLERANCE or abs(y_resolution - EXPECTED_RESOLUTION_DEG) > RESOLUTION_TOLERANCE:
            raise ValueError(f'Raster resolution differs from the expected {EXPECTED_RESOLUTION_DEG}°：{input_path}\nActual resolution: {x_resolution}° × {y_resolution}°。')
        if reference_width is None:
            reference_width = source.width
            reference_height = source.height
            reference_crs = source.crs
            reference_transform = source.transform
        elif source.width != reference_width or source.height != reference_height or source.crs != reference_crs or (not source.transform.almost_equals(reference_transform)):
            raise RuntimeError(f"Annual rasters have mismatched dimensions, CRS, or affine transforms.\nReference: {PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_2001.tif'}\nMismatched raster: {input_path}")
        masked_data = source.read(BAND_INDEX, masked=True)
        values = np.asarray(masked_data.filled(np.nan), dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        annual_data[year] = values
difference_2011_2001 = np.full(annual_data[2001].shape, np.nan, dtype=np.float64)
valid_2011_2001 = np.isfinite(annual_data[2001]) & np.isfinite(annual_data[2011])
difference_2011_2001[valid_2011_2001] = annual_data[2011][valid_2011_2001] - annual_data[2001][valid_2011_2001]
difference_2021_2011 = np.full(annual_data[2011].shape, np.nan, dtype=np.float64)
valid_2021_2011 = np.isfinite(annual_data[2011]) & np.isfinite(annual_data[2021])
difference_2021_2011[valid_2021_2011] = annual_data[2021][valid_2021_2011] - annual_data[2011][valid_2021_2011]
mean_years = range(MEAN_START_YEAR, MEAN_END_YEAR + 1)
mean_input_paths = {year: INPUT_DIR / INPUT_NAME_TEMPLATE.format(year=year) for year in mean_years}
missing_mean_paths = [path for path in mean_input_paths.values() if not path.is_file()]
if missing_mean_paths:
    raise FileNotFoundError('Missing rasters for the 21-year mean:\n' + '\n'.join((str(path) for path in missing_mean_paths)))
study_period_sum = np.zeros(annual_data[2001].shape, dtype=np.float64)
study_period_common_valid = np.ones(annual_data[2001].shape, dtype=bool)
for year in mean_years:
    if year in annual_data:
        current_mean_data = annual_data[year]
    else:
        current_mean_path = mean_input_paths[year]
        print(f'Read annual input for 21-year mean: {year}：{current_mean_path}')
        with rasterio.open(current_mean_path) as source:
            if BAND_INDEX < 1 or BAND_INDEX > source.count:
                raise ValueError(f'Band index out of range: {current_mean_path} has {source.count} bands; current BAND_INDEX={BAND_INDEX}。')
            if source.width != reference_width or source.height != reference_height or source.crs != reference_crs or (not source.transform.almost_equals(reference_transform)):
                raise RuntimeError(f"Mean-input rasters have mismatched dimensions, CRS, or affine transforms.\nReference: {PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_2001.tif'}\nMismatched raster: {current_mean_path}")
            current_masked_data = source.read(BAND_INDEX, masked=True)
            current_mean_data = np.asarray(current_masked_data.filled(np.nan), dtype=np.float64)
            current_mean_data[~np.isfinite(current_mean_data)] = np.nan
    current_mean_valid = np.isfinite(current_mean_data)
    study_period_sum[current_mean_valid] += current_mean_data[current_mean_valid]
    study_period_common_valid &= current_mean_valid
study_period_mean = np.full(annual_data[2001].shape, np.nan, dtype=np.float64)
study_period_mean[study_period_common_valid] = study_period_sum[study_period_common_valid] / len(mean_input_paths)
study_period_mean_values = study_period_mean[study_period_common_valid]
study_period_pixel_latitudes = reference_transform.f + (np.arange(reference_height, dtype=np.float64) + 0.5) * reference_transform.e
study_period_row_area_weights = np.cos(np.deg2rad(study_period_pixel_latitudes))
if np.any(study_period_row_area_weights <= 0.0):
    raise RuntimeError('Invalid study-area latitudes prevent pie-chart area weighting.')
study_period_common_area_weights = np.broadcast_to(study_period_row_area_weights[:, np.newaxis], study_period_mean.shape)[study_period_common_valid]
study_period_class_masks = [study_period_mean_values < 0.5, (study_period_mean_values >= 0.5) & (study_period_mean_values < 0.6), (study_period_mean_values >= 0.6) & (study_period_mean_values < 0.7), (study_period_mean_values >= 0.7) & (study_period_mean_values < 0.8), study_period_mean_values >= 0.8]
study_period_class_area_weights = np.array([np.sum(study_period_common_area_weights[current_class_mask]) for current_class_mask in study_period_class_masks], dtype=np.float64)
if not np.isfinite(study_period_class_area_weights).all() or np.sum(study_period_class_area_weights) <= 0.0:
    raise RuntimeError('No pixels are valid in all 21 years for area statistics.')
study_period_class_percentages = study_period_class_area_weights / np.sum(study_period_class_area_weights) * 100.0
print(f'Area shares of 2001-2021 mean DSI (<0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, >=0.8): {np.round(study_period_class_percentages, 4)}')
panel_data = [annual_data[2001], difference_2011_2001, annual_data[2011], difference_2021_2011, annual_data[2021]]
panel_types = ['static', 'difference', 'static', 'difference', 'static']
panel_years = [2001, None, 2011, None, 2021]
longitude_pixel_centers = reference_transform.c + (np.arange(reference_width, dtype=np.float64) + 0.5) * reference_transform.a
latitude_pixel_centers = reference_transform.f + (np.arange(reference_height, dtype=np.float64) + 0.5) * reference_transform.e
longitude_left = float(np.min(longitude_pixel_centers) - x_resolution / 2.0)
longitude_right = float(np.max(longitude_pixel_centers) + x_resolution / 2.0)
latitude_bottom = float(np.min(latitude_pixel_centers) - y_resolution / 2.0)
latitude_top = float(np.max(latitude_pixel_centers) + y_resolution / 2.0)
longitude_group_count = int(np.ceil(reference_width / PIXELS_PER_GROUP))
latitude_group_count = int(np.ceil(reference_height / PIXELS_PER_GROUP))
longitude_coordinates = np.empty(longitude_group_count, dtype=np.float64)
for group_index in range(longitude_group_count):
    column_start = group_index * PIXELS_PER_GROUP
    column_end = min(column_start + PIXELS_PER_GROUP, reference_width)
    longitude_coordinates[group_index] = np.mean(longitude_pixel_centers[column_start:column_end])
latitude_coordinates = np.empty(latitude_group_count, dtype=np.float64)
for group_index in range(latitude_group_count):
    row_start = group_index * PIXELS_PER_GROUP
    row_end = min(row_start + PIXELS_PER_GROUP, reference_height)
    latitude_coordinates[group_index] = np.mean(latitude_pixel_centers[row_start:row_end])
longitude_means = []
latitude_means = []
for current_data in panel_data:
    current_longitude_means = np.full(longitude_group_count, np.nan, dtype=np.float64)
    for group_index in range(longitude_group_count):
        column_start = group_index * PIXELS_PER_GROUP
        column_end = min(column_start + PIXELS_PER_GROUP, reference_width)
        group_data = current_data[:, column_start:column_end]
        valid = np.isfinite(group_data)
        if np.any(valid):
            current_longitude_means[group_index] = np.mean(group_data[valid])
    current_latitude_means = np.full(latitude_group_count, np.nan, dtype=np.float64)
    for group_index in range(latitude_group_count):
        row_start = group_index * PIXELS_PER_GROUP
        row_end = min(row_start + PIXELS_PER_GROUP, reference_height)
        group_data = current_data[row_start:row_end, :]
        valid = np.isfinite(group_data)
        if np.any(valid):
            current_latitude_means[group_index] = np.mean(group_data[valid])
    longitude_means.append(current_longitude_means)
    latitude_means.append(current_latitude_means)
for panel_index in range(len(panel_data)):
    if not np.any(np.isfinite(latitude_means[panel_index])):
        raise RuntimeError(f'{PANEL_TITLES[panel_index]} has no valid pixels for latitude statistics.')
if OUTPUT_PATH.exists() and (not OVERWRITE):
    raise FileExistsError(f'Output exists; set OVERWRITE=True to replace it: {OUTPUT_PATH}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': BASE_FONT_SIZE, 'axes.labelsize': BASE_FONT_SIZE, 'axes.labelpad': scaled_size(4), 'axes.titlepad': scaled_size(6), 'xtick.labelsize': BASE_FONT_SIZE, 'ytick.labelsize': BASE_FONT_SIZE, 'axes.linewidth': scaled_size(0.8), 'xtick.major.size': scaled_size(3.5), 'ytick.major.size': scaled_size(3.5), 'xtick.major.width': scaled_size(0.8), 'ytick.major.width': scaled_size(0.8), 'xtick.major.pad': scaled_size(3.5), 'ytick.major.pad': scaled_size(3.5), 'lines.linewidth': scaled_size(1.5), 'patch.linewidth': scaled_size(1.0), 'axes.unicode_minus': False})
projection = ccrs.PlateCarree()
figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
grid = figure.add_gridspec(nrows=9, ncols=2, width_ratios=(5.0, 1.0), height_ratios=(0.55, 0.55, 1.0, 1.0, 1.0, 1.0, 1.0, 0.55, 0.55), wspace=0.055, hspace=0.24)
top_difference_axis = figure.add_subplot(grid[0, 0])
top_annual_axis = figure.add_subplot(grid[1, 0], sharex=top_difference_axis)
top_pie_axis = figure.add_subplot(grid[0:2, 1])
map_axes = []
latitude_axes = []
map_images = []
for panel_index in range(5):
    map_axis = figure.add_subplot(grid[panel_index + 2, 0], projection=projection)
    latitude_axis = figure.add_subplot(grid[panel_index + 2, 1])
    current_type = panel_types[panel_index]
    if current_type == 'static':
        current_colormap = STATIC_COLORMAP
        current_norm = STATIC_NORM
    else:
        current_colormap = DIFFERENCE_COLORMAP
        current_norm = DIFFERENCE_NORM
    map_image = map_axis.imshow(np.ma.masked_invalid(panel_data[panel_index]), extent=(longitude_left, longitude_right, latitude_bottom, latitude_top), origin='upper', transform=projection, cmap=current_colormap, norm=current_norm, interpolation='nearest', zorder=1)
    map_axis.set_extent([longitude_left, longitude_right, latitude_bottom, latitude_top], crs=projection)
    map_axis.set_aspect('auto')
    map_axis.add_geometries(vector_geometries, crs=projection, facecolor='none', edgecolor=VECTOR_LINE_COLOR, linewidth=VECTOR_LINE_WIDTH, alpha=VECTOR_LINE_ALPHA, zorder=5)
    map_axis.set_yticks([0, 20, 40, 60], crs=projection)
    map_axis.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='°'))
    map_axis.tick_params(axis='y', colors=TEXT_COLOR, direction='out', length=scaled_size(2.8), width=scaled_size(0.65))
    map_axis.tick_params(axis='x', bottom=False, labelbottom=False)
    map_axis.gridlines(crs=projection, draw_labels=False, xlocs=np.arange(-150, 181, 50), ylocs=[0, 20, 40, 60], color=GRID_COLOR, linewidth=scaled_size(0.45), linestyle='--', alpha=0.46, zorder=2)
    map_axis.text(0.006, 0.975, PANEL_TITLES[panel_index], transform=map_axis.transAxes, ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='normal', color=TEXT_COLOR, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72, 'pad': 1.0}, zorder=10)
    map_axis.spines['geo'].set_color(AXIS_COLOR)
    map_axis.spines['geo'].set_linewidth(scaled_size(0.75))
    current_latitude_means = latitude_means[panel_index]
    current_valid = np.isfinite(current_latitude_means)
    if current_type == 'static':
        current_year = panel_years[panel_index]
        latitude_axis.plot(current_latitude_means, latitude_coordinates, color=YEAR_LINE_COLORS[current_year], linewidth=LINE_WIDTH, solid_capstyle='round', zorder=3)
        latitude_axis.fill_betweenx(latitude_coordinates, STATIC_MIN, current_latitude_means, where=current_valid, interpolate=True, color=YEAR_FILL_COLORS[current_year], alpha=FILL_ALPHA, linewidth=0, zorder=2)
        latitude_axis.set_xlim(STATIC_MIN, STATIC_MAX)
        latitude_axis.set_xticks([0.4, 0.6, 0.8])
    else:
        positive_latitude_means = np.where(current_valid & (current_latitude_means >= 0.0), current_latitude_means, np.nan)
        negative_latitude_means = np.where(current_valid & (current_latitude_means <= 0.0), current_latitude_means, np.nan)
        latitude_axis.plot(positive_latitude_means, latitude_coordinates, color=POSITIVE_COLOR, linewidth=LINE_WIDTH, solid_capstyle='round', zorder=4)
        latitude_axis.plot(negative_latitude_means, latitude_coordinates, color=NEGATIVE_COLOR, linewidth=LINE_WIDTH, solid_capstyle='round', zorder=4)
        latitude_axis.fill_betweenx(latitude_coordinates, 0.0, current_latitude_means, where=current_valid & (current_latitude_means >= 0.0), interpolate=True, color=POSITIVE_COLOR, alpha=0.42, linewidth=0, zorder=2)
        latitude_axis.fill_betweenx(latitude_coordinates, 0.0, current_latitude_means, where=current_valid & (current_latitude_means < 0.0), interpolate=True, color=NEGATIVE_COLOR, alpha=0.42, linewidth=0, zorder=2)
        latitude_axis.axvline(0.0, color=AXIS_COLOR, linewidth=scaled_size(0.65), linestyle='--', zorder=1)
        latitude_axis.set_xlim(DIFFERENCE_PROFILE_MIN, DIFFERENCE_PROFILE_MAX)
        latitude_axis.set_xticks([-0.03, 0.0, 0.03])
    latitude_axis.set_ylim(latitude_bottom, latitude_top)
    latitude_axis.tick_params(axis='x', colors=TEXT_COLOR, direction='out', length=scaled_size(2.8), width=scaled_size(0.65))
    latitude_axis.tick_params(axis='y', left=False, labelleft=False)
    latitude_axis.grid(axis='x', color=GRID_COLOR, linewidth=scaled_size(0.55), linestyle='--', alpha=GRID_ALPHA, zorder=1)
    latitude_axis.spines['top'].set_visible(False)
    latitude_axis.spines['right'].set_visible(False)
    latitude_axis.spines['left'].set_color(AXIS_COLOR)
    latitude_axis.spines['bottom'].set_color(AXIS_COLOR)
    latitude_axis.spines['left'].set_linewidth(scaled_size(0.75))
    latitude_axis.spines['bottom'].set_linewidth(scaled_size(0.75))
    latitude_axis.text(0.035, 0.96, LATITUDE_PANEL_LABELS[panel_index], ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='normal', color=TEXT_COLOR, transform=latitude_axis.transAxes, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72, 'pad': 1.0}, zorder=10)
    map_axes.append(map_axis)
    latitude_axes.append(latitude_axis)
    map_images.append(map_image)
latitude_axes[-1].set_xlabel('Mean DSI', color=TEXT_COLOR, labelpad=scaled_size(2))
top_annual_means = longitude_means[0]
top_annual_valid = np.isfinite(top_annual_means)
top_annual_line = top_annual_axis.plot(longitude_coordinates, top_annual_means, color=YEAR_LINE_COLORS[2001], linewidth=LINE_WIDTH, label='2001 DSI', solid_capstyle='round', zorder=5)[0]
top_annual_axis.fill_between(longitude_coordinates, STATIC_MIN, top_annual_means, where=top_annual_valid, interpolate=True, color=YEAR_FILL_COLORS[2001], alpha=0.25, linewidth=0, zorder=2)
top_difference_means = longitude_means[1]
top_difference_valid = np.isfinite(top_difference_means)
top_positive_difference_means = np.where(top_difference_valid & (top_difference_means >= 0.0), top_difference_means, np.nan)
top_negative_difference_means = np.where(top_difference_valid & (top_difference_means <= 0.0), top_difference_means, np.nan)
top_difference_axis.plot(longitude_coordinates, top_positive_difference_means, color=POSITIVE_COLOR, linewidth=LINE_WIDTH, solid_capstyle='round', zorder=6)
top_difference_axis.plot(longitude_coordinates, top_negative_difference_means, color=NEGATIVE_COLOR, linewidth=LINE_WIDTH, solid_capstyle='round', zorder=6)
top_difference_axis.fill_between(longitude_coordinates, 0.0, top_difference_means, where=top_difference_valid & (top_difference_means >= 0.0), interpolate=True, color=POSITIVE_COLOR, alpha=0.34, linewidth=0, zorder=3)
top_difference_axis.fill_between(longitude_coordinates, 0.0, top_difference_means, where=top_difference_valid & (top_difference_means < 0.0), interpolate=True, color=NEGATIVE_COLOR, alpha=0.34, linewidth=0, zorder=3)
top_difference_axis.axhline(0.0, color=AXIS_COLOR, linewidth=scaled_size(0.55), linestyle='--', alpha=0.75, zorder=1)
top_annual_axis.set_xlim(longitude_left, longitude_right)
top_annual_axis.set_ylim(STATIC_MIN, STATIC_MAX)
top_annual_axis.set_xticks(LONGITUDE_TICKS)
top_annual_axis.set_xticklabels(LONGITUDE_TICK_LABELS)
top_annual_axis.set_yticks([0.4, 0.6, 0.8])
top_difference_axis.set_ylim(DIFFERENCE_PROFILE_MIN, DIFFERENCE_PROFILE_MAX)
top_difference_axis.set_yticks([-0.03, 0.0, 0.03])
top_difference_axis.set_xticks(LONGITUDE_TICKS)
top_annual_axis.grid(axis='y', color=GRID_COLOR, linewidth=scaled_size(0.6), linestyle='--', alpha=GRID_ALPHA, zorder=1)
top_annual_axis.tick_params(axis='both', colors=TEXT_COLOR, direction='out', length=scaled_size(2.8), width=scaled_size(0.65))
top_annual_axis.tick_params(axis='x', bottom=False, labelbottom=False)
top_annual_axis.tick_params(axis='y', colors=TEXT_COLOR)
top_difference_axis.tick_params(axis='both', colors=TEXT_COLOR, direction='out', length=scaled_size(2.8), width=scaled_size(0.65))
top_difference_axis.tick_params(axis='y', colors=TEXT_COLOR)
top_difference_axis.tick_params(axis='x', bottom=False, labelbottom=False)
top_difference_axis.grid(axis='y', color=GRID_COLOR, linewidth=scaled_size(0.6), linestyle='--', alpha=GRID_ALPHA, zorder=1)
top_annual_axis.spines['top'].set_visible(False)
top_annual_axis.spines['right'].set_visible(False)
top_annual_axis.spines['left'].set_color(AXIS_COLOR)
top_annual_axis.spines['bottom'].set_color(AXIS_COLOR)
top_annual_axis.spines['left'].set_linewidth(scaled_size(0.75))
top_annual_axis.spines['bottom'].set_linewidth(scaled_size(0.75))
top_difference_axis.spines['top'].set_visible(False)
top_difference_axis.spines['right'].set_visible(False)
top_difference_axis.spines['left'].set_color(AXIS_COLOR)
top_difference_axis.spines['bottom'].set_color(AXIS_COLOR)
top_difference_axis.spines['left'].set_linewidth(scaled_size(0.75))
top_difference_axis.spines['bottom'].set_linewidth(scaled_size(0.75))
for label_axis, panel_label in ((top_difference_axis, '(a)'), (top_annual_axis, '(c)')):
    label_axis.text(0.006, 0.95, panel_label, ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='normal', color=TEXT_COLOR, transform=label_axis.transAxes, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72, 'pad': 1.0}, zorder=10)
bottom_annual_axis = figure.add_subplot(grid[7, 0], sharex=top_annual_axis)
bottom_difference_axis = figure.add_subplot(grid[8, 0], sharex=bottom_annual_axis)
bottom_annual_information_axis = figure.add_subplot(grid[7, 1])
bottom_difference_information_axis = figure.add_subplot(grid[8, 1])
bottom_annual_means = longitude_means[2]
bottom_annual_valid = np.isfinite(bottom_annual_means)
bottom_annual_line = bottom_annual_axis.plot(longitude_coordinates, bottom_annual_means, color=YEAR_LINE_COLORS[2011], linewidth=LINE_WIDTH, label='2011 DSI', solid_capstyle='round', zorder=5)[0]
bottom_annual_axis.fill_between(longitude_coordinates, STATIC_MIN, bottom_annual_means, where=bottom_annual_valid, interpolate=True, color=YEAR_FILL_COLORS[2011], alpha=0.25, linewidth=0, zorder=2)
bottom_difference_means = longitude_means[3]
bottom_difference_valid = np.isfinite(bottom_difference_means)
bottom_positive_difference_means = np.where(bottom_difference_valid & (bottom_difference_means >= 0.0), bottom_difference_means, np.nan)
bottom_negative_difference_means = np.where(bottom_difference_valid & (bottom_difference_means <= 0.0), bottom_difference_means, np.nan)
bottom_difference_axis.plot(longitude_coordinates, bottom_positive_difference_means, color=POSITIVE_COLOR, linewidth=LINE_WIDTH, solid_capstyle='round', zorder=6)
bottom_difference_axis.plot(longitude_coordinates, bottom_negative_difference_means, color=NEGATIVE_COLOR, linewidth=LINE_WIDTH, solid_capstyle='round', zorder=6)
bottom_difference_axis.fill_between(longitude_coordinates, 0.0, bottom_difference_means, where=bottom_difference_valid & (bottom_difference_means >= 0.0), interpolate=True, color=POSITIVE_COLOR, alpha=0.34, linewidth=0, zorder=3)
bottom_difference_axis.fill_between(longitude_coordinates, 0.0, bottom_difference_means, where=bottom_difference_valid & (bottom_difference_means < 0.0), interpolate=True, color=NEGATIVE_COLOR, alpha=0.34, linewidth=0, zorder=3)
bottom_difference_axis.axhline(0.0, color=AXIS_COLOR, linewidth=scaled_size(0.55), linestyle='--', alpha=0.75, zorder=1)
bottom_annual_axis.set_xlim(longitude_left, longitude_right)
bottom_annual_axis.set_ylim(STATIC_MAX, STATIC_MIN)
bottom_annual_axis.set_xticks(LONGITUDE_TICKS)
bottom_annual_axis.set_xticklabels(LONGITUDE_TICK_LABELS)
bottom_annual_axis.set_yticks([0.8, 0.6, 0.4])
bottom_annual_axis.xaxis.tick_top()
bottom_difference_axis.set_ylim(DIFFERENCE_PROFILE_MAX, DIFFERENCE_PROFILE_MIN)
bottom_difference_axis.set_yticks([-0.03, 0.0, 0.03])
bottom_difference_axis.set_xticks(LONGITUDE_TICKS)
bottom_annual_axis.grid(axis='y', color=GRID_COLOR, linewidth=scaled_size(0.6), linestyle='--', alpha=GRID_ALPHA, zorder=1)
bottom_annual_axis.tick_params(axis='both', colors=TEXT_COLOR, direction='out', length=scaled_size(2.8), width=scaled_size(0.65))
bottom_annual_axis.tick_params(axis='y', colors=TEXT_COLOR)
bottom_difference_axis.tick_params(axis='both', colors=TEXT_COLOR, direction='out', length=scaled_size(2.8), width=scaled_size(0.65))
bottom_difference_axis.tick_params(axis='y', colors=TEXT_COLOR)
bottom_difference_axis.tick_params(axis='x', bottom=False, labelbottom=False)
bottom_difference_axis.grid(axis='y', color=GRID_COLOR, linewidth=scaled_size(0.6), linestyle='--', alpha=GRID_ALPHA, zorder=1)
bottom_annual_axis.spines['bottom'].set_visible(False)
bottom_annual_axis.spines['right'].set_visible(False)
bottom_annual_axis.spines['left'].set_color(AXIS_COLOR)
bottom_annual_axis.spines['top'].set_color(AXIS_COLOR)
bottom_annual_axis.spines['left'].set_linewidth(scaled_size(0.75))
bottom_annual_axis.spines['top'].set_linewidth(scaled_size(0.75))
bottom_difference_axis.spines['bottom'].set_visible(False)
bottom_difference_axis.spines['top'].set_color(AXIS_COLOR)
bottom_difference_axis.spines['right'].set_visible(False)
bottom_difference_axis.spines['left'].set_color(AXIS_COLOR)
bottom_difference_axis.spines['left'].set_linewidth(scaled_size(0.75))
bottom_difference_axis.spines['top'].set_linewidth(scaled_size(0.75))
for label_axis, panel_label in ((bottom_annual_axis, '(n)'), (bottom_difference_axis, '(o)')):
    label_axis.text(0.006, 0.05, panel_label, ha='left', va='bottom', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='normal', color=TEXT_COLOR, transform=label_axis.transAxes, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72, 'pad': 1.0}, zorder=10)
pie_wedges, _ = top_pie_axis.pie(study_period_class_area_weights, colors=STATIC_COLORS, startangle=90, counterclock=False, wedgeprops={'edgecolor': 'white', 'linewidth': scaled_size(0.65)})
for class_index, wedge in enumerate(pie_wedges):
    middle_angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
    percentage_text = f'{study_period_class_percentages[class_index]:.1f}%'
    if class_index in (0, 4):
        anchor_x = np.cos(middle_angle) * 0.9
        anchor_y = np.sin(middle_angle) * 0.9
        text_x = np.cos(middle_angle) * 1.18
        text_y = np.sin(middle_angle) * 1.18
        top_pie_axis.annotate(percentage_text, xy=(anchor_x, anchor_y), xytext=(text_x, text_y), ha='left' if text_x >= 0.0 else 'right', va='center', fontsize=SECONDARY_FONT_SIZE, color=TEXT_COLOR, arrowprops={'arrowstyle': '-', 'color': AXIS_COLOR, 'linewidth': scaled_size(0.55)})
    else:
        top_pie_axis.text(np.cos(middle_angle) * 0.66, np.sin(middle_angle) * 0.66, percentage_text, ha='center', va='center', fontsize=SECONDARY_FONT_SIZE, color=TEXT_COLOR)
top_pie_axis.text(0.5, 0.0, '2001–2021 mean DSI', ha='center', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='normal', color=TEXT_COLOR, transform=top_pie_axis.transAxes)
top_pie_axis.text(0.02, 0.98, '(b)', ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='normal', color=TEXT_COLOR, transform=top_pie_axis.transAxes, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72, 'pad': 1.0}, zorder=10)
top_pie_axis.set_aspect('equal')
bottom_annual_information_axis.set_axis_off()
static_colorbar_axis = bottom_annual_information_axis.inset_axes([0.12, 0.42, 0.76, 0.16])
static_colorbar = figure.colorbar(map_images[0], cax=static_colorbar_axis, orientation='horizontal', boundaries=STATIC_CLASS_BOUNDARIES, spacing='uniform', drawedges=True)
static_colorbar.set_ticks(STATIC_CLASS_TICKS)
static_colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=SECONDARY_FONT_SIZE, direction='out', length=scaled_size(2.5), width=scaled_size(0.6), pad=scaled_size(1.5))
static_colorbar.outline.set_edgecolor(AXIS_COLOR)
static_colorbar.outline.set_linewidth(scaled_size(0.6))
static_colorbar.dividers.set_color('white')
static_colorbar.dividers.set_linewidth(scaled_size(0.5))
static_colorbar.set_label('Annual DSI', fontsize=BASE_FONT_SIZE, color=TEXT_COLOR, labelpad=scaled_size(3))
bottom_difference_information_axis.set_axis_off()
difference_colorbar_axis = bottom_difference_information_axis.inset_axes([0.12, 0.5, 0.76, 0.16])
difference_colorbar = figure.colorbar(map_images[1], cax=difference_colorbar_axis, orientation='horizontal')
difference_colorbar.set_ticks([-0.05, 0.0, 0.05])
difference_colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=SECONDARY_FONT_SIZE, direction='out', length=scaled_size(2.5), width=scaled_size(0.6))
difference_colorbar.outline.set_edgecolor(AXIS_COLOR)
difference_colorbar.outline.set_linewidth(scaled_size(0.6))
difference_colorbar.set_label('DSI difference', fontsize=BASE_FONT_SIZE, color=TEXT_COLOR, labelpad=scaled_size(3))
figure.subplots_adjust(left=0.06, right=0.985, bottom=0.045, top=0.982)
save_main_figure(figure, 6, bbox_inches='tight', pad_inches=0.02, facecolor='white', edgecolor='none', pil_kwargs={'quality': 95, 'subsampling': 0})
plt.close(figure)
print(f'Figure saved: {OUTPUT_PATH}')
print(f'Annual-map limits: {STATIC_MIN} ～ {STATIC_MAX}; difference-map limits: {DIFFERENCE_MIN} ～ {DIFFERENCE_MAX}; difference-profile limits: {DIFFERENCE_PROFILE_MIN} ～ {DIFFERENCE_PROFILE_MAX}; resolution: {DPI} DPI。')
