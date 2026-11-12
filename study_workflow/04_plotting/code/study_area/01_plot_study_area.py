"""Plot the study area (Figure 1)."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, save_main_figure
from pathlib import Path
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import numpy as np
import rasterio
import tifffile
from PIL import Image
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Polygon
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape
VERSION_DIR = (CODE_ROOT / '04_plotting/code/study_area/01_plot_study_area.py').parent.parent
STUDY_DIR = VERSION_DIR.parent
DATA_ROOT = PROJECT_ROOT / '02_preprocessing/data'
BIOME_PATH = PROJECT_ROOT / '04_plotting/intermediate/study_area/biome_display_source.tif'
LAND_COVER_DIR = PROJECT_ROOT / '02_preprocessing/data/auxiliary/land_cover_0p1/class'
COMPONENT_ROOT = PROJECT_ROOT / '02_preprocessing/data/monthly_0p1/fvc_original'
CACHE_DIR = PROJECT_ROOT / '04_plotting/intermediate/study_area'
REPORT_DIR = PROJECT_ROOT / '06_verification/study_area'
FIGURE_DIR = FIGURES_ROOT / 'jpg'
LAND_COVER_YEAR = 2021
START_YEAR = 2001
END_YEAR = 2021
COMPONENT_SCALE = 10000.0
RENORMALIZE_COMPONENTS = False
MIN_VALID_MONTHS = 12
REBUILD_COMPONENT_CACHE = False
SHOW_STUDY_OUTLINE = True
SHOW_MISSING_COMPONENTS = True
FIGURE_DPI = 400
TARGET_FIGURE_WIDTH_CM = 17.0
TARGET_FIGURE_HEIGHT_CM = 10.7
FONT_FAMILY = 'Arial'
BASE_FONT_SIZE = 7
PANEL_TITLE_FONT_SIZE = 8
LEGEND_FONT_SIZE = 6
OUTLINE_WIDTH = 0.175
MAP_EXTENT = (-180.0, 180.0, 0.0, 66.65)
COMPONENT_MAP_EXTENT = (-180.0, 180.0, -1.0, 67.0)
LONGITUDE_TICKS = [-150, -100, -50, 0, 50, 100, 150]
LATITUDE_TICKS = [0, 20, 40, 60]
BIOME_DISPLAY_DOWNSAMPLE = 6
RGB_COMPONENT_ORDER = ('NPV', 'PV', 'BS')
COMPONENT_LEGEND_ORDER = ('NPV', 'PV', 'BS')
COMPONENT_COLOR_MODE = 'soft_mix'
COMPONENT_COLORS = {'NPV': '#CCA476', 'PV': '#78A28A', 'BS': '#A7B6C7'}
COMPONENT_BACKGROUND = '#F7F5F0'
TRIANGLE_LEGEND_SIZE_CM = 1.8
TRIANGLE_LEGEND_LEFT_CM = 0.35
TRIANGLE_LEGEND_BOTTOM_CM = 0.15
RIGHT_PANEL_FRACTION = 3.0 / 5.0
STAT_BAR_WIDTH_CM = 0.22 * 3.0
STAT_BAR_HEIGHT_CM = 0.65 * 3.0
STAT_BAR_LEFT_CM = 0.45
STAT_BAR_LABEL_X = -0.2
PANEL_GAP_CM = 0.2
COMPONENT_PANEL_GAP_CM = 0.5
LOCATOR_RIGHT_GAP_CM = 0.85
CATEGORY_LEGEND_FONT_SIZE = 5.6
CATEGORY_MAP_DOWN_DEGREES = 10.0
SHOW_LOCATOR = True
LOCATOR_CENTRAL_LONGITUDE = 30.0
LOCATOR_LONGITUDE_TICKS = [-180, -120, -60, 0, 60, 120]
LOCATOR_BASEMAP = remap_path(cartopy.config['repo_data_dir']) / 'raster' / 'natural_earth' / '50-natural-earth-1-downsampled.png'
BIOME_LABELS = ['T3 Shrublands & shrubby woodlands', 'T4 Savannas & grasslands', 'T5 Deserts & semi-deserts', 'T3 + T4', 'T3 + T5', 'T4 + T5', 'T3 + T4 + T5']
BIOME_COLORS = ['#9D78AA', '#75A483', '#D7B373', '#637B9D', '#B67E74', '#A4AD65', '#665B63']
LAND_LABELS = ['Forest', 'Grassland', 'Cropland', 'Wetlands', 'Settlements', 'Other land']
LAND_COLORS = ['#4B866B', '#A5BC79', '#E4C56C', '#77AFBC', '#C87870', '#CEBFA8']
MISSING_COLOR = '#DADAD6'
TEXT_COLOR = '#343432'
OUTLINE_COLOR = '#3E4548'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
if START_YEAR > END_YEAR:
    raise ValueError('START_YEAR must not exceed END_YEAR')
available_years = sorted((int(path.stem.rsplit('_', 1)[1]) for path in LAND_COVER_DIR.glob('MCD12C1_LULC_CLASS_*.tif')))
if not available_years:
    raise FileNotFoundError('No annual reclassified land-cover raster found')
if LAND_COVER_YEAR is None:
    LAND_COVER_YEAR = available_years[-1]
land_path = LAND_COVER_DIR / f'MCD12C1_LULC_CLASS_{LAND_COVER_YEAR}.tif'
with rasterio.open(land_path) as source:
    land_values = source.read(1)
    grid_transform, grid_crs = (source.transform, source.crs)
    grid_shape, grid_bounds = (source.shape, source.bounds)
    grid_profile = source.profile.copy()
    land_nodata = source.nodata
biome_grid = np.zeros(grid_shape, dtype=np.uint8)
with rasterio.open(BIOME_PATH) as source:
    reproject(rasterio.band(source, 1), biome_grid, src_transform=source.transform, src_crs=source.crs, src_nodata=0, dst_transform=grid_transform, dst_crs=grid_crs, dst_nodata=0, resampling=Resampling.nearest)
    display_shape = (math.ceil(source.height / BIOME_DISPLAY_DOWNSAMPLE), math.ceil(source.width / BIOME_DISPLAY_DOWNSAMPLE))
    biome_display = source.read(1, out_shape=display_shape, resampling=Resampling.nearest)
    display_bounds = source.bounds
latitudes = grid_transform.f + (np.arange(grid_shape[0]) + 0.5) * grid_transform.e
longitudes = grid_transform.c + (np.arange(grid_shape[1]) + 0.5) * grid_transform.a
study_mask = (biome_grid > 0) & (latitudes[:, None] >= MAP_EXTENT[2]) & (latitudes[:, None] <= MAP_EXTENT[3])
biome_grid[~study_mask] = 0
land_valid = study_mask & (land_values != land_nodata) & (land_values >= 1) & (land_values <= 6)
missing_land = study_mask & ~land_valid
if not np.any(study_mask):
    raise ValueError('Biome-based study mask is empty')
outline_geometries = []
if SHOW_STUDY_OUTLINE:
    for geometry, value in shapes(study_mask.astype(np.uint8), mask=study_mask, transform=grid_transform):
        outline_geometries.append(shape(geometry).simplify(0.015, preserve_topology=True))
component_names = ('PV', 'NPV', 'BS')
monthly_files = []
biome_stat = BIOME_PATH.stat()
signature_text = [f'biome={BIOME_PATH}|{biome_stat.st_size}|{biome_stat.st_mtime_ns}', f'grid={grid_shape}|{grid_transform}|{grid_crs}']
for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        paths = [COMPONENT_ROOT / name / str(year) / f'{name}_{year}_{month:02d}.tif' for name in component_names]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            signature_text.append(f'{path}|{stat.st_size}|{stat.st_mtime_ns}')
        monthly_files.append(paths)
signature = hashlib.sha256('\n'.join(signature_text).encode('utf-8')).hexdigest()
mean_path = CACHE_DIR / f'PV_NPV_BS_mean_{START_YEAR}_{END_YEAR}.tif'
count_path = CACHE_DIR / f'PV_NPV_BS_valid_months_{START_YEAR}_{END_YEAR}.tif'
cache_info_path = CACHE_DIR / f'PV_NPV_BS_mean_{START_YEAR}_{END_YEAR}.json'
cache_ok = False
if mean_path.exists() and count_path.exists() and cache_info_path.exists() and (not REBUILD_COMPONENT_CACHE):
    cache_info = json.loads(cache_info_path.read_text(encoding='utf-8'))
    cache_ok = cache_info.get('source_signature') == signature and cache_info.get('component_scale') == COMPONENT_SCALE
if cache_ok:
    with rasterio.open(mean_path) as source:
        if source.shape != grid_shape or source.transform != grid_transform or source.crs != grid_crs:
            raise ValueError('Component cache grid is obsolete; enable REBUILD_COMPONENT_CACHE')
        component_mean = source.read()
    with rasterio.open(count_path) as source:
        month_counts = source.read(1)
    print('Loaded raw-component mean cache', flush=True)
else:
    component_sum = np.zeros((3, *grid_shape), dtype=np.float64)
    month_counts = np.zeros(grid_shape, dtype=np.uint16)
    for month_index, paths in enumerate(monthly_files):
        monthly_values = []
        common_valid = study_mask.copy()
        for path in paths:
            with rasterio.open(path) as source:
                if source.shape != grid_shape or source.crs != grid_crs or source.transform != grid_transform:
                    raise ValueError(f'Component grid mismatch: {path}')
                data = source.read(1)
                common_valid &= (data != source.nodata) & (data >= 0) & (data <= COMPONENT_SCALE)
                monthly_values.append(data)
        for index, data in enumerate(monthly_values):
            component_sum[index, common_valid] += data[common_valid] / COMPONENT_SCALE
        month_counts[common_valid] += 1
        if (month_index + 1) % 12 == 0:
            print(f'Component means: {START_YEAR + month_index // 12} completed', flush=True)
    component_mean = np.full((3, *grid_shape), -9999.0, dtype=np.float32)
    for index in range(3):
        np.divide(component_sum[index], month_counts, out=component_mean[index], where=month_counts > 0)
    profile = grid_profile.copy()
    profile.update(driver='GTiff', dtype='float32', count=3, nodata=-9999.0, compress='deflate', tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(mean_path, 'w', **profile) as output:
        output.write(component_mean)
        output.update_tags(statistic='mean of common-valid monthly raw fractions', years=f'{START_YEAR}-{END_YEAR}', input_scale=str(COMPONENT_SCALE), renormalized='False')
        for index, name in enumerate(component_names, 1):
            output.set_band_description(index, name)
    profile.update(dtype='uint16', count=1, nodata=0)
    with rasterio.open(count_path, 'w', **profile) as output:
        output.write(month_counts, 1)
        output.set_band_description(1, 'common-valid monthly observation count')
    cache_info_path.write_text(json.dumps({'source_signature': signature, 'component_scale': COMPONENT_SCALE, 'period': [START_YEAR, END_YEAR], 'source_month_count': len(monthly_files), 'source_file_count': len(monthly_files) * 3, 'renormalized': False}, indent=2), encoding='utf-8')
    del component_sum
component_valid = study_mask & (month_counts >= MIN_VALID_MONTHS) & np.all(np.isfinite(component_mean), axis=0)
component_valid &= np.all((component_mean >= 0) & (component_mean <= 1), axis=0)
display_fractions = component_mean.copy()
if RENORMALIZE_COMPONENTS:
    fraction_sum = np.sum(component_mean, axis=0)
    component_valid &= fraction_sum > 0
    for index in range(3):
        np.divide(component_mean[index], fraction_sum, out=display_fractions[index], where=component_valid)
rgb = np.zeros((*grid_shape, 4), dtype=np.float32)
if COMPONENT_COLOR_MODE == 'soft_mix':
    background_rgb = np.array(to_rgb(COMPONENT_BACKGROUND), dtype=np.float32)
    rgb[..., :3] = background_rgb
    for index, name in enumerate(component_names):
        color_delta = np.array(to_rgb(COMPONENT_COLORS[name])) - background_rgb
        rgb[..., :3] += np.where(component_valid, display_fractions[index], 0)[..., None] * color_delta
    rgb[..., :3] = np.clip(rgb[..., :3], 0, 1)
elif COMPONENT_COLOR_MODE == 'rgb':
    for channel, name in enumerate(RGB_COMPONENT_ORDER):
        rgb[..., channel] = np.where(component_valid, display_fractions[component_names.index(name)], 0)
else:
    raise ValueError('COMPONENT_COLOR_MODE must be soft_mix or rgb')
rgb[..., 3] = component_valid.astype(np.float32)
if SHOW_MISSING_COMPONENTS:
    rgb[study_mask & ~component_valid, :3] = to_rgb(MISSING_COLOR)
    rgb[study_mask & ~component_valid, 3] = 1.0
area_weights = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], grid_shape)
component_area_means = {name: float(np.average(component_mean[index][component_valid], weights=area_weights[component_valid])) for index, name in enumerate(component_names)}
display_area_means = {name: float(np.average(display_fractions[index][component_valid], weights=area_weights[component_valid])) for index, name in enumerate(component_names)}
plt.rcParams.update({'font.family': FONT_FAMILY, 'font.size': BASE_FONT_SIZE, 'axes.titlesize': PANEL_TITLE_FONT_SIZE, 'axes.labelsize': BASE_FONT_SIZE, 'xtick.labelsize': BASE_FONT_SIZE, 'ytick.labelsize': BASE_FONT_SIZE, 'axes.linewidth': 0.35, 'xtick.major.size': 1.8, 'ytick.major.size': 1.8, 'xtick.major.width': 0.35, 'ytick.major.width': 0.35, 'xtick.major.pad': 2, 'ytick.major.pad': 2, 'xtick.direction': 'out', 'ytick.direction': 'out', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42})
figure = plt.figure(figsize=(TARGET_FIGURE_WIDTH_CM / 2.54, TARGET_FIGURE_HEIGHT_CM / 2.54), dpi=FIGURE_DPI)
data_crs = ccrs.PlateCarree()
full_left_cm = TARGET_FIGURE_WIDTH_CM * 0.06
full_right_cm = TARGET_FIGURE_WIDTH_CM * 0.99
full_width_cm = full_right_cm - full_left_cm
right_width_cm = full_width_cm * RIGHT_PANEL_FRACTION
right_left_cm = full_right_cm - right_width_cm
latitude_ratio = (MAP_EXTENT[3] - MAP_EXTENT[2]) / (MAP_EXTENT[1] - MAP_EXTENT[0])
component_latitude_ratio = (COMPONENT_MAP_EXTENT[3] - COMPONENT_MAP_EXTENT[2]) / (COMPONENT_MAP_EXTENT[1] - COMPONENT_MAP_EXTENT[0])
full_map_height_cm = full_width_cm * component_latitude_ratio
right_map_height_cm = right_width_cm * latitude_ratio
category_map_downshift_cm = right_width_cm / (MAP_EXTENT[1] - MAP_EXTENT[0]) * CATEGORY_MAP_DOWN_DEGREES
component_annotation_height_cm = 0.0
category_annotation_height_cm = 1.45
component_panel_height_cm = full_map_height_cm + component_annotation_height_cm
category_panel_height_cm = right_map_height_cm + category_annotation_height_cm
component_panel_bottom_cm = TARGET_FIGURE_HEIGHT_CM - 0.1 - component_panel_height_cm
biome_panel_top_cm = component_panel_bottom_cm - COMPONENT_PANEL_GAP_CM
biome_panel_bottom_cm = biome_panel_top_cm - category_panel_height_cm
land_panel_top_cm = biome_panel_bottom_cm - PANEL_GAP_CM
land_panel_bottom_cm = land_panel_top_cm - category_panel_height_cm
if land_panel_bottom_cm < 0.08:
    raise ValueError('Insufficient figure height; increase TARGET_FIGURE_HEIGHT_CM')
panels_cm = [[full_left_cm, component_panel_bottom_cm, full_width_cm, component_panel_height_cm], [right_left_cm, biome_panel_bottom_cm, right_width_cm, category_panel_height_cm], [right_left_cm, land_panel_bottom_cm, right_width_cm, category_panel_height_cm]]
annotation_heights_cm = [component_annotation_height_cm, category_annotation_height_cm, category_annotation_height_cm]
map_axes = []
for panel_index, (panel, annotation_height) in enumerate(zip(panels_cm, annotation_heights_cm)):
    left_cm, bottom_cm, width_cm, height_cm = panel
    figure.add_artist(Rectangle((left_cm / TARGET_FIGURE_WIDTH_CM, bottom_cm / TARGET_FIGURE_HEIGHT_CM), width_cm / TARGET_FIGURE_WIDTH_CM, height_cm / TARGET_FIGURE_HEIGHT_CM, transform=figure.transFigure, fill=False, linewidth=0.35, edgecolor=OUTLINE_COLOR, zorder=20))
    axis = figure.add_axes([left_cm / TARGET_FIGURE_WIDTH_CM, (bottom_cm + annotation_height - (category_map_downshift_cm if panel_index > 0 else 0)) / TARGET_FIGURE_HEIGHT_CM, width_cm / TARGET_FIGURE_WIDTH_CM, (height_cm - annotation_height) / TARGET_FIGURE_HEIGHT_CM], projection=data_crs)
    axis.set_extent(COMPONENT_MAP_EXTENT if panel_index == 0 else MAP_EXTENT, crs=data_crs)
    axis.spines['geo'].set_visible(False)
    map_axes.append(axis)
biome_cmap = ListedColormap(BIOME_COLORS)
biome_cmap.set_bad((1, 1, 1, 0))
land_cmap = ListedColormap(LAND_COLORS)
land_cmap.set_bad((1, 1, 1, 0))
map_axes[0].imshow(rgb, extent=(grid_bounds.left, grid_bounds.right, grid_bounds.bottom, grid_bounds.top), origin='upper', interpolation='nearest', transform=data_crs, zorder=1)
map_axes[1].imshow(np.ma.masked_equal(biome_display, 0), cmap=biome_cmap, norm=BoundaryNorm(np.arange(0.5, 8.5), 7), extent=(display_bounds.left, display_bounds.right, display_bounds.bottom, display_bounds.top), origin='upper', interpolation='nearest', transform=data_crs, zorder=1)
map_axes[2].imshow(np.ma.array(land_values, mask=~land_valid), cmap=land_cmap, norm=BoundaryNorm(np.arange(0.5, 7.5), 6), extent=(grid_bounds.left, grid_bounds.right, grid_bounds.bottom, grid_bounds.top), origin='upper', interpolation='nearest', transform=data_crs, zorder=1)
component_title = 'Relative vegetation–soil composition' if RENORMALIZE_COMPONENTS else 'Mean vegetation–soil fractions'
titles = [f'(a) {component_title}, {START_YEAR}–{END_YEAR}', '(c) Ecosystem biomes with potential NPV occurrence', f'(d) Land cover, {LAND_COVER_YEAR}']
for index, (axis, title) in enumerate(zip(map_axes, titles)):
    if SHOW_STUDY_OUTLINE:
        axis.add_geometries(outline_geometries, crs=data_crs, facecolor='none', edgecolor=OUTLINE_COLOR, linewidth=OUTLINE_WIDTH if index == 0 else 0.11, alpha=0.65, zorder=3)
    axis.set_xticks(LONGITUDE_TICKS, crs=data_crs)
    axis.set_yticks(LATITUDE_TICKS, crs=data_crs)
    axis.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='°', number_format='.0f'))
    axis.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='°', number_format='.0f'))
    axis.tick_params(labelsize=BASE_FONT_SIZE if index == 0 else 5.6, bottom=index == 0, labelbottom=index == 0)
    axis.gridlines(crs=data_crs, draw_labels=False, xlocs=LONGITUDE_TICKS, ylocs=LATITUDE_TICKS, color='#777777', linewidth=0.125, linestyle=':', alpha=0.25, zorder=2)
    title_panel_left, title_panel_bottom, title_panel_width, title_panel_height = panels_cm[index]
    title_map_height = title_panel_height - annotation_heights_cm[index]
    figure.text((title_panel_left + 0.012 * title_panel_width) / TARGET_FIGURE_WIDTH_CM, (title_panel_bottom + title_panel_height - 0.034 * title_map_height) / TARGET_FIGURE_HEIGHT_CM, title, ha='left', va='top', fontsize=PANEL_TITLE_FONT_SIZE, fontweight='bold', color=TEXT_COLOR, zorder=25, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 1})
for longitude in LONGITUDE_TICKS:
    x_cm = right_left_cm + (longitude - MAP_EXTENT[0]) / (MAP_EXTENT[1] - MAP_EXTENT[0]) * right_width_cm
    if longitude == 0:
        longitude_text = '0°'
    else:
        longitude_text = f"{abs(longitude)}°{('W' if longitude < 0 else 'E')}"
    figure.add_artist(Line2D([x_cm / TARGET_FIGURE_WIDTH_CM] * 2, [land_panel_bottom_cm / TARGET_FIGURE_HEIGHT_CM, (land_panel_bottom_cm - 0.035) / TARGET_FIGURE_HEIGHT_CM], transform=figure.transFigure, color=OUTLINE_COLOR, linewidth=0.3))
    figure.text(x_cm / TARGET_FIGURE_WIDTH_CM, (land_panel_bottom_cm - 0.1) / TARGET_FIGURE_HEIGHT_CM, longitude_text, ha='center', va='top', fontsize=5.6, color=TEXT_COLOR)
triangle_axis = figure.add_axes([(full_left_cm + TRIANGLE_LEGEND_LEFT_CM) / TARGET_FIGURE_WIDTH_CM, (component_panel_bottom_cm + TRIANGLE_LEGEND_BOTTOM_CM) / TARGET_FIGURE_HEIGHT_CM, TRIANGLE_LEGEND_SIZE_CM / TARGET_FIGURE_WIDTH_CM, TRIANGLE_LEGEND_SIZE_CM / TARGET_FIGURE_HEIGHT_CM])
triangle_height = np.sqrt(3) / 2
triangle_x, triangle_y = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, triangle_height, 260))
weight_pv = triangle_y / triangle_height
weight_bs = triangle_x - 0.5 * weight_pv
weight_npv = 1 - weight_pv - weight_bs
triangle_inside = (weight_pv >= 0) & (weight_bs >= 0) & (weight_npv >= 0)
triangle_rgba = np.zeros((*triangle_x.shape, 4), dtype=np.float32)
for name, weight in [('NPV', weight_npv), ('BS', weight_bs), ('PV', weight_pv)]:
    if COMPONENT_COLOR_MODE == 'soft_mix':
        vertex_color = np.asarray(to_rgb(COMPONENT_COLORS[name]))
    else:
        vertex_color = np.zeros(3)
        vertex_color[RGB_COMPONENT_ORDER.index(name)] = 1
    triangle_rgba[..., :3] += weight[..., None] * vertex_color
triangle_rgba[..., :3] = np.clip(triangle_rgba[..., :3], 0, 1)
triangle_rgba[..., 3] = triangle_inside.astype(np.float32)
triangle_axis.imshow(triangle_rgba, extent=(0, 1, 0, triangle_height), origin='lower', interpolation='bilinear')
triangle_axis.add_patch(Polygon([[0, 0], [1, 0], [0.5, triangle_height]], fill=False, edgecolor=OUTLINE_COLOR, linewidth=0.35))
triangle_axis.text(0.5, triangle_height + 0.07, 'PV', ha='center', va='bottom', fontsize=6, color=TEXT_COLOR)
triangle_axis.text(0, -0.07, 'NPV', ha='center', va='top', fontsize=6, color=TEXT_COLOR)
triangle_axis.text(1, -0.07, 'BS', ha='center', va='top', fontsize=6, color=TEXT_COLOR)
triangle_axis.set_xlim(-0.22, 1.22)
triangle_axis.set_ylim(-0.22, triangle_height + 0.18)
triangle_axis.set_aspect('equal')
triangle_axis.set_axis_off()
biome_area = np.bincount(biome_grid[study_mask], weights=area_weights[study_mask], minlength=8)[1:8]
land_area = np.bincount(land_values[land_valid], weights=area_weights[land_valid], minlength=7)[1:7]
biome_percentages = biome_area / biome_area.sum() * 100
land_percentages = land_area / land_area.sum() * 100
biome_order = [0, 1, 2, 3, 4, 5]
biome_short_labels = ['T3 Shrublands', 'T4 Savanna/grassland', 'T5 (semi-)deserts', 'T3 + T4', 'T3 + T5', 'T4 + T5']
category_legend_artists = []
category_legend_left_cm = right_left_cm + STAT_BAR_LEFT_CM + STAT_BAR_WIDTH_CM + 0.2
category_legend_width_cm = full_right_cm - 0.12 - category_legend_left_cm
for panel_index, (percentages, palette, names) in enumerate([(biome_percentages, BIOME_COLORS, biome_short_labels), (land_percentages, LAND_COLORS, LAND_LABELS)], start=1):
    panel_left, panel_bottom, panel_width, panel_height = panels_cm[panel_index]
    bar_axis = figure.add_axes([(panel_left + STAT_BAR_LEFT_CM) / TARGET_FIGURE_WIDTH_CM, (panel_bottom + 0.31) / TARGET_FIGURE_HEIGHT_CM, STAT_BAR_WIDTH_CM / TARGET_FIGURE_WIDTH_CM, STAT_BAR_HEIGHT_CM / TARGET_FIGURE_HEIGHT_CM])
    cumulative = 0.0
    for value, color in zip(percentages, palette):
        bar_axis.bar(0.5, value, bottom=cumulative, width=1.0, color=color, edgecolor='none')
        cumulative += value
    bar_axis.set_xlim(0, 1)
    bar_axis.set_ylim(0, 100)
    bar_axis.set_xticks([])
    bar_axis.set_yticks([])
    bar_axis.text(STAT_BAR_LABEL_X, 0.5, 'Area(%)', transform=bar_axis.transAxes, rotation=90, ha='center', va='center', fontsize=6, color=TEXT_COLOR)
    bar_axis.text(0.5, 1.04, '100', transform=bar_axis.transAxes, ha='center', va='bottom', fontsize=5.6, color=TEXT_COLOR)
    bar_axis.text(0.5, -0.035, '0', transform=bar_axis.transAxes, ha='center', va='top', fontsize=5.6, color=TEXT_COLOR)
    for spine in bar_axis.spines.values():
        spine.set_linewidth(0.25)
    legend_labels = []
    for label, percentage in zip(names, percentages[:6]):
        percentage_text = '<0.1' if 0 < percentage < 0.05 else f'{percentage:.1f}'
        legend_labels.append(f'{label} {percentage_text}%')
    order = [0, 3, 1, 4, 2, 5]
    legend = figure.legend([Patch(facecolor=palette[i], edgecolor='none') for i in order], [legend_labels[i] for i in order], ncol=3, loc='center left', mode='expand', bbox_to_anchor=(category_legend_left_cm / TARGET_FIGURE_WIDTH_CM, (panel_bottom + 0.64) / TARGET_FIGURE_HEIGHT_CM, category_legend_width_cm / TARGET_FIGURE_WIDTH_CM, 0), fontsize=CATEGORY_LEGEND_FONT_SIZE, frameon=False, handlelength=0.9, handletextpad=0.4, columnspacing=0.6, labelspacing=0.55, borderaxespad=0, borderpad=0)
    category_legend_artists.append(legend)
if SHOW_LOCATOR:
    if not LOCATOR_BASEMAP.is_file():
        raise FileNotFoundError(f'Missing locator basemap: {LOCATOR_BASEMAP}')
    locator_width_cm = right_left_cm - full_left_cm - LOCATOR_RIGHT_GAP_CM
    lower_center_y_cm = (biome_panel_top_cm + land_panel_bottom_cm) / 2
    locator_bottom_cm = lower_center_y_cm - locator_width_cm / 2
    locator_crs = ccrs.Orthographic(central_longitude=LOCATOR_CENTRAL_LONGITUDE, central_latitude=90)
    locator = figure.add_axes([full_left_cm / TARGET_FIGURE_WIDTH_CM, locator_bottom_cm / TARGET_FIGURE_HEIGHT_CM, locator_width_cm / TARGET_FIGURE_WIDTH_CM, locator_width_cm / TARGET_FIGURE_HEIGHT_CM], projection=locator_crs)
    locator.set_global()
    with Image.open(LOCATOR_BASEMAP) as basemap_source:
        basemap_rgb = np.asarray(basemap_source.convert('RGB'), dtype=np.float32) / 255.0
    luminance = basemap_rgb @ np.array([0.2126, 0.7152, 0.0722])
    basemap_soft = 0.2 * basemap_rgb + 0.15 * luminance[..., None] + 0.65
    locator.imshow(np.clip(basemap_soft, 0, 1), origin='upper', extent=(-180, 180, -90, 90), transform=data_crs, regrid_shape=400, zorder=0)
    locator_mask = np.ma.masked_where(~study_mask, np.ones(grid_shape, dtype=np.uint8))
    locator.imshow(locator_mask, origin='upper', extent=(grid_bounds.left, grid_bounds.right, grid_bounds.bottom, grid_bounds.top), transform=data_crs, cmap=ListedColormap(['#AA7562']), vmin=0, vmax=1, interpolation='nearest', regrid_shape=400, alpha=0.85, zorder=2)
    locator.gridlines(xlocs=LOCATOR_LONGITUDE_TICKS, ylocs=[0, 30, 60], color='#858B8C', linewidth=0.3, linestyle=':', alpha=0.65, zorder=3)
    for longitude in LOCATOR_LONGITUDE_TICKS:
        angle = np.deg2rad(longitude - LOCATOR_CENTRAL_LONGITUDE)
        direction_x, direction_y = (np.sin(angle), -np.cos(angle))
        locator.plot([0.5 + 0.5 * direction_x, 0.5 + 0.513 * direction_x], [0.5 + 0.5 * direction_y, 0.5 + 0.513 * direction_y], transform=locator.transAxes, clip_on=False, color=OUTLINE_COLOR, linewidth=0.35, zorder=5)
        longitude_text = '180°' if abs(longitude) == 180 else '0°' if longitude == 0 else f"{abs(longitude)}°{('W' if longitude < 0 else 'E')}"
        locator.text(0.5 + 0.53 * direction_x, 0.5 + 0.53 * direction_y, longitude_text, transform=locator.transAxes, clip_on=False, fontsize=5.1, ha='left' if direction_x > 0.2 else 'right' if direction_x < -0.2 else 'center', va='bottom' if direction_y > 0.2 else 'top' if direction_y < -0.2 else 'center', color=TEXT_COLOR, zorder=5)
    locator.spines['geo'].set_linewidth(0.45)
    locator.spines['geo'].set_edgecolor(OUTLINE_COLOR)
    locator.set_title('(b) Geographical location', fontsize=PANEL_TITLE_FONT_SIZE, fontfamily=FONT_FAMILY, fontweight='bold', loc='left', pad=7, color=TEXT_COLOR)
    locator.text(0.5, 0.5, '90°N', transform=locator.transAxes, fontsize=5.1, ha='center', va='center', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 1})
    locator.text(0.5, -0.05, 'Northern Hemisphere', transform=locator.transAxes, fontsize=6, ha='center', va='center', color=TEXT_COLOR)
    locator.legend([Patch(facecolor='#AA7562', edgecolor='none')], ['Study domain (0–66.6°N)'], loc='upper center', bbox_to_anchor=(0.5, -0.105), frameon=False, fontsize=5.6, handlelength=1.0, handletextpad=0.4)
figure.canvas.draw()
renderer = figure.canvas.get_renderer()
category_legend_bounds_cm = []
for legend, panel in zip(category_legend_artists, panels_cm[1:]):
    box = legend.get_window_extent(renderer).transformed(figure.transFigure.inverted())
    panel_left, panel_bottom, panel_width, panel_height = panel
    if box.x0 < (panel_left + STAT_BAR_LEFT_CM + STAT_BAR_WIDTH_CM + 0.2) / TARGET_FIGURE_WIDTH_CM - 1e-10 or box.x1 > (panel_left + panel_width) / TARGET_FIGURE_WIDTH_CM + 1e-10:
        raise ValueError('Category legend overlaps the proportion bar or exceeds the panel; adjust CATEGORY_LEGEND_FONT_SIZE')
    if box.y0 < panel_bottom / TARGET_FIGURE_HEIGHT_CM or box.y1 > (panel_bottom + category_annotation_height_cm) / TARGET_FIGURE_HEIGHT_CM:
        raise ValueError('Category legend exceeds its internal annotation strip')
    category_legend_bounds_cm.append([box.x0 * TARGET_FIGURE_WIDTH_CM, box.x1 * TARGET_FIGURE_WIDTH_CM])
if not np.allclose(category_legend_bounds_cm[0], category_legend_bounds_cm[1], atol=1e-08):
    raise ValueError('Legend endpoints in panels c and d are not aligned')
stem = 'Fig1'
save_main_figure(figure, 1, facecolor='white', pil_kwargs={'quality': 95, 'subsampling': 0})
plt.close(figure)
report = {'study_selection': 'Union of major-distribution T3/T4/T5 ecosystem regions, restricted to map latitude extent; no AI threshold applied', 'land_cover_year': LAND_COVER_YEAR, 'land_cover_source': str(land_path), 'component_period': [START_YEAR, END_YEAR], 'component_renormalized_for_display': RENORMALIZE_COMPONENTS, 'component_scale': COMPONENT_SCALE, 'minimum_valid_months': MIN_VALID_MONTHS, 'rgb_channel_order': list(RGB_COMPONENT_ORDER) if COMPONENT_COLOR_MODE == 'rgb' else None, 'map_extent': MAP_EXTENT, 'component_legend_order': list(COMPONENT_LEGEND_ORDER), 'component_color_mode': COMPONENT_COLOR_MODE, 'component_colors': COMPONENT_COLORS, 'component_background': COMPONENT_BACKGROUND, 'component_legend_shape': 'triangle', 'component_legend_position': 'lower left', 'component_map_extent': list(COMPONENT_MAP_EXTENT), 'component_actual_map_extent': list(map_axes[0].get_extent(crs=data_crs)), 'component_longitude_ticks': map_axes[0].get_xticks().tolist(), 'component_longitude_labels_visible': all((label.get_visible() for label in map_axes[0].get_xticklabels())), 'panel_title_style': {'font_family': FONT_FAMILY, 'font_size': PANEL_TITLE_FONT_SIZE, 'font_weight': 'bold'}, 'component_triangle_vertices': {'top': 'PV', 'left': 'NPV', 'right': 'BS'}, 'component_triangle_size_cm': TRIANGLE_LEGEND_SIZE_CM, 'panel_labels': {'a': 'components', 'b': 'locator', 'c': 'biomes', 'd': 'land_cover'}, 'layout': 'components top; biomes middle-right; land cover bottom-right; locator spans lower-left', 'classification_panel_width_fraction': right_width_cm / full_width_cm, 'component_panel_gap_cm': COMPONENT_PANEL_GAP_CM, 'locator_right_gap_cm': LOCATOR_RIGHT_GAP_CM, 'classification_map_downshift_degrees': CATEGORY_MAP_DOWN_DEGREES, 'classification_map_downshift_cm': category_map_downshift_cm, 'classification_actual_map_extents': [list(axis.get_extent(crs=data_crs)) for axis in map_axes[1:]], 'stat_bar_width_cm': STAT_BAR_WIDTH_CM, 'stat_bar_height_cm': STAT_BAR_HEIGHT_CM, 'stat_bar_title': 'Area(%)', 'stat_bar_title_rotation': 90, 'stat_bar_endpoint_labels': {'top': 100, 'bottom': 0}, 'locator_width_cm': locator_width_cm if SHOW_LOCATOR else None, 'biome_area_percentages': biome_percentages.tolist(), 'land_cover_area_percentages': land_percentages.tolist(), 'area_weighting': 'cos(latitude), 0.1-degree analysis grid; mutually exclusive combination classes', 'locator_basemap': str(LOCATOR_BASEMAP), 'locator_projection': 'North-polar Orthographic', 'locator_longitude_ticks': LOCATOR_LONGITUDE_TICKS, 'locator_longitude_label_position': 'outside circular boundary', 'annotation_band_inside_panel': True, 'figure_width_cm': TARGET_FIGURE_WIDTH_CM, 'figure_height_cm': TARGET_FIGURE_HEIGHT_CM, 'dpi': FIGURE_DPI, 'font_family': FONT_FAMILY, 'biome_legend_values': [index + 1 for index in biome_order], 'legend_columns': {'c': 3, 'd': 3}, 'show_north_arrow': False, 'bottom_fraction_note_shown': False, 'classification_legend_fits_inside_panel': True, 'classification_legend_endpoints_aligned': True, 'category_legend_bounds_cm': category_legend_bounds_cm, 'stat_bar_left_cm': STAT_BAR_LEFT_CM, 'stat_bar_title_offset_axes': STAT_BAR_LABEL_X, 'study_pixels_at_0p1_degree': int(study_mask.sum()), 'land_cover_missing_study_pixels': int(missing_land.sum()), 'component_valid_study_pixels': int(component_valid.sum()), 'component_missing_study_pixels': int((study_mask & ~component_valid).sum()), 'valid_month_count_range': [int(month_counts[component_valid].min()), int(month_counts[component_valid].max())], 'area_weighted_raw_component_means': component_area_means, 'area_weighted_display_component_means': display_area_means, 'sum_area_weighted_raw_means': sum(component_area_means.values()), 'source_signature': signature, 'figures_stem': stem}
(REPORT_DIR / f'{stem}_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
print(f"Figure saved: {FIGURE_DIR / (stem + '.jpg')}", flush=True)
