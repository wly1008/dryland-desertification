"""Calculate and plot paired with/without-NPV DSI state differences (Figure 8)."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, study_boundary, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, save_main_figure, source_record
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd
import numpy as np
from pyproj import Geod
import rasterio
from rasterio.features import geometry_mask
from tqdm import tqdm
INPUT_ROOT = PROJECT_ROOT / '03_analysis/01_dsi/data/02DI'
INPUT_DIRS = {'without_npv': PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV/02study_period_weights', 'with_npv': PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'}
STUDY_AREA_PATH = study_boundary()
OUTPUT_ROOT = PROJECT_ROOT / '03_analysis/04_npv_ablation'
DATA_DIR = PROJECT_ROOT / '03_analysis/04_npv_ablation/data/figure8'
FIGURE_DIR = FIGURES_ROOT / 'jpg'
START_YEAR = 2004
END_YEAR = 2019
YEARS = list(range(START_YEAR, END_YEAR + 1))
PERIOD_LABEL = f'{START_YEAR}_{END_YEAR}'
STATE_LIMITS = (0.35, 0.85)
DIFFERENCE_LIMIT = 0.08
HISTOGRAM_BIN_WIDTH = 0.002
FIGURE_WIDTH_CM = 17.0
FIGURE_HEIGHT_CM = 14.7
FINAL_DPI = 400
JPG_DPI = 400
OUTPUT_NODATA = -9999.0
NEGATIVE_COLOR = '#567FA6'
POSITIVE_COLOR = '#B45F55'
TEXT_COLOR = '#30363D'
for directory in (PROJECT_ROOT / '03_analysis/04_npv_ablation/data/figure8', FIGURES_ROOT / 'jpg'):
    directory.mkdir(parents=True, exist_ok=True)
for group_name, input_directory in INPUT_DIRS.items():
    for year in YEARS:
        input_path = input_directory / f'DI_{year}.tif'
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
if not STUDY_AREA_PATH.is_file():
    raise FileNotFoundError(STUDY_AREA_PATH)
with rasterio.open(PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV/02study_period_weights' / f'DI_{START_YEAR}.tif') as source:
    reference_profile = source.profile.copy()
    reference_transform = source.transform
    reference_crs = source.crs
    reference_shape = source.shape
    reference_bounds = source.bounds
if reference_crs is None or reference_crs.to_epsg() != 4326:
    raise ValueError('Map and area calculations require EPSG:4326 inputs.')
if not (abs(reference_transform.b) < 1e-12 and abs(reference_transform.d) < 1e-12 and (reference_transform.a > 0) and (reference_transform.e < 0)):
    raise ValueError('Inputs must use an unrotated north-up longitude-latitude grid.')
study_area = gpd.read_file(STUDY_AREA_PATH)
if study_area.crs is None:
    raise ValueError('Study-area vector has no CRS.')
study_area = study_area.to_crs(reference_crs)
if study_area.empty or not study_area.geometry.is_valid.all():
    raise ValueError('Study-area vector is empty or contains invalid geometry.')
study_geometries = list(study_area.geometry)
study_mask = geometry_mask(study_geometries, out_shape=reference_shape, transform=reference_transform, invert=True, all_touched=False)
sums = {name: np.zeros(reference_shape, dtype=np.float64) for name in INPUT_DIRS}
valid_counts = {name: np.zeros(reference_shape, dtype=np.uint8) for name in INPUT_DIRS}
common_valid = np.ones(reference_shape, dtype=bool)
input_manifest = []
group_weights = {}
input_jobs = [(name, year) for name in INPUT_DIRS for year in YEARS]
for group_name, year in tqdm(input_jobs, desc='Read and validate annual DSI', unit='raster'):
    input_path = INPUT_DIRS[group_name] / f'DI_{year}.tif'
    with rasterio.open(input_path) as source:
        if source.count != 1 or source.shape != reference_shape:
            raise ValueError(f'Band count or grid shape mismatch: {input_path}')
        if source.crs != reference_crs or not source.transform.almost_equals(reference_transform):
            raise ValueError(f'CRS or grid-position mismatch: {input_path}')
        tags = source.tags()
        if tags.get('DIRECTION_STATUS') != 'all indicators already unified; higher means more desertification':
            raise ValueError(f'Indicator-direction metadata mismatch: {input_path}')
        if tags.get('DI_FORMULA') != 'DI=sum(W_k*clip(x_k/10000,0,1))':
            raise ValueError(f'DSI formula or normalization-scale metadata mismatch: {input_path}')
        if tags.get('YEAR') != str(year) or tags.get('WEIGHT_MODE') != 'study_period_weights':
            raise ValueError(f'Year or weight-mode mismatch: {input_path}')
        weights = dict((item.split('=', 1) for item in tags['APPLIED_WEIGHTS'].split(';')))
        weights = {name: float(value) for name, value in weights.items()}
        if group_name in group_weights and weights != group_weights[group_name]:
            raise ValueError(f'Weights changed within the study period: {input_path}')
        group_weights[group_name] = weights
        annual = source.read(1, masked=True)
        valid = ~np.ma.getmaskarray(annual) & np.isfinite(annual.data)
        if np.any((annual.data[valid] < 0) | (annual.data[valid] > 1)):
            raise ValueError(f'Valid DSI outside [0, 1]: {input_path}')
        sums[group_name] += np.where(valid, annual.data, 0.0)
        valid_counts[group_name] += valid.astype(np.uint8)
        common_valid &= valid
        input_manifest.append({'group': group_name, 'year': year, **source_record(input_path), 'valid_pixels': int(np.count_nonzero(valid)), 'period_start': tags.get('PERIOD_START'), 'period_end': tags.get('PERIOD_END')})
complete_before_study_mask = int(np.count_nonzero(common_valid))
common_valid &= study_mask
common_count = int(np.count_nonzero(common_valid))
if common_count == 0:
    raise ValueError('No study-area pixels are valid across all years and both DSI variants.')
for group_name in INPUT_DIRS:
    if not np.all(valid_counts[group_name][common_valid] == len(YEARS)):
        raise AssertionError('Common-valid mask includes pixels missing one or more years.')
means = {}
for group_name in INPUT_DIRS:
    means[group_name] = sums[group_name] / len(YEARS)
    means[group_name][~common_valid] = np.nan
difference = means['with_npv'] - means['without_npv']
geod = Geod(ellps='WGS84')
row_areas_km2 = np.empty(reference_shape[0], dtype=np.float64)
west = reference_transform.c
east = west + reference_transform.a
for row in range(reference_shape[0]):
    north = reference_transform.f + row * reference_transform.e
    south = north + reference_transform.e
    signed_area, perimeter = geod.polygon_area_perimeter([west, east, east, west], [north, north, south, south])
    row_areas_km2[row] = abs(signed_area) / 1000000.0
area_grid = np.broadcast_to(row_areas_km2[:, None], reference_shape)
pixel_areas = area_grid[common_valid]
delta_values = difference[common_valid]
total_area_km2 = float(pixel_areas.sum())
delta_mean = float(np.average(delta_values, weights=pixel_areas))
delta_mean_absolute = float(np.average(np.abs(delta_values), weights=pixel_areas))
negative_area_pct = float(pixel_areas[delta_values < 0].sum() / total_area_km2 * 100)
positive_area_pct = float(pixel_areas[delta_values > 0].sum() / total_area_km2 * 100)
zero_area_pct = float(pixel_areas[delta_values == 0].sum() / total_area_km2 * 100)
sort_order = np.argsort(delta_values)
sorted_delta = delta_values[sort_order]
cumulative_area = np.cumsum(pixel_areas[sort_order])
delta_quantiles = np.interp(np.asarray([0.025, 0.25, 0.5, 0.75, 0.975]) * total_area_km2, cumulative_area, sorted_delta)
histogram_lower = np.floor(delta_values.min() / HISTOGRAM_BIN_WIDTH) * HISTOGRAM_BIN_WIDTH
histogram_upper = np.ceil(delta_values.max() / HISTOGRAM_BIN_WIDTH) * HISTOGRAM_BIN_WIDTH
histogram_edges = np.arange(round(histogram_lower / HISTOGRAM_BIN_WIDTH), round(histogram_upper / HISTOGRAM_BIN_WIDTH) + 1) * HISTOGRAM_BIN_WIDTH
histogram_area, histogram_edges = np.histogram(delta_values, bins=histogram_edges, weights=pixel_areas)
histogram_pct = histogram_area / total_area_km2 * 100
histogram_centers = (histogram_edges[:-1] + histogram_edges[1:]) / 2
if not np.isclose(histogram_pct.sum(), 100, atol=1e-08):
    raise AssertionError('Histogram does not cover the complete valid area.')
if not np.isclose(negative_area_pct + positive_area_pct + zero_area_pct, 100):
    raise AssertionError('Negative, positive, and zero difference-area shares do not sum to 100%.')
if max(abs(delta_values.min()), abs(delta_values.max())) > DIFFERENCE_LIMIT:
    raise ValueError('Differences exceed the color limits; increase DIFFERENCE_LIMIT to avoid unreported clipping.')
for group_name in INPUT_DIRS:
    values = means[group_name][common_valid]
    if values.min() < STATE_LIMITS[0] or values.max() > STATE_LIMITS[1]:
        raise ValueError('Multiyear means exceed display limits; adjust STATE_LIMITS.')
output_profile = reference_profile.copy()
output_profile.update(dtype='float32', count=1, nodata=OUTPUT_NODATA, compress='deflate')
output_arrays = {'DSI_mean_without_NPV': means['without_npv'], 'DSI_mean_with_NPV': means['with_npv'], 'DSI_mean_difference_with_minus_without_NPV': difference}
for raster_name, values in output_arrays.items():
    output_path = DATA_DIR / f'{raster_name}_{PERIOD_LABEL}.tif'
    with rasterio.open(output_path, 'w', **output_profile) as destination:
        destination.write(np.where(common_valid, values, OUTPUT_NODATA).astype('float32'), 1)
        destination.update_tags(START_YEAR=str(START_YEAR), END_YEAR=str(END_YEAR), VALID_YEARS=str(len(YEARS)), MASK_RULE='pixel center within study polygon; valid in both variants in all years', DIRECTION='higher DSI means more desertification', DIFFERENCE_DEFINITION='mean(with NPV) minus mean(without NPV)')
summary = {'start_year': START_YEAR, 'end_year': END_YEAR, 'number_of_years': len(YEARS), 'year_interpretation': 'annual filename labels; non-growing seasons cross calendar years', 'first_period_start': input_manifest[0]['period_start'], 'last_period_end': input_manifest[len(YEARS) - 1]['period_end'], 'study_area': str(STUDY_AREA_PATH), 'complete_pixels_before_polygon_mask': complete_before_study_mask, 'common_valid_pixels': common_count, 'pixels_removed_by_polygon_mask': complete_before_study_mask - common_count, 'common_area_km2': total_area_km2, 'area_weighting': 'WGS84 ellipsoidal polygon area of each longitude-latitude grid cell', 'mean_without_npv': float(np.average(means['without_npv'][common_valid], weights=pixel_areas)), 'mean_with_npv': float(np.average(means['with_npv'][common_valid], weights=pixel_areas)), 'delta_mean': delta_mean, 'delta_mean_absolute': delta_mean_absolute, 'delta_min': float(delta_values.min()), 'delta_max': float(delta_values.max()), 'delta_area_weighted_quantiles': dict(zip(['q025', 'q25', 'q50', 'q75', 'q975'], delta_quantiles.tolist())), 'negative_area_percent': negative_area_pct, 'positive_area_percent': positive_area_pct, 'exact_zero_area_percent': zero_area_pct, 'histogram_bin_width': HISTOGRAM_BIN_WIDTH, 'histogram_area_percent_sum': float(histogram_pct.sum()), 'state_color_limits': list(STATE_LIMITS), 'difference_color_limits': [-DIFFERENCE_LIMIT, DIFFERENCE_LIMIT], 'group_weights': group_weights, 'inputs': input_manifest}
(DATA_DIR / f'figure8_statistics_{PERIOD_LABEL}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
with (DATA_DIR / f'figure8_difference_distribution_{PERIOD_LABEL}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.writer(stream)
    writer.writerow(['bin_left', 'bin_right', 'area_km2', 'area_percent'])
    writer.writerows(zip(histogram_edges[:-1], histogram_edges[1:], histogram_area, histogram_pct))
plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'font.size': 7.0, 'axes.titlesize': 8.0, 'axes.labelsize': 7.0, 'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0, 'legend.fontsize': 6.0, 'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR, 'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR, 'axes.edgecolor': '#90979E', 'axes.linewidth': 0.5, 'savefig.facecolor': 'white', 'svg.fonttype': 'none', 'axes.unicode_minus': True})
state_cmap = LinearSegmentedColormap.from_list('DSI_state', ['#F3EFE3', '#D7DEC4', '#B7CCB4', '#8FB4A6', '#5C6678'], N=256)
delta_cmap = LinearSegmentedColormap.from_list('NPV_difference', ['#35628D', '#9DB8CD', '#F9F7F2', '#D9A59A', '#A74740'], N=256)
state_cmap.set_bad('#FFFFFF')
delta_cmap.set_bad('#FFFFFF')
state_norm = Normalize(*STATE_LIMITS)
delta_norm = TwoSlopeNorm(vmin=-DIFFERENCE_LIMIT, vcenter=0, vmax=DIFFERENCE_LIMIT)
data_projection = ccrs.PlateCarree()
circular_projection = ccrs.LambertAzimuthalEqualArea(central_longitude=0.0, central_latitude=90.0)
figure = plt.figure(figsize=(FIGURE_WIDTH_CM / 2.54, FIGURE_HEIGHT_CM / 2.54), facecolor='white')
image_extent = [reference_bounds.left, reference_bounds.right, reference_bounds.bottom, reference_bounds.top]
rectangular_positions = [[0.045, 0.788, 0.89, 0.203], [0.045, 0.583, 0.89, 0.203]]
rectangular_arrays = [means['without_npv'], means['with_npv']]
rectangular_titles = ['(a) Mean DSI without NPV', '(b) Mean DSI with NPV']
rectangular_images = []
rectangular_axes = []
for panel_index, (position, values, title) in enumerate(zip(rectangular_positions, rectangular_arrays, rectangular_titles)):
    axis = figure.add_axes(position, projection=data_projection)
    axis.set_facecolor('#FFFFFF')
    map_image = axis.imshow(np.ma.masked_invalid(values), origin='upper', extent=image_extent, transform=data_projection, interpolation='nearest', rasterized=True, cmap=state_cmap, norm=state_norm)
    axis.add_geometries(study_geometries, crs=data_projection, facecolor='none', edgecolor='#646B70', linewidth=0.23, alpha=0.75, zorder=3)
    axis.set_extent([-180, 180, -1, 68], crs=data_projection)
    axis.set_xticks([-150, -100, -50, 0, 50, 100, 150], crs=data_projection)
    axis.set_yticks([0, 20, 40, 60], crs=data_projection)
    axis.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    axis.yaxis.set_major_formatter(LatitudeFormatter())
    axis.tick_params(length=2, width=0.45, pad=2, labelbottom=panel_index == len(rectangular_positions) - 1)
    axis.gridlines(xlocs=[-150, -100, -50, 0, 50, 100, 150], ylocs=[0, 20, 40, 60], linewidth=0.3, color='#BFC6CB', alpha=0.55, zorder=0)
    axis.text(0.012, 0.955, title, transform=axis.transAxes, ha='left', va='top', fontsize=8, fontweight='bold', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1.0}, zorder=5)
    if panel_index == 0:
        axis.text(0.988, 0.955, f'{START_YEAR}–{END_YEAR}', transform=axis.transAxes, ha='right', va='top', fontsize=6, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1.0}, zorder=5)
    rectangular_images.append(map_image)
    rectangular_axes.append(axis)
rectangular_group_top = max((axis.get_position().y1 for axis in rectangular_axes))
rectangular_group_bottom = min((axis.get_position().y0 for axis in rectangular_axes))
rectangular_title_x = rectangular_axes[0].get_position().x0 + 0.012 * rectangular_axes[0].get_position().width
state_colorbar_top = rectangular_group_top - 0.052
state_colorbar_bottom = rectangular_group_bottom + 0.085
state_colorbar_x = 0.95
state_colorbar_width = 0.014
state_colorbar_center_x = state_colorbar_x + state_colorbar_width / 2.0
state_colorbar_axis = figure.add_axes([state_colorbar_x, state_colorbar_bottom, state_colorbar_width, state_colorbar_top - state_colorbar_bottom])
state_colorbar = figure.colorbar(rectangular_images[0], cax=state_colorbar_axis, orientation='vertical')
state_colorbar.outline.set_linewidth(0.4)
state_colorbar.ax.set_yticks([])
figure.text(state_colorbar_center_x, rectangular_group_top - 0.018, f'{STATE_LIMITS[1]:.2f}', ha='center', va='center', rotation=90, fontsize=6.0)
figure.text(state_colorbar_center_x, rectangular_group_bottom + 0.06, f'{STATE_LIMITS[0]:.2f}', ha='center', va='center', rotation=90, fontsize=6.0)
figure.text(state_colorbar_center_x, rectangular_group_bottom + 0.018, 'DSI', ha='center', va='center', rotation=90, fontsize=6.0, fontweight='bold')
boundary_angles = np.linspace(0.0, 2.0 * np.pi, 361)
boundary_vertices = np.column_stack([np.sin(boundary_angles), np.cos(boundary_angles)])
circular_boundary = mpath.Path(boundary_vertices * 0.5 + np.array([0.5, 0.5]))
circular_axis = figure.add_axes([0.075, 0.048, 0.41, 0.474], projection=circular_projection)
circular_axis.set_facecolor('#FFFFFF')
circular_axis.set_extent([-180, 180, 0, 90], crs=data_projection)
circular_axis.set_boundary(circular_boundary, transform=circular_axis.transAxes)
circular_image = circular_axis.imshow(np.ma.masked_invalid(difference), origin='upper', extent=image_extent, transform=data_projection, interpolation='nearest', regrid_shape=1600, rasterized=True, cmap=delta_cmap, norm=delta_norm, zorder=1)
circular_axis.add_geometries(study_geometries, crs=data_projection, facecolor='none', edgecolor='#646B70', linewidth=0.23, alpha=0.78, zorder=3)
circular_axis.gridlines(crs=data_projection, draw_labels=False, xlocs=np.arange(-180, 181, 30), ylocs=[20, 40, 60, 80], linewidth=0.3, color='#BFC6CB', linestyle='--', alpha=0.5, zorder=2)
circular_axis.spines['geo'].set_edgecolor('#646B70')
circular_axis.spines['geo'].set_linewidth(0.55)
circular_position = circular_axis.get_position()
bottom_title_y = circular_position.y1 + 0.008
figure.text(rectangular_title_x, bottom_title_y, '(c) Mean difference, $\\Delta$DSI', ha='left', va='bottom', fontsize=8, fontweight='bold')
delta_colorbar_top = circular_position.y1 - 0.052
delta_colorbar_bottom = circular_position.y0 + 0.085
delta_colorbar_axis = figure.add_axes([0.032, delta_colorbar_bottom, 0.016, delta_colorbar_top - delta_colorbar_bottom])
delta_colorbar = figure.colorbar(circular_image, cax=delta_colorbar_axis, orientation='vertical')
delta_colorbar.outline.set_linewidth(0.4)
delta_colorbar.ax.set_yticks([])
figure.text(0.04, circular_position.y1 - 0.018, f'{DIFFERENCE_LIMIT:+.2f}', ha='center', va='center', rotation=90, fontsize=6.0)
figure.text(0.04, circular_position.y0 + 0.06, f'{-DIFFERENCE_LIMIT:+.2f}', ha='center', va='center', rotation=90, fontsize=6.0)
figure.text(0.04, circular_position.y0 + 0.008, '$\\Delta$DSI', ha='center', va='center', rotation=90, fontsize=6.0, fontweight='bold')
histogram_axis = figure.add_axes([0.535, 0.063, 0.43, 0.445])
figure.text(histogram_axis.get_position().x0, bottom_title_y, '(d) Area-weighted difference distribution', ha='left', va='bottom', fontsize=8, fontweight='bold')
histogram_axis.bar(histogram_centers, histogram_pct, width=HISTOGRAM_BIN_WIDTH * 0.94, color=np.where(histogram_centers < 0, NEGATIVE_COLOR, POSITIVE_COLOR), edgecolor='none', alpha=0.9, zorder=2)
histogram_axis.axvline(0, color='#596168', linewidth=0.75, linestyle=(0, (3, 2)), zorder=3)
histogram_axis.axvline(delta_mean, color='#343C45', linewidth=0.9, linestyle=(0, (1, 1.5)), zorder=3)
histogram_axis.set_xlim(-0.04, 0.08)
histogram_axis.set_ylim(0, max(histogram_pct) * 1.16)
histogram_axis.set_xlabel('$\\Delta$DSI', labelpad=3)
histogram_axis.set_ylabel('Area (%)', labelpad=2)
histogram_axis.xaxis.set_major_locator(MultipleLocator(0.02))
histogram_axis.yaxis.set_major_locator(MultipleLocator(2))
histogram_axis.tick_params(length=2, width=0.45, pad=2)
histogram_axis.spines[['top', 'right']].set_visible(False)
histogram_axis.set_axisbelow(True)
histogram_axis.grid(axis='y', color='#DEE2E5', linewidth=0.4)
histogram_axis.text(0.02, 0.94, f'Lower: {negative_area_pct:.2f}%', transform=histogram_axis.transAxes, ha='left', va='top', fontsize=6, color=NEGATIVE_COLOR)
histogram_axis.text(0.98, 0.94, f'Higher: {positive_area_pct:.2f}%', transform=histogram_axis.transAxes, ha='right', va='top', fontsize=6, color=POSITIVE_COLOR)
histogram_axis.text(0.02, 0.75, f'Mean: {delta_mean:+.4f}', transform=histogram_axis.transAxes, ha='left', va='top', fontsize=6)
figure_stem = f'Fig8_NPV_state_effect_{PERIOD_LABEL}'
save_main_figure(figure, 8, bbox_inches='tight', pad_inches=0, facecolor='white', pil_kwargs={'quality': 95, 'subsampling': 0})
plt.close(figure)
print(json.dumps({key: summary[key] for key in ['common_valid_pixels', 'pixels_removed_by_polygon_mask', 'common_area_km2', 'mean_without_npv', 'mean_with_npv', 'delta_mean', 'negative_area_percent', 'positive_area_percent']}, ensure_ascii=False, indent=2))
print(f'Figure 8 saved to: {FIGURE_DIR}')
