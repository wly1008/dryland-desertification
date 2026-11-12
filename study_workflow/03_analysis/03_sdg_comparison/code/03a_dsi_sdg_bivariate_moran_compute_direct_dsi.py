"""Calculate local bivariate Moran statistics for direct DSI slopes and SDG products."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import platform
import time
from datetime import datetime
from pathlib import Path
import esda
import libpysal
import numba
import numpy as np
import openpyxl
import pyproj
import rasterio
import scipy
from esda.moran import Moran_BV, Moran_Local_BV
from libpysal.graph import Graph
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from pyproj import Geod
from scipy import sparse, stats
SCRIPT_DIR = CODE_ROOT / '03_analysis/03_sdg_comparison/code'
WORKING_ROOT = PROJECT_ROOT / '03_analysis/03_sdg_comparison'
VERIFY_DIR = PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1'
DSI_DIR = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'
DSI_PATHS = {'2004_2019': PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_2004_2019.tif'}
SDG_PATHS = {'FAO-WOCAT': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_FAO-WOCAT/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif', 'JRC': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_JRC/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif', 'Trends.Earth': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_Trends.Earth/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif'}
OUTPUT_ROOT = PROJECT_ROOT / '03_analysis/03_sdg_comparison/data/03_bivariate_moran_lisa_direct_dsi'
SDG_MODES = ['continuous']
THRESHOLD_FRACTION = 0.1
PERMUTATIONS = 999
BASE_RANDOM_SEED = 20260818
FDR_ALPHA = 0.05
import os
LOCAL_N_JOBS = int(os.environ.get('STUDY_LISA_JOBS', '1'))
if LOCAL_N_JOBS < 1:
    raise ValueError('STUDY_LISA_JOBS must be a positive integer')
ROW_STANDARDIZATION_TOLERANCE = 1e-06
PRODUCTS_TO_RUN = set(filter(None, os.environ.get('STUDY_LISA_PRODUCTS', '').split(','))) or None
if PRODUCTS_TO_RUN and not PRODUCTS_TO_RUN.issubset(SDG_PATHS):
    raise ValueError('STUDY_LISA_PRODUCTS contains an unknown product')
PRODUCT_SEED_OFFSETS = {'FAO-WOCAT': 0, 'JRC': 1000, 'Trends.Earth': 2000}
LOCAL_PERMUTATION_SEEDS = {'FAO-WOCAT': 20260821, 'JRC': 20261821, 'Trends.Earth': 20260821}
LISA_LABELS = {0: 'Not significant', 1: 'HH: DSI degradation / neighboring SDG improvement', 2: 'LL: DSI improvement / neighboring SDG degradation', 3: 'HL: DSI degradation / neighboring SDG degradation', 4: 'LH: DSI improvement / neighboring SDG improvement'}

def build_compact_queen_graph(valid_mask):
    height, width = valid_mask.shape
    compact_grid = np.full((height, width), -1, dtype=np.int32)
    original_flat_indices = np.flatnonzero(valid_mask.ravel())
    compact_grid.ravel()[original_flat_indices] = np.arange(original_flat_indices.size, dtype=np.int32)
    row_parts = []
    column_parts = []
    for row_shift, column_shift in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        source_rows = slice(max(0, -row_shift), min(height, height - row_shift))
        source_columns = slice(max(0, -column_shift), min(width, width - column_shift))
        target_rows = slice(max(0, row_shift), min(height, height + row_shift))
        target_columns = slice(max(0, column_shift), min(width, width + column_shift))
        paired = valid_mask[source_rows, source_columns] & valid_mask[target_rows, target_columns]
        if np.any(paired):
            row_parts.append(compact_grid[source_rows, source_columns][paired])
            column_parts.append(compact_grid[target_rows, target_columns][paired])
    if not row_parts:
        raise ValueError('No Queen-neighbor pairs were found')
    adjacency_rows = np.concatenate(row_parts).astype(np.int64, copy=False)
    adjacency_columns = np.concatenate(column_parts).astype(np.int64, copy=False)
    adjacency = sparse.csr_array((np.ones(adjacency_rows.size, dtype=np.float64), (adjacency_rows, adjacency_columns)), shape=(original_flat_indices.size, original_flat_indices.size))
    adjacency.sum_duplicates()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    retained = degrees > 0.0
    isolated_count = int(np.count_nonzero(~retained))
    if isolated_count:
        original_flat_indices = original_flat_indices[retained]
        adjacency = adjacency[retained][:, retained].tocsr()
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    if original_flat_indices.size == 0 or np.any(degrees <= 0.0):
        raise ValueError('Queen graph contains no usable non-island observations')
    graph = Graph.from_sparse(adjacency).transform('R')
    row_sums = np.asarray(graph.sparse.sum(axis=1)).ravel()
    maximum_row_sum_error = float(np.max(np.abs(row_sums - 1.0)))
    if maximum_row_sum_error > ROW_STANDARDIZATION_TOLERANCE:
        raise ValueError(f'Row-standardized Queen weights do not sum to one within tolerance: max error={maximum_row_sum_error:.3e}')
    if np.any(graph.sparse.diagonal() != 0.0):
        raise ValueError('Queen graph contains self-neighbor links')
    return (graph, original_flat_indices, isolated_count)

def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or p_values.size == 0:
        raise ValueError('FDR input must be a non-empty one-dimensional array')
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError('FDR input p-values must be finite and inside [0, 1]')
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * p_values.size / np.arange(1, p_values.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted

def write_raster(output_path, array, reference_profile, dtype, nodata):
    profile = reference_profile.copy()
    profile.update(driver='GTiff', count=1, dtype=dtype, nodata=nodata, compress='DEFLATE', tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(dtype, copy=False), 1)

def style_plain_workbook(workbook):
    for sheet in workbook.worksheets:
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        sheet.freeze_panes = 'A2'
        for column_cells in sheet.columns:
            column_letter = get_column_letter(column_cells[0].column)
            max_length = max((len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells))
            sheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 65)
synthetic_mask = np.ones((3, 3), dtype=bool)
synthetic_graph, synthetic_flat, synthetic_islands = build_compact_queen_graph(synthetic_mask)
synthetic_cardinalities = np.diff(synthetic_graph.sparse.indptr)
if synthetic_islands != 0:
    raise ValueError('Synthetic full 3x3 grid unexpectedly contains an island')
if synthetic_cardinalities[4] != 8:
    raise ValueError('Synthetic Queen center cell must have eight neighbors')
if not np.all(synthetic_cardinalities[[0, 2, 6, 8]] == 3):
    raise ValueError('Synthetic Queen corner cells must have three neighbors')
synthetic_x = np.arange(9, dtype=np.float64)
synthetic_y = synthetic_x.copy()
np.random.seed(BASE_RANDOM_SEED)
synthetic_global = Moran_BV(synthetic_x, synthetic_y, synthetic_graph, transformation='R', permutations=19)
synthetic_local_1 = Moran_Local_BV(synthetic_x, synthetic_y, synthetic_graph, transformation='R', permutations=19, n_jobs=1, keep_simulations=False, seed=BASE_RANDOM_SEED, island_weight=0, alternative='directed')
synthetic_local_2 = Moran_Local_BV(synthetic_x, synthetic_y, synthetic_graph, transformation='R', permutations=19, n_jobs=1, keep_simulations=False, seed=BASE_RANDOM_SEED, island_weight=0, alternative='directed')
if not np.isfinite(synthetic_global.I):
    raise ValueError("Synthetic global bivariate Moran's I is not finite")
if not np.array_equal(synthetic_local_1.p_sim, synthetic_local_2.p_sim):
    raise ValueError('Synthetic local permutation results are not reproducible')
synthetic_quadrant_x = np.array([1.0, -1.0, 1.0, -1.0])
synthetic_quadrant_lag = np.array([1.0, -1.0, -1.0, 1.0])
synthetic_quadrants = np.zeros(4, dtype=np.uint8)
synthetic_quadrants[(synthetic_quadrant_x > 0) & (synthetic_quadrant_lag > 0)] = 1
synthetic_quadrants[(synthetic_quadrant_x < 0) & (synthetic_quadrant_lag < 0)] = 2
synthetic_quadrants[(synthetic_quadrant_x > 0) & (synthetic_quadrant_lag < 0)] = 3
synthetic_quadrants[(synthetic_quadrant_x < 0) & (synthetic_quadrant_lag > 0)] = 4
if not np.array_equal(synthetic_quadrants, np.array([1, 2, 3, 4], dtype=np.uint8)):
    raise ValueError('Synthetic HH/LL/HL/LH coding failed')
synthetic_q = benjamini_hochberg(np.array([0.001, 0.02, 0.04, 0.8]))
if np.any((synthetic_q < 0.0) | (synthetic_q > 1.0)):
    raise ValueError('Synthetic FDR adjustment failed')
for required_path in list(DSI_PATHS.values()) + list(SDG_PATHS.values()):
    if not required_path.exists():
        raise FileNotFoundError(f'Missing required raster: {required_path}')
dsi_rasters = {}
reference_shape = None
reference_transform = None
reference_crs = None
reference_profile = None
reference_bounds = None
for version_name, dsi_path in DSI_PATHS.items():
    with rasterio.open(dsi_path) as src:
        masked = src.read(1, masked=True)
        data = np.asarray(masked.data, dtype=np.float64)
        valid = ~np.ma.getmaskarray(masked) & np.isfinite(data)
        if reference_shape is None:
            reference_shape = src.shape
            reference_transform = src.transform
            reference_crs = src.crs
            reference_profile = src.profile.copy()
            reference_bounds = src.bounds
        else:
            if src.shape != reference_shape:
                raise ValueError(f'DSI shape mismatch for {version_name}')
            if not src.transform.almost_equals(reference_transform):
                raise ValueError(f'DSI transform mismatch for {version_name}')
            if src.crs != reference_crs:
                raise ValueError(f'DSI CRS mismatch for {version_name}')
    dsi_rasters[version_name] = {'path': dsi_path, 'data': data, 'valid': valid}
if reference_crs is None or not reference_crs.is_geographic:
    raise ValueError('Expected a geographic WGS84-like reference grid')
sdg_rasters = {}
for product_name, sdg_path in SDG_PATHS.items():
    if PRODUCTS_TO_RUN is not None and product_name not in PRODUCTS_TO_RUN:
        continue
    with rasterio.open(sdg_path) as src:
        if src.shape != reference_shape:
            raise ValueError(f'SDG shape mismatch for {product_name}')
        if not src.transform.almost_equals(reference_transform):
            raise ValueError(f'SDG transform mismatch for {product_name}')
        if src.crs != reference_crs:
            raise ValueError(f'SDG CRS mismatch for {product_name}')
        masked = src.read(1, masked=True)
        data = np.asarray(masked.data, dtype=np.float64)
        valid = ~np.ma.getmaskarray(masked) & np.isfinite(data)
    valid_values = data[valid]
    sdg_min = float(np.min(valid_values))
    sdg_max = float(np.max(valid_values))
    threshold_low = THRESHOLD_FRACTION * sdg_min
    threshold_high = THRESHOLD_FRACTION * sdg_max
    if not sdg_min < 0.0 < sdg_max:
        raise ValueError(f'SDG values must span zero for {product_name}')
    if not np.isclose(threshold_low / sdg_max, -0.1, rtol=1e-06, atol=1e-08):
        raise ValueError(f'Lower normalized threshold is not -0.1 for {product_name}')
    if not np.isclose(threshold_high / sdg_max, 0.1, rtol=1e-06, atol=1e-08):
        raise ValueError(f'Upper normalized threshold is not 0.1 for {product_name}')
    classified = np.zeros(reference_shape, dtype=np.int8)
    classified[valid & (data < threshold_low)] = -1
    classified[valid & (data > threshold_high)] = 1
    sdg_rasters[product_name] = {'path': sdg_path, 'data': data, 'valid': valid, 'classified': classified, 'min': sdg_min, 'max': sdg_max, 'threshold_low': threshold_low, 'threshold_high': threshold_high, 'valid_count': int(np.count_nonzero(valid))}
geod = Geod(ellps='WGS84')
pixel_area_km2_by_row = np.zeros(reference_shape[0], dtype=np.float64)
left_longitude = reference_transform.c
right_longitude = left_longitude + reference_transform.a
for row_index in range(reference_shape[0]):
    top_latitude = reference_transform.f + row_index * reference_transform.e
    bottom_latitude = top_latitude + reference_transform.e
    signed_area, _ = geod.polygon_area_perimeter([left_longitude, right_longitude, right_longitude, left_longitude], [top_latitude, top_latitude, bottom_latitude, bottom_latitude])
    pixel_area_km2_by_row[row_index] = abs(signed_area) / 1000000.0
if np.any(~np.isfinite(pixel_area_km2_by_row)) or np.any(pixel_area_km2_by_row <= 0.0):
    raise ValueError('Invalid WGS84 pixel-area calculation')
for version_index, (version_name, dsi_item) in enumerate(dsi_rasters.items()):
    version_start = time.perf_counter()
    output_dir = OUTPUT_ROOT / version_name
    raster_dir = output_dir / 'rasters'
    output_dir.mkdir(parents=True, exist_ok=True)
    raster_dir.mkdir(parents=True, exist_ok=True)
    correlation_rows = []
    global_moran_rows = []
    lisa_area_rows = []
    graph_rows = []
    for product_index, (product_name, sdg_item) in enumerate(sdg_rasters.items()):
        product_start = time.perf_counter()
        paired_valid = dsi_item['valid'] & sdg_item['valid']
        paired_count_before_islands = int(np.count_nonzero(paired_valid))
        graph, compact_flat, isolated_count = build_compact_queen_graph(paired_valid)
        compact_count = int(compact_flat.size)
        compact_rows = compact_flat // reference_shape[1]
        compact_areas = pixel_area_km2_by_row[compact_rows]
        graph_edge_count = int(graph.n_edges)
        graph_component_count = int(graph.n_components)
        x_raw = dsi_item['data'].ravel()[compact_flat]
        y_values_by_mode = {'continuous': sdg_item['data'].ravel()[compact_flat], 'classified': sdg_item['classified'].ravel()[compact_flat].astype(np.float64)}
        x_sd = float(np.std(x_raw, ddof=0))
        if not np.isfinite(x_sd) or x_sd <= 0.0:
            raise ValueError(f'DSI standard deviation is invalid for {version_name}, {product_name}')
        x_z = (x_raw - np.mean(x_raw)) / x_sd
        if abs(float(np.mean(x_z))) > 1e-10 or abs(float(np.std(x_z)) - 1.0) > 1e-10:
            raise ValueError('DSI z-standardization validation failed')
        graph_rows.append([product_name, paired_count_before_islands, isolated_count, compact_count, graph_edge_count, graph_component_count, float(np.min(np.diff(graph.sparse.indptr))), float(np.max(np.diff(graph.sparse.indptr))), float(np.min(np.asarray(graph.sparse.sum(axis=1)).ravel())), float(np.max(np.asarray(graph.sparse.sum(axis=1)).ravel()))])
        for mode_index, mode_name in enumerate(SDG_MODES):
            analysis_start = time.perf_counter()
            y_raw = y_values_by_mode[mode_name]
            y_sd = float(np.std(y_raw, ddof=0))
            if not np.isfinite(y_sd) or y_sd <= 0.0:
                raise ValueError(f'SDG standard deviation is invalid for {version_name}, {product_name}, {mode_name}')
            y_z = (y_raw - np.mean(y_raw)) / y_sd
            if abs(float(np.mean(y_z))) > 1e-10 or abs(float(np.std(y_z)) - 1.0) > 1e-10:
                raise ValueError('SDG z-standardization validation failed')
            spearman = stats.spearmanr(x_raw, y_raw)
            kendall = stats.kendalltau(x_raw, y_raw)
            correlation_rows.append([product_name, mode_name, compact_count, float(spearman.statistic), float(spearman.pvalue), float(kendall.statistic), float(kendall.pvalue)])
            combination_seed = BASE_RANDOM_SEED + version_index * 10000 + PRODUCT_SEED_OFFSETS[product_name] + mode_index * 100
            np.random.seed(combination_seed + 1)
            moran_forward = Moran_BV(x_z, y_z, graph, transformation='R', permutations=PERMUTATIONS)
            np.random.seed(combination_seed + 2)
            moran_reverse = Moran_BV(y_z, x_z, graph, transformation='R', permutations=PERMUTATIONS)
            for direction, moran_result, direction_seed in [('X_DSI_to_WY_SDG', moran_forward, combination_seed + 1), ('Y_SDG_to_WX_DSI', moran_reverse, combination_seed + 2)]:
                global_moran_rows.append([product_name, mode_name, direction, compact_count, float(moran_result.I), float(moran_result.p_sim), float(moran_result.EI_sim), float(moran_result.seI_sim), float(moran_result.z_sim), float(moran_result.p_z_sim), PERMUTATIONS, direction_seed])
            local_result = Moran_Local_BV(x_z, y_z, graph, transformation='R', permutations=PERMUTATIONS, n_jobs=LOCAL_N_JOBS, keep_simulations=False, seed=LOCAL_PERMUTATION_SEEDS[product_name] + version_index * 10000 + mode_index * 100, island_weight=0, alternative='directed')
            local_i = np.asarray(local_result.Is, dtype=np.float64)
            local_p = np.asarray(local_result.p_sim, dtype=np.float64)
            local_q = benjamini_hochberg(local_p)
            spatial_lag_y = np.asarray(graph.sparse @ y_z, dtype=np.float64).ravel()
            if np.any(~np.isfinite(local_i)):
                raise ValueError('Non-finite local Moran statistic')
            if np.any((local_p < 0.0) | (local_p > 1.0)):
                raise ValueError('Local permutation p-value outside [0, 1]')
            if np.any((local_q < 0.0) | (local_q > 1.0)):
                raise ValueError('Local FDR q-value outside [0, 1]')
            significant = local_q < FDR_ALPHA
            lisa_codes = np.zeros(compact_count, dtype=np.uint8)
            lisa_codes[significant & (x_z > 0.0) & (spatial_lag_y > 0.0)] = 1
            lisa_codes[significant & (x_z < 0.0) & (spatial_lag_y < 0.0)] = 2
            lisa_codes[significant & (x_z > 0.0) & (spatial_lag_y < 0.0)] = 3
            lisa_codes[significant & (x_z < 0.0) & (spatial_lag_y > 0.0)] = 4
            zero_sign_significant = significant & ((x_z == 0.0) | (spatial_lag_y == 0.0))
            if np.any(zero_sign_significant):
                significant[zero_sign_significant] = False
                lisa_codes[zero_sign_significant] = 0
            classified_significant = int(np.count_nonzero(lisa_codes > 0))
            if classified_significant != int(np.count_nonzero(significant)):
                raise ValueError('Significant local pixels were not uniquely classified')
            total_area_km2 = float(np.sum(compact_areas))
            for lisa_code in range(5):
                code_mask = lisa_codes == lisa_code
                code_count = int(np.count_nonzero(code_mask))
                code_area_km2 = float(np.sum(compact_areas[code_mask]))
                lisa_area_rows.append([product_name, mode_name, lisa_code, LISA_LABELS[lisa_code], code_count, code_count / compact_count, code_area_km2, code_area_km2 / total_area_km2, compact_count, total_area_km2, FDR_ALPHA])
            local_i_grid = np.full(reference_shape, -9999.0, dtype=np.float32)
            local_q_grid = np.full(reference_shape, -9999.0, dtype=np.float32)
            lisa_grid = np.full(reference_shape, 255, dtype=np.uint8)
            local_i_grid.ravel()[compact_flat] = local_i.astype(np.float32)
            local_q_grid.ravel()[compact_flat] = local_q.astype(np.float32)
            lisa_grid.ravel()[compact_flat] = lisa_codes
            safe_product = product_name.replace('.', '_').replace('-', '_')
            raster_prefix = f'{safe_product}_{mode_name}'
            output_local_i = raster_dir / f'{raster_prefix}_local_bivariate_moran_I.tif'
            output_local_q = raster_dir / f'{raster_prefix}_local_bivariate_moran_qvalue.tif'
            output_lisa = raster_dir / f'{raster_prefix}_LISA_cluster.tif'
            write_raster(output_local_i, local_i_grid, reference_profile, 'float32', -9999.0)
            write_raster(output_local_q, local_q_grid, reference_profile, 'float32', -9999.0)
            write_raster(output_lisa, lisa_grid, reference_profile, 'uint8', 255)
            with rasterio.open(output_lisa) as check_src:
                check_grid = check_src.read(1)
                if check_src.shape != reference_shape:
                    raise ValueError('Output LISA raster shape mismatch')
                if not check_src.transform.almost_equals(reference_transform):
                    raise ValueError('Output LISA raster transform mismatch')
                if check_src.crs != reference_crs:
                    raise ValueError('Output LISA raster CRS mismatch')
                for lisa_code in range(5):
                    raster_count = int(np.count_nonzero(check_grid == lisa_code))
                    expected_count = int(np.count_nonzero(lisa_codes == lisa_code))
                    if raster_count != expected_count:
                        raise ValueError('LISA raster count does not match calculated count')
            elapsed = time.perf_counter() - analysis_start
            print(f'{version_name} | {product_name} | {mode_name}: n={compact_count:,}, I={moran_forward.I:.6f}, p={moran_forward.p_sim:.4f}, FDR-significant={classified_significant:,}, elapsed={elapsed:.1f}s')
        del graph
        elapsed_product = time.perf_counter() - product_start
        print(f'Completed {version_name} | {product_name}: edges={graph_edge_count:,}, components={graph_component_count:,}, elapsed={elapsed_product:.1f}s')
    if PRODUCTS_TO_RUN is not None:
        print('Selected-product raster calculation completed; the consolidated workbook is written only by a complete three-product run.')
        continue
    output_xlsx = output_dir / 'DSI_SDG_bivariate_Moran_LISA_direct_DSI_results_2004_2019.xlsx'
    workbook = Workbook()
    correlation_sheet = workbook.active
    correlation_sheet.title = 'Same-pixel correlation'
    global_sheet = workbook.create_sheet('Global Moran')
    lisa_sheet = workbook.create_sheet('LISA area')
    threshold_sheet = workbook.create_sheet('Classification threshold')
    parameter_sheet = workbook.create_sheet('Analysis parameters')
    source_sheet = workbook.create_sheet('Data sources')
    correlation_sheet.append(['Product', 'SDG mode', 'Valid non-isolated pixel count', 'Spearman rho', 'Spearman p', 'Kendall tau-b', 'Kendall p'])
    for row in correlation_rows:
        correlation_sheet.append(row)
    global_sheet.append(['Product', 'SDG mode', 'Direction', 'Valid non-isolated pixel count', 'Bivariate Moran I', 'Permutation P value', 'Permutation expectation', 'Permutation standard deviation', 'Permutation z score', 'Normal-approximation P value', 'Permutation count', 'Random seed'])
    for row in global_moran_rows:
        global_sheet.append(row)
    lisa_sheet.append(['Product', 'SDG mode', 'LISA code', 'Class interpretation', 'Pixel count', 'Pixel fraction', 'Area (km2)', 'Area fraction', 'Total valid non-isolated pixels', 'Total valid area (km2)', 'FDR threshold'])
    for row in lisa_area_rows:
        lisa_sheet.append(row)
    threshold_sheet.append(['Product', 'Valid SDG pixels in the full grid', 'SDG minimum', 'SDG maximum', 'Tlow=0.1×SDGmin', 'Thigh=0.1×SDGmax', 'Degraded code', 'Intermediate code', 'Improved code'])
    for product_name, sdg_item in sdg_rasters.items():
        threshold_sheet.append([product_name, sdg_item['valid_count'], sdg_item['min'], sdg_item['max'], sdg_item['threshold_low'], sdg_item['threshold_high'], -1, 0, 1])
    parameter_sheet.append(['Parameter', 'Setting', 'Unit', 'Description'])
    parameter_rows = [['Analysis name', 'Bivariate Moran I and LISA', '', 'Candidate analysis 3'], ['DSI version', version_name, '', 'All trends or conditional P<0.05 analysis'], ['DSI direction', 'X=DSI trend', '', 'Negative means improvement; positive means degradation'], ['Primary SDG analysis', 'Continuous summed SDG', '', 'Positive means improvement; negative means degradation'], ['Alternative SDG classification', 'Three-class codes -1/0/+1', '', 'Use the existing 10% threshold'], ['Standardization', 'Z scores among paired non-isolated pixels', '', 'Population standard deviation; ddof=0'], ['Spatial weights', 'Queen eight-neighbor adjacency', '', 'Actual adjacency among valid cells'], ['Weight transformation', 'Row-standardized', '', 'Weights sum to one for each non-isolated pixel'], ['Date-line adjacency', 'Not connected', '', 'Raster longitude range does not close a complete 360 degrees'], ['Global direction 1', 'X_DSI_to_WY_SDG', '', 'Local DSI and neighboring SDG'], ['Global direction 2', 'Y_SDG_to_WX_DSI', '', 'Local SDG and neighboring DSI'], ['Local LISA direction', 'X_DSI_to_WY_SDG', '', 'Used for HH/LL/HL/LH maps'], ['Permutation count', PERMUTATIONS, 'replicates', 'Consistent global and local settings'], ['Base random seed', BASE_RANDOM_SEED, '', 'Fixed offsets for each combination'], ['Local parallel jobs', LOCAL_N_JOBS, '', 'Worker count configured by STUDY_LISA_JOBS'], ['Local permutation matrix', 'Not retained', '', 'keep_simulations=False limits memory use'], ['Multiple testing', 'Benjamini-Hochberg FDR', '', 'Correct each version/product/SDG-mode combination separately'], ['FDR threshold', FDR_ALPHA, 'q value', 'Show LISA classes where q<0.05'], ['Area calculation', 'WGS84 ellipsoidal pixel area', 'km²', 'Calculated by latitude row'], ['Computation engine', 'libpysal Graph + esda', '', 'SciPy CSR sparse Queen graph']]
    for row in graph_rows:
        product_name = row[0]
        parameter_rows.extend([[f'{product_name}: paired pixels', row[1], 'pixels', 'Before excluding isolated pixels'], [f'{product_name}: isolated pixels', row[2], 'pixels', 'No valid Queen neighbors'], [f'{product_name}: non-isolated pixels', row[3], 'pixels', 'Final spatial-analysis sample'], [f'{product_name}: directed adjacency edges', row[4], 'edges', 'Sparse Graph'], [f'{product_name}: connected components', row[5], 'items', 'Non-isolated spatial graph'], [f'{product_name}: minimum neighbors', row[6], 'items', 'Queen neighbors'], [f'{product_name}: maximum neighbors', row[7], 'items', 'Queen neighbors'], [f'{product_name}: minimum row-weight sum', row[8], '', 'must equal 1'], [f'{product_name}: maximum row-weight sum', row[9], '', 'must equal 1']])
    software_rows = [['Software version: Python', platform.python_version(), '', ''], ['Software version: NumPy', np.__version__, '', ''], ['Software version: SciPy', scipy.__version__, '', ''], ['Software version: Numba', numba.__version__, '', ''], ['Software version: Rasterio', rasterio.__version__, '', ''], ['Software version: PyProj', pyproj.__version__, '', ''], ['Software version: libpysal', libpysal.__version__, '', ''], ['Software version: esda', esda.__version__, '', ''], ['Software version: openpyxl', openpyxl.__version__, '', ''], ['Generated at', datetime.now().astimezone().isoformat(timespec='seconds'), '', ''], ['Total runtime for current version', time.perf_counter() - version_start, 'seconds', 'Includes statistics and output']]
    for row in parameter_rows + software_rows:
        parameter_sheet.append(row)
    source_sheet.append(['Product', 'Data type', 'File path', 'Raster dimensions', 'Affine transform', 'Coordinate reference system'])
    source_sheet.append(['DSI', 'DSI trend', str(dsi_item['path']), str(reference_shape), str(reference_transform), str(reference_crs)])
    for product_name, sdg_item in sdg_rasters.items():
        source_sheet.append([product_name, 'SDG sum', str(sdg_item['path']), str(reference_shape), str(reference_transform), str(reference_crs)])
    style_plain_workbook(workbook)
    for row in range(2, correlation_sheet.max_row + 1):
        correlation_sheet.cell(row, 3).number_format = '#,##0'
        for column in [4, 6]:
            correlation_sheet.cell(row, column).number_format = '0.000000'
        for column in [5, 7]:
            correlation_sheet.cell(row, column).number_format = '0.000E+00'
    for row in range(2, global_sheet.max_row + 1):
        global_sheet.cell(row, 4).number_format = '#,##0'
        for column in [5, 7, 8, 9]:
            global_sheet.cell(row, column).number_format = '0.000000'
        for column in [6, 10]:
            global_sheet.cell(row, column).number_format = '0.000E+00'
        global_sheet.cell(row, 11).number_format = '#,##0'
        global_sheet.cell(row, 12).number_format = '0'
    for row in range(2, lisa_sheet.max_row + 1):
        lisa_sheet.cell(row, 3).number_format = '0'
        lisa_sheet.cell(row, 5).number_format = '#,##0'
        lisa_sheet.cell(row, 6).number_format = '0.0000%'
        lisa_sheet.cell(row, 7).number_format = '#,##0.000'
        lisa_sheet.cell(row, 8).number_format = '0.0000%'
        lisa_sheet.cell(row, 9).number_format = '#,##0'
        lisa_sheet.cell(row, 10).number_format = '#,##0.000'
        lisa_sheet.cell(row, 11).number_format = '0.0000'
    for row in range(2, threshold_sheet.max_row + 1):
        threshold_sheet.cell(row, 2).number_format = '#,##0'
        for column in range(3, 7):
            threshold_sheet.cell(row, column).number_format = '0.000000'
        for column in range(7, 10):
            threshold_sheet.cell(row, column).number_format = '0'
    workbook.save(output_xlsx)
    verified_workbook = load_workbook(output_xlsx, read_only=True, data_only=True)
    expected_sheets = ['Same-pixel correlation', 'Global Moran', 'LISA area', 'Classification threshold', 'Analysis parameters', 'Data sources']
    if verified_workbook.sheetnames != expected_sheets:
        raise ValueError(f'Unexpected workbook sheets: {verified_workbook.sheetnames}')
    for sheet_name in expected_sheets:
        verified_sheet = verified_workbook[sheet_name]
        if verified_sheet.max_row < 2 or verified_sheet.max_column < 2:
            raise ValueError(f'Worksheet is unexpectedly empty: {sheet_name}')
        for row in verified_sheet.iter_rows(min_row=2, values_only=True):
            for value in row:
                if isinstance(value, float) and (not np.isfinite(value)):
                    raise ValueError(f'Non-finite workbook value in {sheet_name}')
    verified_workbook.close()
    print(f'Saved workbook: {output_xlsx}')
    print(f'Completed version {version_name} in {time.perf_counter() - version_start:.1f}s')
print('All bivariate Moran and LISA calculations completed successfully.')
