"""Calculate global bivariate Moran statistics for DSI-SDG comparisons."""
from __future__ import annotations
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import csv
import json
from pathlib import Path
import numpy as np
import rasterio
from esda.moran import Moran_BV
from libpysal.graph import Graph
from scipy import sparse
SCRIPT_DIR = CODE_ROOT / '03_analysis/03_sdg_comparison/code'
WORKING_ROOT = PROJECT_ROOT / '03_analysis/03_sdg_comparison'
VERIFY_DIR = PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1'
DSI_PATH = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_2004_2019.tif'
SDG_PATHS = {'FAO-WOCAT': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_FAO-WOCAT/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif', 'JRC': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_JRC/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif', 'Trends.Earth': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_Trends.Earth/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif'}
OUTPUT_DIR = PROJECT_ROOT / '03_analysis/03_sdg_comparison/data/global_moran'
OUTPUT_CSV = PROJECT_ROOT / '03_analysis/03_sdg_comparison/data/global_moran/global_bivariate_moran_results.csv'
OUTPUT_JSON = PROJECT_ROOT / '03_analysis/03_sdg_comparison/data/global_moran/global_bivariate_moran_results.json'
PERMUTATIONS = 999
BASE_RANDOM_SEED = 20260818
ROW_STANDARDIZATION_TOLERANCE = 1e-06

def read_raster(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return float data, validity mask, and grid metadata."""
    if not path.exists():
        raise FileNotFoundError(f'Missing required raster: {path}')
    with rasterio.open(path) as src:
        masked = src.read(1, masked=True)
        data = np.asarray(masked.data, dtype=np.float64)
        valid = ~np.ma.getmaskarray(masked) & np.isfinite(data)
        metadata = {'shape': src.shape, 'transform': src.transform, 'crs': src.crs}
    return (data, valid, metadata)

def build_compact_queen_graph(valid_mask: np.ndarray) -> tuple[Graph, np.ndarray, int]:
    """Build row-standardized Queen weights for valid cells, excluding islands."""
    height, width = valid_mask.shape
    compact_grid = np.full((height, width), -1, dtype=np.int32)
    original_flat_indices = np.flatnonzero(valid_mask.ravel())
    compact_grid.ravel()[original_flat_indices] = np.arange(original_flat_indices.size, dtype=np.int32)
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
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
        raise ValueError(f'Row-standardized Queen weights failed validation: maximum row-sum error={maximum_row_sum_error:.3e}')
    if np.any(graph.sparse.diagonal() != 0.0):
        raise ValueError('Queen graph contains self-neighbor links')
    return (graph, original_flat_indices, isolated_count)

def z_standardize(values: np.ndarray, label: str) -> np.ndarray:
    standard_deviation = float(np.std(values, ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError(f'Invalid standard deviation for {label}')
    standardized = (values - np.mean(values)) / standard_deviation
    if abs(float(np.mean(standardized))) > 1e-10:
        raise ValueError(f'Mean-zero validation failed for {label}')
    if abs(float(np.std(standardized, ddof=0)) - 1.0) > 1e-10:
        raise ValueError(f'Unit-variance validation failed for {label}')
    return standardized

def validate_alignment(reference: dict, candidate: dict, product: str) -> None:
    if candidate['shape'] != reference['shape']:
        raise ValueError(f'Raster shape mismatch for {product}')
    if not candidate['transform'].almost_equals(reference['transform']):
        raise ValueError(f'Raster transform mismatch for {product}')
    if candidate['crs'] != reference['crs']:
        raise ValueError(f'Raster CRS mismatch for {product}')

def calculate_results() -> list[dict]:
    dsi_data, dsi_valid, dsi_metadata = read_raster(DSI_PATH)
    results: list[dict] = []
    for product_index, (product, sdg_path) in enumerate(SDG_PATHS.items()):
        sdg_data, sdg_valid, sdg_metadata = read_raster(sdg_path)
        validate_alignment(dsi_metadata, sdg_metadata, product)
        paired_valid = dsi_valid & sdg_valid
        paired_count_before_islands = int(np.count_nonzero(paired_valid))
        graph, compact_flat, isolated_count = build_compact_queen_graph(paired_valid)
        x_raw = dsi_data.ravel()[compact_flat]
        y_raw = sdg_data.ravel()[compact_flat]
        x_z = z_standardize(x_raw, f'DSI ({product})')
        y_z = z_standardize(y_raw, f'SDG sum ({product})')
        seed = BASE_RANDOM_SEED + product_index * 1000 + 1
        np.random.seed(seed)
        moran = Moran_BV(x_z, y_z, graph, transformation='R', permutations=PERMUTATIONS)
        result = {'product': product, 'paired_valid_pixels': paired_count_before_islands, 'island_pixels_excluded': isolated_count, 'valid_non_island_pixels': int(compact_flat.size), 'morans_i': float(moran.I), 'permutation_expected_i': float(moran.EI_sim), 'permutation_sd': float(moran.seI_sim), 'permutation_z': float(moran.z_sim), 'permutation_p': float(moran.p_sim), 'normal_approximation_p': float(moran.p_z_sim), 'permutations': PERMUTATIONS, 'random_seed': seed}
        if not all((np.isfinite(result[key]) for key in ['morans_i', 'permutation_expected_i', 'permutation_sd', 'permutation_z', 'permutation_p', 'normal_approximation_p'])):
            raise ValueError(f'Non-finite Moran result for {product}')
        results.append(result)
        print(f"{product}: n={result['valid_non_island_pixels']:,}, I={result['morans_i']:.9f}, z={result['permutation_z']:.3f}, P={result['permutation_p']:.4f}")
    return results

def write_results(results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with OUTPUT_CSV.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    payload = {'title': 'Global bivariate Moran results', 'analysis_period': '2004-2019', 'direction': 'DSI trend to spatial lag of continuous SDG sum', 'dsi_orientation': 'Direct DSI: negative = improvement; positive = degradation', 'spatial_weights': 'Row-standardized Queen contiguity', 'island_handling': 'Excluded', 'results': results}
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'CSV:  {OUTPUT_CSV}')
    print(f'JSON: {OUTPUT_JSON}')

def main() -> None:
    results = calculate_results()
    if len(results) != len(SDG_PATHS):
        raise ValueError('The output must contain exactly one row per SDG product')
    write_results(results)
if __name__ == '__main__':
    main()
