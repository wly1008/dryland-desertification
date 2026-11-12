"""Configure DSI indicators, years, granulation, paths, and weighting."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import re
from pathlib import Path
PROJECT_DIR = CODE_ROOT / '03_analysis/01_dsi/code'
STATISTIC_DIR_NAME = '01mean'
PERIOD_DIR_NAME = '02non-growing_season'
PERIOD_OUTPUT_DIR_NAME = re.sub('^\\d+', '', PERIOD_DIR_NAME)
STATISTIC_NAME_MAP = {'01mean': 'mean', '02median': 'median'}
if STATISTIC_DIR_NAME not in STATISTIC_NAME_MAP:
    raise ValueError(f'Unsupported STATISTIC_DIR_NAME: {STATISTIC_DIR_NAME}; available values: {tuple(STATISTIC_NAME_MAP)}')
STATISTIC_NAME = STATISTIC_NAME_MAP[STATISTIC_DIR_NAME]
RS_PERIOD_ROOT = PROJECT_ROOT / '02_preprocessing/data/normalization_layout/rs/02non-growing_season'
FVC_PERIOD_ROOT = PROJECT_ROOT / '02_preprocessing/data/normalization_layout/fvc/02non-growing_season'
RS_MEAN_ROOT = PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/rs_indices'
FVC_MEAN_ROOT = PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/fvc'
PERIOD_KIND = PERIOD_OUTPUT_DIR_NAME
PERIOD_METADATA_CONFIG = {'non-growing_season': {'definition': 'October through April of the following year', 'start_month': 10, 'end_year_offset': 1, 'end_month': 4, 'source_count': 7}, 'growing_season': {'definition': 'May through September of the same year', 'start_month': 5, 'end_year_offset': 0, 'end_month': 9, 'source_count': 5}, 'year': {'definition': 'January through December of the same year', 'start_month': 1, 'end_year_offset': 0, 'end_month': 12, 'source_count': 12}}
PERIOD_METADATA = PERIOD_METADATA_CONFIG[PERIOD_KIND]
PERIOD_DEFINITION = PERIOD_METADATA['definition']
PERIOD_START_MONTH = PERIOD_METADATA['start_month']
PERIOD_END_YEAR_OFFSET = PERIOD_METADATA['end_year_offset']
PERIOD_END_MONTH = PERIOD_METADATA['end_month']
INDICATOR_CONFIG = {'Albedo': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/rs_indices/Albedo', 'direction': 'positive', 'direction_status': 'already unified; higher means more desertification', 'selection_ready': True}, 'DFI': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/rs_indices/DFI', 'direction': 'positive_after_normalization', 'direction_status': 'confirmed and unified; higher means more desertification', 'selection_ready': True}, 'SMI': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/rs_indices/SMI', 'direction': 'positive', 'direction_status': 'already unified; higher salinity means more desertification', 'selection_ready': True}, 'TGSI': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/rs_indices/TGSI', 'direction': 'positive', 'direction_status': 'already unified; higher means more desertification', 'selection_ready': True}, 'Wet': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/rs_indices/Wet', 'direction': 'negative_before_normalization', 'direction_status': 'already reversed; higher means more desertification', 'selection_ready': True}, 'LST': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/rs_indices/LST', 'direction': 'positive', 'direction_status': 'already unified; higher means more desertification', 'selection_ready': True}, 'MSAVI': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/rs_indices/MSAVI', 'direction': 'positive_after_normalization', 'direction_status': 'confirmed and unified; higher means more desertification', 'selection_ready': True}, 'BS': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/fvc/BS', 'direction': 'positive_after_normalization', 'direction_status': 'confirmed and unified; higher means more desertification', 'selection_ready': True}, 'DA': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/fvc/DA', 'direction': 'positive_after_normalization', 'direction_status': 'confirmed and unified; higher means more desertification', 'selection_ready': True}, 'IS': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/fvc/IS', 'direction': 'positive_after_normalization', 'direction_status': 'confirmed and unified; higher means more desertification', 'selection_ready': True}, 'PV': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/fvc/PV', 'direction': 'negative_before_normalization', 'direction_status': 'already reversed; higher means more desertification', 'selection_ready': True}, 'NPV': {'input_dir': PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/fvc/NPV', 'direction': 'negative_before_normalization', 'direction_status': 'already reversed; higher means more desertification', 'selection_ready': True}}
AVAILABLE_INDICATORS = tuple(INDICATOR_CONFIG)
PREPROCESS_INDICATORS = AVAILABLE_INDICATORS
import os
DSI_VARIANT = os.environ.get('REPRO_DSI_VARIANT', 'with_NPV')
if DSI_VARIANT not in ('with_NPV', 'without_NPV'):
    raise ValueError('REPRO_DSI_VARIANT must be with_NPV or without_NPV')
SELECTED_INDICATORS = ('Albedo', 'SMI', 'TGSI', 'Wet', 'LST', 'PV') + (('NPV',) if DSI_VARIANT == 'with_NPV' else ())
SELECTED_INDICATORS_IN_CONFIG_ORDER = tuple((indicator for indicator in AVAILABLE_INDICATORS if indicator in SELECTED_INDICATORS))
INDICATOR_GROUP_NAME = '_'.join(SELECTED_INDICATORS_IN_CONFIG_ORDER)
INPUT_FILE_PATTERN = '{indicator}_{year}.tif'
START_YEAR = 2001
END_YEAR = 2021
YEARS = tuple(range(START_YEAR, END_YEAR + 1))
STUDY_PERIOD_SHEET_NAME = 'study_period'
INPUT_NODATA = 32767.0
INPUT_BAND_COUNT = 1
INPUT_DTYPE = 'float32'
INPUT_SCALE = 1.0
INPUT_OFFSET = 0.0
INPUT_SCALE_MAX = 10000.0
NORMALIZED_MIN = 0.0
NORMALIZED_MAX = 1.0
CLIP_NORMALIZED_VALUES = True
EXPECTED_CRS = 'EPSG:4326'
EXPECTED_WIDTH = 3591
EXPECTED_HEIGHT = 668
EXPECTED_X_RESOLUTION = 0.1
EXPECTED_Y_RESOLUTION = 0.1
EXPECTED_SOURCE_COUNT = PERIOD_METADATA['source_count']
BIN_COUNT = 15
BIN_EDGES = tuple((index / BIN_COUNT for index in range(BIN_COUNT + 1)))
RECLASS_MIN_VALUE = 1
RECLASS_MAX_VALUE = BIN_COUNT
RECLASS_DTYPE = 'uint8'
RECLASS_NODATA = 0
MASK_DTYPE = 'uint8'
MASK_NODATA = 0
MASK_INVALID_VALUE = 0
MASK_VALID_VALUE = 1
MASK_NBITS = 2
MASK_PREDICTOR = None
MASK_FILE_PATTERN = 'valid_mask_{year}.tif'
DESERTIFICATION_OUTPUT_ROOT = PROJECT_ROOT / '03_analysis/01_dsi/data'
DI_INDICATOR_GROUP_OUTPUT_ROOT = DESERTIFICATION_OUTPUT_ROOT / '02DI' / INDICATOR_GROUP_NAME
DI_QUALITY_INDICATOR_GROUP_ROOT = DESERTIFICATION_OUTPUT_ROOT / '03quality' / INDICATOR_GROUP_NAME
ANNUAL_WEIGHTS_DIR_NAME = '01annual_weights'
STUDY_PERIOD_WEIGHTS_DIR_NAME = '02study_period_weights'
DI_WEIGHT_MODES = ('annual_weights', 'study_period_weights')
DI_OUTPUT_DIRS = {'annual_weights': DI_INDICATOR_GROUP_OUTPUT_ROOT / ANNUAL_WEIGHTS_DIR_NAME, 'study_period_weights': DI_INDICATOR_GROUP_OUTPUT_ROOT / STUDY_PERIOD_WEIGHTS_DIR_NAME}
DI_QUALITY_OUTPUT_DIRS = {'annual_weights': DI_QUALITY_INDICATOR_GROUP_ROOT / ANNUAL_WEIGHTS_DIR_NAME, 'study_period_weights': DI_QUALITY_INDICATOR_GROUP_ROOT / STUDY_PERIOD_WEIGHTS_DIR_NAME}
DI_QUALITY_XLSX_BY_MODE = {weight_mode: output_dir / 'DI_quality.xlsx' for weight_mode, output_dir in DI_QUALITY_OUTPUT_DIRS.items()}
DI_QUALITY_PERCENTILES = (1.0, 5.0, 50.0, 95.0, 99.0)
DI_FILE_PATTERN = 'DI_{year}.tif'
DI_DTYPE = 'float32'
DI_NODATA = 32767.0
DI_VALID_MIN = 0.0
DI_VALID_MAX = 1.0
INTERMEDIATE_OUTPUT_ROOT = DESERTIFICATION_OUTPUT_ROOT / '01indices'
COMMON_PREPROCESS_OUTPUT_ROOT = INTERMEDIATE_OUTPUT_ROOT / '01all_indicators'
INDICATOR_GROUP_OUTPUT_ROOT = INTERMEDIATE_OUTPUT_ROOT / '02indicator_groups' / INDICATOR_GROUP_NAME
DATA_QUALITY_OUTPUT_DIR = COMMON_PREPROCESS_OUTPUT_ROOT / '02data_quality'
GRANULATION_OUTPUT_DIR = COMMON_PREPROCESS_OUTPUT_ROOT / '03knowledge_granulation'
VALID_MASK_OUTPUT_DIR = COMMON_PREPROCESS_OUTPUT_ROOT / '04valid_mask'
CLASS_STATISTICS_OUTPUT_DIR = COMMON_PREPROCESS_OUTPUT_ROOT / '05class_statistics'
ENTROPY_WEIGHT_OUTPUT_DIR = INDICATOR_GROUP_OUTPUT_ROOT / '06Gk_Er_W'
DATA_QUALITY_XLSX = DATA_QUALITY_OUTPUT_DIR / 'data_quality.xlsx'
VALID_PIXEL_COUNT_XLSX = VALID_MASK_OUTPUT_DIR / 'valid_pixel_count.xlsx'
CLASS_COUNTS_XLSX = CLASS_STATISTICS_OUTPUT_DIR / 'class_counts.xlsx'
CLASS_PROBABILITIES_XLSX = CLASS_STATISTICS_OUTPUT_DIR / 'class_probabilities.xlsx'
GK_ER_W_XLSX = ENTROPY_WEIGHT_OUTPUT_DIR / 'Gk_Er_W.xlsx'
GRANULATION_FILE_PATTERN = '{indicator}_{year}.tif'
GTIFF_TILED = True
GTIFF_BLOCK_X_SIZE = 256
GTIFF_BLOCK_Y_SIZE = 256
GTIFF_COMPRESSION = 'deflate'
GTIFF_ZLEVEL = 6
GTIFF_BIGTIFF = 'IF_SAFER'
INTEGER_PREDICTOR = 2
FLOAT_PREDICTOR = 3
GRID_ABSOLUTE_TOLERANCE = 1e-12
PROBABILITY_SUM_TOLERANCE = 1e-10
WEIGHT_SUM_TOLERANCE = 1e-10
VALUE_RANGE_TOLERANCE = 1e-07
EXCEL_ENGINE = 'openpyxl'
EXCEL_INDEX_LABEL = 'index'
EXCEL_COUNT_ROW_LABEL = 'count'
EXCEL_PROBABILITY_SUM_ROW_LABEL = 'sum'
OVERWRITE = True
if not SELECTED_INDICATORS:
    raise ValueError('SELECTED_INDICATORS must not be empty')
if len(SELECTED_INDICATORS) != len(set(SELECTED_INDICATORS)):
    raise ValueError('SELECTED_INDICATORS contains duplicates')
UNKNOWN_INDICATORS = tuple((indicator for indicator in SELECTED_INDICATORS if indicator not in INDICATOR_CONFIG))
if UNKNOWN_INDICATORS:
    raise KeyError(f'Unconfigured indicators: {UNKNOWN_INDICATORS}')
if len(SELECTED_INDICATORS_IN_CONFIG_ORDER) != len(SELECTED_INDICATORS):
    raise RuntimeError('Cannot establish the fixed selected-indicator order')
if not INDICATOR_GROUP_NAME:
    raise ValueError('INDICATOR_GROUP_NAME must not be empty')
if tuple(DI_OUTPUT_DIRS) != DI_WEIGHT_MODES:
    raise RuntimeError('DI_OUTPUT_DIRS modes or ordering differ from DI_WEIGHT_MODES')
if tuple(DI_QUALITY_OUTPUT_DIRS) != DI_WEIGHT_MODES:
    raise RuntimeError('DI_QUALITY_OUTPUT_DIRS modes or ordering differ from DI_WEIGHT_MODES')
if tuple(DI_QUALITY_XLSX_BY_MODE) != DI_WEIGHT_MODES:
    raise RuntimeError('DI_QUALITY_XLSX_BY_MODE modes or ordering differ from DI_WEIGHT_MODES')
if len(set(DI_OUTPUT_DIRS.values())) != len(DI_OUTPUT_DIRS):
    raise RuntimeError('Weighting modes must have distinct DSI output directories')
if len(set(DI_QUALITY_OUTPUT_DIRS.values())) != len(DI_QUALITY_OUTPUT_DIRS):
    raise RuntimeError('Weighting modes must have distinct quality-output directories')
UNCONFIRMED_PREPROCESS_INDICATORS = tuple((indicator for indicator in PREPROCESS_INDICATORS if not INDICATOR_CONFIG[indicator]['selection_ready']))
if UNCONFIRMED_PREPROCESS_INDICATORS:
    raise ValueError(f'Desertification directions are unconfirmed for preprocessing indicators: {UNCONFIRMED_PREPROCESS_INDICATORS}')
UNCONFIRMED_SELECTED_INDICATORS = tuple((indicator for indicator in SELECTED_INDICATORS if not INDICATOR_CONFIG[indicator]['selection_ready']))
if UNCONFIRMED_SELECTED_INDICATORS:
    raise ValueError(f'Desertification directions are unconfirmed; cannot select: {UNCONFIRMED_SELECTED_INDICATORS}. Verify directions and update direction, direction_status, and selection_ready in INDICATOR_CONFIG.')
if START_YEAR > END_YEAR:
    raise ValueError('START_YEAR must not exceed END_YEAR')
if BIN_COUNT < 2 or BIN_COUNT > 254:
    raise ValueError('BIN_COUNT must be between 2 and 254 for UInt8 classes and NoData')
if RECLASS_MIN_VALUE != 1 or RECLASS_MAX_VALUE != BIN_COUNT:
    raise ValueError('Reclassified values must use consecutive codes 1 through BIN_COUNT')
if RECLASS_NODATA in range(RECLASS_MIN_VALUE, RECLASS_MAX_VALUE + 1):
    raise ValueError('RECLASS_NODATA must not overlap valid class codes')
if MASK_NBITS != 2:
    raise ValueError('Current valid masks require MASK_NBITS=2')
if MASK_NODATA != 0 or MASK_INVALID_VALUE != 0 or MASK_VALID_VALUE != 1:
    raise ValueError('Mask codes must be valid=1 and invalid/NoData=0')
if INPUT_NODATA <= INPUT_SCALE_MAX:
    raise ValueError('INPUT_NODATA must lie outside 0 through INPUT_SCALE_MAX')
if DI_NODATA >= DI_VALID_MIN and DI_NODATA <= DI_VALID_MAX:
    raise ValueError('DI_NODATA must lie outside the valid DSI range')
