"""Independent weighting sensitivity analysis; original processing is read-only."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / '03_analysis/07_sensitivity/data/weights'
ORIGINAL_CODE_ROOT = _archive_root / '03_analysis/01_dsi/code'
SOURCE_ROOT = PROJECT_ROOT / '02_preprocessing/data'
INDEX_ROOT = PROJECT_ROOT / '03_analysis/01_dsi/data'
COMMON_ROOT = PROJECT_ROOT / '03_analysis/01_dsi/data/01indices/01all_indicators'
ORIGINAL_DSI_ROOT = PROJECT_ROOT / '03_analysis/01_dsi/data/02DI'
WEIGHT_ROOT = PROJECT_ROOT / '03_analysis/01_dsi/data/01indices/02indicator_groups'
FIGURE_ROOT = FIGURES_ROOT
ABLATION_ROOT = PROJECT_ROOT / '03_analysis/04_npv_ablation'
VERIFY_ROOT = PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1'
LONG_TREND_ROOT = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'
INDICATORS = ('Albedo', 'SMI', 'TGSI', 'Wet', 'LST', 'PV', 'NPV')
WITHOUT_NPV = INDICATORS[:-1]
ALL_PREPROCESS_INDICATORS = ('Albedo', 'DFI', 'SMI', 'TGSI', 'Wet', 'LST', 'MSAVI', 'BS', 'DA', 'IS', 'PV', 'NPV')
YEARS = tuple(range(2001, 2022))
ABLATION_YEARS = tuple(range(2004, 2020))
SCENARIOS = {'entropy_G15': 15, 'entropy_G10': 10, 'entropy_G20': 20, 'equal': None}
BASELINE = 'entropy_G15'
VARIANTS = ('without_NPV', 'with_NPV')
PRODUCTS = ('FAO-WOCAT', 'JRC', 'Trends.Earth')
CLASS_NAMES = ('Significant degradation', 'No significant trend', 'Significant recovery')
INPUT_SCALE_MAX = 10000.0
INPUT_NODATA = 32767.0
ANALYSIS_NODATA = -9999.0
ALPHA = 0.05
SDG_THRESHOLD_FRACTION = 0.1
BLOCK_SIZE_DEG = 5.0
BOOTSTRAP_REPLICATES = 1999
BOOTSTRAP_BATCH = 100
RANDOM_SEED = 20260913
EXPECTED_LONG_PIXELS = 474119
EXPECTED_ABLATION_PIXELS = 454755
EXPECTED_EXTERNAL_PIXELS = 420238
RUN_WEIGHTS = True
RUN_DSI = True
RUN_EVALUATION = True
RUN_REPORT = True

def input_path(indicator, year):
    category = '02FVC' if indicator in ('PV', 'NPV') else '01RS_indexs'
    return PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021' / ('fvc' if indicator in ('PV', 'NPV') else 'rs_indices') / indicator / f'{indicator}_{year}.tif'

def mask_path(year):
    return PROJECT_ROOT / '03_analysis/01_dsi/data/01indices/01all_indicators/04valid_mask' / f'valid_mask_{year}.tif'

def group_name(variant):
    return '_'.join(INDICATORS if variant == 'with_NPV' else WITHOUT_NPV)

def original_dsi_path(variant, year):
    return ORIGINAL_DSI_ROOT / group_name(variant) / '02study_period_weights' / f'DI_{year}.tif'

def dsi_path(scenario, variant, year):
    if scenario == BASELINE:
        return original_dsi_path(variant, year)
    return PROJECT_ROOT / '03_analysis/07_sensitivity/data/weights/scenarios' / scenario / 'annual_DSI' / variant / f'DI_{year}.tif'

def external_path(product):
    return PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum' / f'TrendsEarth_SDG15.3.1_2000-2023_{product}' / 'SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif'

def stage_path(stage):
    return PROJECT_ROOT / '03_analysis/07_sensitivity/data/weights/audit' / f'{stage}_complete.json'
