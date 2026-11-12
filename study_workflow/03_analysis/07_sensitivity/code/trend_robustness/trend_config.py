"""A versus D only: fixed G15 DSI; no B/C or weighting cross-product."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / '03_analysis/07_sensitivity'
OUT = PROJECT_ROOT / '03_analysis/07_sensitivity/data/presentation/trend_robustness'
FIGURE_ROOT = FIGURES_ROOT
SOURCE_ROOT = PROJECT_ROOT / '02_preprocessing/data'
DSI_ROOT = PROJECT_ROOT / '03_analysis/01_dsi/data/02DI'
LONG_ROOT = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'
ABLATION_ROOT = PROJECT_ROOT / '03_analysis/04_npv_ablation'
PATHWAY_ROOT = PROJECT_ROOT / '03_analysis/05_cover_combinations'
EXTERNAL_ROOT = PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum'
YEARS = tuple(range(2001, 2022))
SHORT_YEARS = tuple(range(2004, 2020))
COMPONENTS = ('PV', 'NPV', 'BS')
PRODUCTS = ('FAO-WOCAT', 'JRC', 'Trends.Earth')
METHODS = ('A_OLS', 'D_Sen_MK_BH')
ALPHA = 0.05
CHUNK_PIXELS = 8192
BOOTSTRAP_REPLICATES = 1999
BOOTSTRAP_BATCH = 100
BLOCK_SIZE_DEG = 5.0
SEED = 20260913
MAIN_PATHWAYS = (14, 15, 17, 13, 23, 11, 19, 9)
EXPECTED_LONG = 474119
EXPECTED_SHORT = 454755
EXPECTED_EXTERNAL = 420238
EXPECTED_FVC = 457258
EXPECTED_JOINT = 454669

def annual_path(variant, year):
    group = 'Albedo_SMI_TGSI_Wet_LST_PV_NPV' if variant == 'with_NPV' else 'Albedo_SMI_TGSI_Wet_LST_PV'
    return DSI_ROOT / group / '02study_period_weights' / f'DI_{year}.tif'

def external_path(product):
    return EXTERNAL_ROOT / f'TrendsEarth_SDG15.3.1_2000-2023_{product}' / 'SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif'
