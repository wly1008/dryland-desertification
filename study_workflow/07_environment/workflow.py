"""Ordered scientific workflow. VRT creation and pre-VRT tile mosaics are outside scope."""
from pathlib import Path
from project_paths import PROJECT_ROOT, study_boundary, CODE_ROOT, parameter_file, VRT_ROOT, raster_source
UP = '01_upstream_reference/code/'
RES = UP + 'resample_to_0p1/'
PRE = '02_preprocessing/code/'
DSI = '03_analysis/01_dsi/code/'
AN = '03_analysis/'
PLOT_TASKS = {'fig1': '04_plotting/code/study_area/01_plot_study_area.py', 'fig3': '04_plotting/code/sdg_comparison/01_dsi_sdg_density_regression.py', 'fig4': '04_plotting/code/sdg_comparison/02_sdg_class_dsi_violin.py', 'fig5': '04_plotting/code/sdg_comparison/03_dsi_sdg_bivariate_lisa.py', 'fig6': '04_plotting/code/spacetime/04_plot_three_year_cartopy_comparison.py', 'fig7': '04_plotting/code/spacetime/04_plot_DI_trend_and_significance.py', 'fig8': '04_plotting/code/npv_ablation/01_figure8_npv_state_effect.py', 'fig9': '04_plotting/code/npv_ablation/03_plot_figure9_npv_trend_effect.py', 'fig10': '04_plotting/code/cover_combinations/09_plot_fvc_dsi_relationship.py', 'fig11_12': '04_plotting/code/land_transitions/06_plot_publication_figures.py'}
TASKS = {**PLOT_TASKS, 'table2': '04_plotting/code/main_tables/02_build_table2.py', 'table_weights': '04_plotting/code/supplementary/01_build_support_tables.py', 'table_trends': '04_plotting/code/supplementary/04_build_trend_tables.py', 'test_weights': AN + '07_sensitivity/code/weights/test_numerics.py', 'test_trends': AN + '07_sensitivity/code/trend_robustness/test_trend_numerics.py', 'compute_npv': AN + '04_npv_ablation/code/02_compute_figure9_npv_trend_effect.py', 'compute_cover': AN + '05_cover_combinations/code/07_compute_fvc_combinations.py', 'compute_dsi_response': AN + '05_cover_combinations/code/08_compute_main_dsi_response.py', 'compute_land': AN + '06_land_transitions/code/01_compute_lucc_dsi.py', 'compute_lisa': AN + '03_sdg_comparison/code/03a_dsi_sdg_bivariate_moran_compute_direct_dsi.py', 'compute_global_moran': AN + '03_sdg_comparison/code/03c_dsi_sdg_global_bivariate_moran_table.py', 'rebuild_indices': UP + 'index_calculation/calc_all_indices.py', 'fixed_bounds': UP + 'index_calculation/get_fixed_bounds.py', 'rebuild_normalization': UP + 'index_calculation/normalization_fixed_bounds.py', 'resample_reflectance': RES + '01_resample_MODIS_to_0.1deg.py', 'resample_indices': RES + '02_resample_RS_indices_to_0.1deg_INT16.py', 'resample_fvc': RES + '03_resample_FVC_components_to_0.1deg_INT16.py', 'prepare_mask': PRE + 'auxiliary_alignment/prepare_study_mask.py', 'align_land_cover': PRE + 'auxiliary_alignment/align_modis_lulc_to_0p1.py', 'reclassify_land_cover': PRE + 'auxiliary_alignment/reclassify_align_modis_lulc_to_0p1.py', 'resample_sdg': RES + '08_resample_SDG_sum_to_0.1deg.py', 'unify_fvc_direction': PRE + 'component_direction/01normalize_PV_NPV_direction.py', 'seasonal_means': PRE + 'seasonal_aggregation/05calc_period_means.py', 'build_dsi': DSI + '00run_pipeline.py', 'build_trends': AN + '02_trends/code/compute_ols_trends.py', 'sensitivity_weights': AN + '07_sensitivity/code/weights/00run_pipeline.py', 'sensitivity_trends': AN + '07_sensitivity/code/trend_robustness/03_compute_trend_D.py'}
TASK_ARGUMENTS = {'build_dsi': ['--variant', 'both'], 'build_trends': ['--variant', 'both', '--all-periods', '--publish']}
STAGES = ('preprocessing', 'dsi', 'trends', 'analysis', 'sensitivity', 'plots', 'tables')

def stage_tasks(source_level='raw'):
    before = ['rebuild_indices', 'fixed_bounds', 'rebuild_normalization'] if source_level == 'raw' else ['fixed_bounds', 'rebuild_normalization'] if source_level == 'indices' else []
    return {'preprocessing': before + ['resample_reflectance', 'resample_indices', 'resample_fvc', 'prepare_mask', 'align_land_cover', 'reclassify_land_cover', 'resample_sdg', 'unify_fvc_direction', 'seasonal_means'], 'dsi': ['build_dsi'], 'trends': ['build_trends'], 'analysis': ['fig8', 'compute_npv', 'compute_cover', 'compute_dsi_response', 'compute_land', 'compute_lisa', 'compute_global_moran'], 'sensitivity': ['test_weights', 'test_trends', 'sensitivity_weights', 'sensitivity_trends'], 'plots': list(PLOT_TASKS), 'tables': ['table2', 'table_weights', 'table_trends']}

def check_inputs(source_level='raw'):
    missing = []
    families = {}

    def collect(label, folder, names):
        count = 0
        for name in names:
            try:
                raster_source(folder / str(name[0]), name[1])
                count += 1
            except FileNotFoundError:
                missing.append(str(folder / str(name[0]) / name[1]))
        families[label] = count
    monthly = lambda prefix: [(year, f'{prefix}_{year}_{month:02d}') for year in range(2001, 2023) for month in range(1, 13)]
    collect('reflectance', VRT_ROOT / 'reflectance', monthly('MCD43A4'))
    collect('sma', VRT_ROOT / 'sma', monthly('SMA'))
    if source_level in ('raw', 'indices'):
        collect('LST', VRT_ROOT / 'lst', monthly('LST'))
    if source_level in ('indices', 'normalized'):
        folder = 'rs_normalized' if source_level == 'normalized' else 'rs_indices'
        indicators = ('Albedo', 'DFI', 'LST', 'MSAVI', 'SMI', 'TGSI', 'Wet') if source_level == 'normalized' else ('Albedo', 'DFI', 'MSAVI', 'SMI', 'TGSI', 'Wet')
        for indicator in indicators:
            collect(folder + '/' + indicator, VRT_ROOT / folder / indicator, monthly(indicator))
    families['land_cover'] = 0
    for stem in [f'MCD12C1_LULC_{year}' for year in range(2001, 2022)]:
        try:
            raster_source(VRT_ROOT / 'auxiliary_sources/land_cover_native', stem)
            families['land_cover'] += 1
        except FileNotFoundError:
            missing.append('land_cover_native/' + stem)
    for product in ('FAO-WOCAT', 'JRC', 'Trends.Earth'):
        try:
            raster_source(VRT_ROOT / 'auxiliary_sources', f'TrendsEarth_SDG15.3.1_2000-2023_{product}')
            families['SDG/' + product] = 1
        except FileNotFoundError:
            missing.append('SDG/' + product)
    try:
        raster_source(VRT_ROOT / 'auxiliary_sources/study_mask_native', 'arid_region_mask')
        families['study_mask'] = 1
    except FileNotFoundError:
        missing.append('study_mask_native/arid_region_mask')
    base = study_boundary().with_suffix('').relative_to(PROJECT_ROOT).as_posix()
    static = [base + suffix for suffix in ('.shp', '.shx', '.dbf', '.prj')] + ['02_preprocessing/data/auxiliary/masks_0p1/arid_region_mask.tif', '02_preprocessing/data/auxiliary/masks_0p1/arid_region_mask_a.tif', '04_plotting/intermediate/study_area/biome_display_source.tif', '04_plotting/intermediate/publication_tables/table_layouts.json']
    if source_level in ('raw', 'indices'):
        static += ['01_upstream_reference/statistics/' + name + '_stats.xlsx' for name in ('Albedo', 'DFI', 'LST', 'MSAVI', 'SMI', 'TGSI', 'Wet')]
    for rel in static:
        if not parameter_file(rel).is_file():
            missing.append(rel)
    return {'source_level': source_level, 'families': families, 'static_inputs': len(static), 'missing': missing, 'vrt_pixel_self_containment_assumed': True}
