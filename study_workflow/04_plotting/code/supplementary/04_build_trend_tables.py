"""Build Tables S3a–e directly from trend-robustness statistics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from table_utils import ROOT, Tables, read_json, PRODUCTS, METHODS, METHOD_LABELS, SIGNED4, SIGNED3, SIGNED2
source = ROOT / '03_analysis/07_sensitivity/data/presentation/trend_robustness/statistics.json'
results = read_json(source)
book = Tables('Supplementary_Table_S3_Trend_Robustness.xlsx', [source])
rows = []
for variable in ['DSI', 'PV', 'NPV', 'BS']:
    for method in METHODS:
        r = results['datasets'][f'{variable}_2001_2021']['methods'][method]
        rows.append([variable, METHOD_LABELS[method], *[r[k] for k in ['decrease_area_percent', 'no_significant_trend_area_percent', 'increase_area_percent', 'class_agreement_with_A_area_percent', 'direction_agreement_with_A_area_percent']]])
book.add('TableS3a', rows, {j: '0.00' for j in range(2, 7)}, 'S3a_Trend_classes')
rows = []
for method in METHODS:
    r = results['contrast'][method]
    rows.append([METHOD_LABELS[method], *[r[k] for k in ['mean_delta_DSI', 'changed_class_area_percent', 'direct_reversal_area_percent', 'with_NPV_class_agreement_with_A_area_percent', 'without_NPV_class_agreement_with_A_area_percent']]])
book.add('TableS3b', rows, {1: SIGNED4, 2: '0.00', 3: '0.0000', 4: '0.00', 5: '0.00'}, 'S3b_NPV_contrast')
corr = []
ba = []
for index, product in enumerate(PRODUCTS):
    for method in METHODS:
        e = results['external'][method]
        corr.append([product, METHOD_LABELS[method], *e['r'][index], e['delta_r'][index], *e['delta_r_ci95'][index]])
        ba.append([product, METHOD_LABELS[method], *e['ba_percent'][index], e['delta_ba_pp'][index], *e['delta_ba_pp_ci95'][index]])
book.add('TableS3c', corr, {2: '0.0000', 3: '0.0000', 4: SIGNED4, 5: SIGNED4, 6: SIGNED4}, 'S3c_Correlation', bold_delta=True)
book.add('TableS3d', ba, {2: '0.00', 3: '0.00', 4: SIGNED2, 5: SIGNED2, 6: SIGNED2}, 'S3d_Balanced_agreement', bold_delta=True)
rows = []
states = ['↓', 'NS', '↑']
for code in results['display_pathways']:
    for method in METHODS:
        r = results['pathways'][method][code - 1]
        definition = ' / '.join((states[r[k]] for k in ['PV_state', 'NPV_state', 'BS_state']))
        slope = r['median_DSI_slope_per_year']
        increase = r['DSI_increase_within_group_percent']
        rows.append([f'C{code:02d}', definition, METHOD_LABELS[method], r['paired_area_percent'], '—' if slope is None else slope * 1000, '—' if increase is None else increase])
book.add('TableS3e', rows, {3: '0.00', 4: SIGNED3, 5: '0.00'}, 'S3e_Cover_pathways', widths=[18, 25, 27, 24, 27, 27])
book.save()
