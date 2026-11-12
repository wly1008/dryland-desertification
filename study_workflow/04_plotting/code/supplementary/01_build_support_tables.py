"""Build Tables S1 and S2a–d directly from weighting-sensitivity results."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from table_utils import ROOT, Tables, read_json, fixed_weights, INDICATORS, SCENARIOS, LABELS, PRODUCTS, SIGNED4, SIGNED2
source = ROOT / '03_analysis/07_sensitivity/data/weights'
statistics = source / 'all_statistics.json'
weight_file = source / 'weights/weights.json'
results = read_json(statistics)
weights = read_json(weight_file)
main, main_sources = fixed_weights()
for variant in main:
    for indicator, value in main[variant].items():
        assert abs(value - weights['entropy_G15'][variant][indicator]) < 1e-12
book = Tables('Supplementary_Tables_S1_S2.xlsx', [statistics, weight_file, *main_sources])
rows = []
for scenario in SCENARIOS:
    for variant, label in [('with_NPV', 'With NPV'), ('without_NPV', 'Without NPV')]:
        values = weights[scenario][variant]
        assert abs(sum(values.values()) - 1) < 1e-12
        rows.append([LABELS[scenario], label, *[values.get(i, '—') for i in INDICATORS]])
book.add('TableS1', rows, {j: '0.0000' for j in range(2, 9)}, 'S1_Weights', widths=[24, 24, *[14] * 7])
rows = []
for scenario in SCENARIOS:
    r = results[scenario]['long_term']
    rows.append([LABELS[scenario], r['recovery_area_percent'], r['no_significant_trend_area_percent'], r['degradation_area_percent'], r['class_agreement_with_G15_area_percent']])
book.add('TableS2a', rows, {j: '0.00' for j in range(1, 5)}, 'S2a_Long_term')
rows = []
for scenario in SCENARIOS:
    r = results[scenario]['NPV_contrast']
    rows.append([LABELS[scenario], r['mean_delta_DSI'], r['state_increase_area_percent'], r['changed_trend_class_area_percent'], r['direct_reversal_area_percent']])
book.add('TableS2b', rows, {1: SIGNED4, 2: '0.00', 3: '0.00', 4: '0.0000'}, 'S2b_NPV_effects')
corr = []
ba = []
for index, product in enumerate(PRODUCTS):
    for scenario in SCENARIOS:
        e = results[scenario]['external']
        corr.append([product, LABELS[scenario], *e['r'][index], e['delta_r'][index], *e['delta_r_ci95'][index]])
        ba.append([product, LABELS[scenario], *e['ba_percent'][index], e['delta_ba_pp'][index], *e['delta_ba_pp_ci95'][index]])
book.add('TableS2c', corr, {2: '0.0000', 3: '0.0000', 4: SIGNED4, 5: SIGNED4, 6: SIGNED4}, 'S2c_Correlation', bold_delta=True)
book.add('TableS2d', ba, {2: '0.00', 3: '0.00', 4: SIGNED2, 5: SIGNED2, 6: SIGNED2}, 'S2d_Balanced_agreement', bold_delta=True)
book.save()
