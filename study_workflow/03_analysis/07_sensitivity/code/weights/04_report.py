"""Generate weighting-sensitivity tables and a numerical summary."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import csv
from pathlib import Path
import numpy as np
import config as c
from common import finish_stage, read_json, write_csv

def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

def main():
    weights = read_json(c.OUTPUT_ROOT / 'weights' / 'weights.json')
    results = read_json(c.OUTPUT_ROOT / 'all_statistics.json')
    audit = read_json(c.OUTPUT_ROOT / 'audit' / 'evaluation_checks.json')
    s1 = []
    for scenario in c.SCENARIOS:
        for variant in c.VARIANTS:
            s1.append({'scenario': scenario, 'variant': variant, **{name: weights[scenario][variant].get(name, 'NA') for name in c.INDICATORS}})
    write_csv(c.OUTPUT_ROOT / 'tables' / 'Table_S1_weights.csv', s1)
    s2 = []
    for scenario in c.SCENARIOS:
        long, npv = (results[scenario]['long_term'], results[scenario]['NPV_contrast'])
        row = {'scenario': scenario, 'recovery_area_2001_2021_percent': long['recovery_area_percent'], 'degradation_area_2001_2021_percent': long['degradation_area_percent'], 'NS_area_2001_2021_percent': long['no_significant_trend_area_percent'], 'class_agreement_with_G15_area_percent': long['class_agreement_with_G15_area_percent'], 'mean_delta_DSI_2004_2019': npv['mean_delta_DSI'], 'NPV_changed_class_area_percent': npv['changed_trend_class_area_percent'], 'direct_reversal_area_percent': npv['direct_reversal_area_percent']}
        for i, product in enumerate(c.PRODUCTS):
            ext = results[scenario]['external']
            row.update({f'{product}_delta_r': ext['delta_r'][i], f'{product}_delta_r_CI_low': ext['delta_r_ci95'][i][0], f'{product}_delta_r_CI_high': ext['delta_r_ci95'][i][1], f'{product}_delta_BA_pp': ext['delta_ba_pp'][i], f'{product}_delta_BA_CI_low_pp': ext['delta_ba_pp_ci95'][i][0], f'{product}_delta_BA_CI_high_pp': ext['delta_ba_pp_ci95'][i][1]})
        s2.append(row)
    write_csv(c.OUTPUT_ROOT / 'tables' / 'Table_S2_sensitivity_summary.csv', s2)
    report = '# Weight sensitivity results\n\n'
    report += markdown_table(list(s2[0]), [[row[k] for k in s2[0]] for row in s2])
    report += '\n\nG15 is the baseline; alternatives are G10, G20, and equal weights.\n'
    report += 'Paired differences are with NPV minus without NPV. BA differences use percentage points.\n'
    path = c.OUTPUT_ROOT / 'weight_sensitivity.md'
    path.write_text(report, encoding='utf-8')
    finish_stage('04_report', output_report=str(path), supplementary_tables=2)
    print('Weight sensitivity tables and summary generated.', flush=True)
if __name__ == '__main__':
    main()
