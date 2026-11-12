"""Run the DSI construction stages for the selected indicator variants."""
from pathlib import Path
import argparse, os, subprocess, sys
ROOT = next((p for p in Path(__file__).resolve().parents if (p / '07_environment').is_dir()))
CODE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--variant', choices=['both', 'with_NPV', 'without_NPV'], default='both')
p.add_argument('--skip-preprocessing', action='store_true')
p.add_argument('--only-preprocessing', action='store_true')
a = p.parse_args()
preprocessing = ['02data_quality.py', '03knowledge_granulation.py', '04valid_mask.py', '05class_statistics.py']
group = ['06entropy_weight.py', '07calculate_DI.py', '08DI_quality.py']

def run(script, variant=None):
    env = os.environ.copy()
    if variant:
        env['REPRO_DSI_VARIANT'] = variant
    print('DSI stage:', script, variant or 'all indicators', flush=True)
    subprocess.run([sys.executable, '-B', str(CODE / script)], cwd=ROOT, env=env, check=True)
if not a.skip_preprocessing:
    for script in preprocessing:
        run(script)
if not a.only_preprocessing:
    for variant in ['without_NPV', 'with_NPV'] if a.variant == 'both' else [a.variant]:
        for script in group:
            run(script, variant)
print('DSI pipeline completed.', flush=True)
