"""Run independent sensitivity stages with resumable completion records."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, verify_source_record, invalidate_completion_records
import os
import subprocess
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import config as c
from common import code_signature, read_json

def main():
    c.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stages = [(c.RUN_WEIGHTS, '01_weights'), (c.RUN_DSI, '02_dsi'), (c.RUN_EVALUATION, '03_evaluate'), (c.RUN_REPORT, '04_report')]
    markers = [c.stage_path(stage) for enabled, stage in stages if enabled]
    existing = [path for path in markers if path.exists()]
    try:
        if existing and os.environ.get('REPRO_RECOMPUTE_SENSITIVITY') == '1':
            raise ValueError('Explicit recomputation requested')
        for marker in existing:
            if read_json(marker)['code_signature'] != code_signature():
                raise ValueError('Sensitivity code changed')
        manifest = c.OUTPUT_ROOT / 'audit/source_manifest.json'
        if existing:
            if not manifest.exists():
                raise ValueError('Completion marker has no input manifest')
            for record in read_json(manifest):
                verify_source_record(record)
    except (ValueError, FileNotFoundError) as error:
        invalidate_completion_records(markers, str(error))
        print('Recompute weighting sensitivity: ' + str(error), flush=True)
    for enabled, stage in stages:
        if not enabled:
            continue
        marker = c.stage_path(stage)
        if marker.exists():
            info = read_json(marker)
            if info['code_signature'] != code_signature():
                raise ValueError('Code changed during execution')
            manifest = c.OUTPUT_ROOT / 'audit' / 'source_manifest.json'
            if manifest.exists():
                from pathlib import Path
                for record in read_json(manifest):
                    verify_source_record(record)
            print(f'Reuse completed {stage}', flush=True)
            continue
        print(f'START {stage}', flush=True)
        subprocess.run([sys.executable, '-B', str(c.CODE_ROOT / f'{stage}.py')], cwd=str(c.CODE_ROOT), check=True)
        print(f'DONE {stage}', flush=True)
    print(f'COMPLETE: {c.OUTPUT_ROOT}', flush=True)
if __name__ == '__main__':
    main()
