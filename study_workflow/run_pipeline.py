"""List, plan, and run the study analysis tasks."""
from pathlib import Path
import argparse, json, os, runpy, subprocess, sys, time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '07_environment'))
from project_paths import PROJECT_ROOT
from workflow import TASKS, TASK_ARGUMENTS, STAGES, stage_tasks, check_inputs
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--list', action='store_true')
p.add_argument('--task', default='list', choices=['list', 'all', 'check_inputs', *STAGES, *TASKS])
p.add_argument('--source-level', choices=['raw', 'indices', 'normalized'], default=os.environ.get('REPRO_SOURCE_LEVEL', 'raw'), help='raw: reflectance+LST VRT; indices: index+LST VRT; normalized: normalized-index VRT')
p.add_argument('--plan', action='store_true', help='Show ordered scripts without calculation')
p.add_argument('--until', choices=STAGES, help='Stop the all workflow after this stage')
p.add_argument('--recompute-sensitivity', action='store_true', help='Recompute sensitivity results instead of reusing verified outputs')
p.add_argument('--verify', action='store_true', help='Write figures to 06_verification')
p.add_argument('--internal', help=argparse.SUPPRESS)
a, extra = p.parse_known_args()
if extra and (not a.internal):
    p.error('Unknown arguments: ' + ' '.join(extra))
os.environ['MPLBACKEND'] = 'Agg'
if a.recompute_sensitivity:
    os.environ['REPRO_RECOMPUTE_SENSITIVITY'] = '1'
os.environ['REPRO_SOURCE_LEVEL'] = a.source_level
if a.source_level == 'raw':
    os.environ['REPRO_REBUILT_INDICES'] = '1'
else:
    os.environ.pop('REPRO_REBUILT_INDICES', None)
if a.source_level in ('raw', 'indices'):
    os.environ['REPRO_NORMALIZED_ROOT'] = str(PROJECT_ROOT / '01_upstream_reference/rebuilt_data/rs_normalized')
else:
    os.environ.pop('REPRO_NORMALIZED_ROOT', None)
if a.verify:
    os.environ['REPRO_FIGURE_ROOT'] = str(PROJECT_ROOT / '06_verification/reproduced_figures_400dpi')
if a.internal:
    temporary = PROJECT_ROOT / '06_verification/runtime_temp'
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ('TMPDIR', 'TEMP', 'TMP', 'CPL_TMPDIR'):
        os.environ[key] = str(temporary)
    import tempfile
    tempfile.tempdir = str(temporary)
    script = (ROOT / a.internal).resolve()
    if not script.is_relative_to(ROOT) or not script.is_file():
        raise ValueError('Script must be inside the project')
    sys.path.insert(0, str(script.parent))
    from io_guard import install
    install()
    sys.argv = [str(script)] + (extra[1:] if extra[:1] == ['--'] else extra)
    runpy.run_path(str(script), run_name='__main__')
elif a.list or a.task == 'list':
    print('Stages:', ', '.join(STAGES))
    for name, script in TASKS.items():
        print(name, '->', script, ' '.join(TASK_ARGUMENTS.get(name, [])))
elif a.task == 'check_inputs':
    report = check_inputs(a.source_level)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(bool(report['missing']))
else:
    stages = stage_tasks(a.source_level)
    if a.task == 'all':
        selected = []
        for stage in STAGES:
            selected.extend(stages[stage])
            if stage == a.until:
                break
    elif a.task in STAGES:
        selected = stages[a.task]
    else:
        selected = [a.task]
    if a.plan:
        for number, name in enumerate(selected, 1):
            print(f"{number:02d}. {name}: {TASKS[name]} {' '.join(TASK_ARGUMENTS.get(name, []))}")
        sys.exit(0)
    if a.task in ('all', 'preprocessing'):
        report = check_inputs(a.source_level)
        if report['missing']:
            raise FileNotFoundError(json.dumps(report, ensure_ascii=False, indent=2))
    logdir = PROJECT_ROOT / '06_verification/logs'
    logdir.mkdir(parents=True, exist_ok=True)
    results = []
    for task in selected:
        start = time.time()
        command = [sys.executable, '-B', str(ROOT / 'run_pipeline.py'), '--internal', TASKS[task], '--source-level', a.source_level] + (['--verify'] if a.verify else [])
        if TASK_ARGUMENTS.get(task):
            command += ['--', *TASK_ARGUMENTS[task]]
        with (logdir / (task + '.log')).open('w', encoding='utf-8') as log:
            child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result = {'task': task, 'exit_code': child.returncode, 'seconds': round(time.time() - start, 2), 'log': str((logdir / (task + '.log')).relative_to(PROJECT_ROOT))}
        results.append(result)
        print(json.dumps(result), flush=True)
        out = PROJECT_ROOT / '06_verification' / ('publication_400dpi_runs.json' if a.task == 'plots' else a.task + '_run.json')
        out.write_text(json.dumps(results, indent=2), encoding='utf-8')
        if child.returncode:
            sys.exit(child.returncode)
