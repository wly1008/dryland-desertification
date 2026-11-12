"""Resolve project paths and configure figure exports."""
from pathlib import Path
import json, os
CODE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get('STUDY_WORK_ROOT', str(CODE_ROOT))).resolve()
FIGURES_ROOT = Path(os.environ.get('REPRO_FIGURE_ROOT', str(PROJECT_ROOT / '05_outputs/figures')))
TABLES_ROOT = PROJECT_ROOT / '05_outputs/tables'
VRT_ROOT = PROJECT_ROOT / '01_upstream_reference/vrt'
UPSTREAM_DATA_ROOT = PROJECT_ROOT / '01_upstream_reference/rebuilt_data'
_MAP = json.loads(Path(os.environ['STUDY_PATH_MAP']).read_text(encoding='utf-8')) if os.environ.get('STUDY_PATH_MAP') else []
_BUILD_ROOT = PROJECT_ROOT.as_posix()

def canonical_analysis_path(path):
    """Resolve retained legacy metadata to the fixed non-growing-season mean layout."""
    path = Path(path)
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path
    parts = list(relative.parts)
    if len(parts) >= 3 and parts[0] == '03_analysis' and (parts[1] in ('01_dsi', '02_trends')) and (parts[2] == 'data'):
        redundant = {'non-growing_season', '01non-growing_season', '02non-growing_season', 'non_growing_season', '01mean', 'mean'}
        parts = parts[:3] + [x for x in parts[3:] if x not in redundant]
        return PROJECT_ROOT.joinpath(*parts)
    return path

def remap_path(value):
    value = str(value).replace(chr(92), '/')
    if value.startswith('__code__/'):
        return CODE_ROOT / value[len('__code__/'):]
    relative = Path(value)
    if not relative.is_absolute() and relative.parts and (relative.parts[0] in ('00_documentation', '01_upstream_reference', '02_preprocessing', '03_analysis', '04_plotting', '05_outputs', '06_verification', '07_environment')):
        return canonical_analysis_path(PROJECT_ROOT / relative)
    if value.lower() == _BUILD_ROOT.lower() or value.lower().startswith(_BUILD_ROOT.lower() + '/'):
        return canonical_analysis_path(PROJECT_ROOT / value[len(_BUILD_ROOT):].lstrip('/'))
    for old, new in _MAP:
        if value.lower() == old.lower() or value.lower().startswith(old.lower() + '/'):
            return canonical_analysis_path(PROJECT_ROOT / new / value[len(old):].lstrip('/'))
    return canonical_analysis_path(Path(value))
PUBLICATION_DPI = 400

def project_file(value):
    """Resolve project-relative records and retained legacy absolute records."""
    path = remap_path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def project_relative(value):
    path = project_file(value).resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return '__code__/' + path.relative_to(CODE_ROOT).as_posix()

def source_record(value):
    """Content identity survives downloads, extraction and directory relocation."""
    import hashlib
    path = project_file(value)
    status = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': project_relative(path), 'size_bytes': status.st_size, 'sha256': digest}

def verify_source_record(record):
    path = project_file(record['path'])
    if not record.get('sha256'):
        raise ValueError('Legacy completion record has no content hash; migrate it or recompute: ' + str(path))
    actual = source_record(path)
    if (actual['size_bytes'], actual['sha256']) != (record['size_bytes'], record['sha256']):
        raise ValueError('Source content changed: ' + str(path))

def invalidate_completion_records(paths, reason):
    """Keep old provenance before invalidating reusable completion markers."""
    import time
    paths = [project_file(p) for p in paths if project_file(p).exists()]
    if not paths:
        return
    dest = PROJECT_ROOT / '06_verification/cache_history' / str(time.time_ns())
    dest.mkdir(parents=True, exist_ok=False)
    records = [{'path': project_relative(p), 'record': json.loads(p.read_text(encoding='utf-8'))} for p in paths]
    (dest / 'invalidated_records.json').write_text(json.dumps({'reason': reason, 'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    for path in paths:
        path.unlink()

def raster_inputs(folder, pattern='*'):
    """Prefer physical TIFFs when both TIFF and VRT represent the same raster."""
    folder = project_file(folder)
    found = {}
    for suffix in ('.tif', '.tiff', '.vrt'):
        for path in sorted(folder.rglob(pattern + suffix)):
            found.setdefault(path.relative_to(folder).with_suffix('').as_posix(), path)
    return sorted(found.values())

def raster_source(folder, stem):
    folder = project_file(folder)
    for suffix in ('.tif', '.tiff', '.vrt'):
        path = folder / (stem + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError('Missing raster: ' + str(folder / stem) + ' (.tif/.tiff/.vrt)')

def same_source(saved, current):
    """Compare relocated provenance without treating the former location as input."""
    if not saved:
        return False
    current = project_file(current)
    if project_file(saved) == current:
        return True
    manifest = PROJECT_ROOT / '06_verification/external_vrt_manifest.json'
    if manifest.exists():
        saved_text = str(saved).replace('\\', '/').lower()
        for record in json.loads(manifest.read_text(encoding='utf-8')):
            if saved_text == record['source'].replace('\\', '/').lower() and PROJECT_ROOT / record['vrt'] == current:
                return True
    return False

def publication_font():
    from matplotlib.font_manager import findfont
    try:
        findfont('Arial', fallback_to_default=False)
        return 'Arial'
    except ValueError:
        return 'DejaVu Sans'

def figure_output_path(value):
    """Compatibility adapter for numbered main figures; other formats are rejected."""
    import re
    p = Path(value)
    match = re.match('(?:Fig|Figure)([0-9]+)(?:_|\\.|$)', p.name)
    if not match or p.suffix.lower() not in ('.jpg', '.tif'):
        raise ValueError('Publication figures must be FigN.jpg or FigN.tif')
    out = FIGURES_ROOT / ('jpg' if p.suffix.lower() == '.jpg' else 'tif') / ('Fig' + match[1] + p.suffix.lower())
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def save_main_figure(figure, number, *, pil_kwargs=None, **kwargs):
    """Render once at 400 dpi and export matching JPEG and lossless TIFF files."""
    from io import BytesIO
    from PIL import Image
    import numpy as np
    import tifffile
    number = int(number)
    if not 1 <= number <= 12:
        raise ValueError('Unknown main-text figure number')
    dpi = kwargs.pop('dpi', PUBLICATION_DPI)
    if dpi != PUBLICATION_DPI:
        raise ValueError('Publication output must be 400 dpi')
    kwargs.pop('format', None)
    kwargs.setdefault('facecolor', 'white')
    buffer = BytesIO()
    figure.savefig(buffer, format='png', dpi=PUBLICATION_DPI, **kwargs)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        rgb = rendered.convert('RGB')
    try:
        return save_main_image(rgb, number, pil_kwargs=pil_kwargs)
    finally:
        rgb.close()
        buffer.close()

def save_main_image(image, number, *, pil_kwargs=None):
    """Export an existing RGB image without resampling, using the same 400-dpi policy."""
    import numpy as np
    import tifffile
    number = int(number)
    if not 1 <= number <= 12:
        raise ValueError('Unknown main-text figure number')
    rgb = image.convert('RGB')
    jpg = FIGURES_ROOT / 'jpg' / f'Fig{number}.jpg'
    tif = FIGURES_ROOT / 'tif' / f'Fig{number}.tif'
    jpg.parent.mkdir(parents=True, exist_ok=True)
    tif.parent.mkdir(parents=True, exist_ok=True)
    options = {'quality': 95, 'subsampling': 0}
    options.update(pil_kwargs or {})
    temporary = [jpg.with_suffix('.jpg.part'), tif.with_suffix('.tif.part')]
    try:
        rgb.save(temporary[0], format='JPEG', dpi=(PUBLICATION_DPI, PUBLICATION_DPI), **options)
        tifffile.imwrite(temporary[1], np.asarray(rgb), photometric='rgb', compression='deflate', resolution=(PUBLICATION_DPI, PUBLICATION_DPI), resolutionunit='INCH', metadata=None)
        temporary[0].replace(jpg)
        temporary[1].replace(tif)
    finally:
        for p in temporary:
            if p.exists():
                p.unlink()
        rgb.close()
    print(f'Saved: {jpg}; {tif} (400 dpi)', flush=True)
    return (jpg, tif)

def parameter_file(relative):
    """Find an input parameter in the workspace or bundled configuration."""
    path = PROJECT_ROOT / relative
    return path if path.exists() else CODE_ROOT / relative

def study_boundary():
    """Resolve the configured study boundary or a unique legacy boundary file."""
    configured = os.environ.get('STUDY_BOUNDARY')
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else PROJECT_ROOT / path
    folder = PROJECT_ROOT / '02_preprocessing/data/auxiliary/study_boundary'
    standard = folder / 'dryland_boundary_buffer_0p1_northern_hemisphere.shp'
    if standard.is_file():
        return standard
    candidates = list(folder.glob('*_buffer_0.1_clip_northern_hemisphere.shp'))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError('Multiple study boundaries found; set STUDY_BOUNDARY explicitly')
    return standard
