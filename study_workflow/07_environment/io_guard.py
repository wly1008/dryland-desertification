"""Restrict scientific file access to the configured project workspace."""
from pathlib import Path
import builtins, os, sys
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, _BUILD_ROOT

def install():
    root = PROJECT_ROOT.resolve()
    runtime = [CODE_ROOT.resolve(), Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    scientific = {'.tif', '.tiff', '.vrt', '.shp', '.shx', '.dbf', '.gpkg', '.npz', '.npy', '.xlsx', '.docx'}
    old_roots = [_BUILD_ROOT]

    def guard(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(value)).resolve()
        if path.is_relative_to(root) or any((path.is_relative_to(p) for p in runtime)):
            return
        text = str(path).replace('\\', '/').lower()
        if path.suffix.lower() in scientific or any((text == p.lower() or text.startswith(p.lower().rstrip('/') + '/') for p in old_roots)):
            raise RuntimeError('Scientific path must be inside this project: ' + str(path))

    def audit(event, args):
        if event == 'open' and args:
            guard(args[0])
    sys.addaudithook(audit)
    original_import = builtins.__import__
    wrapped = set()

    def wrap(module, name):
        key = (module.__name__, name)
        if key in wrapped or not hasattr(module, name):
            return
        original = getattr(module, name)

        def call(path, *args, **kwargs):
            guard(path)
            return original(path, *args, **kwargs)
        setattr(module, name, call)
        wrapped.add(key)

    def importer(*args, **kwargs):
        result = original_import(*args, **kwargs)
        for module, names in [('rasterio', ['open']), ('osgeo.gdal', ['Open', 'OpenEx']), ('geopandas', ['read_file'])]:
            loaded = sys.modules.get(module)
            if loaded is not None:
                for name in names:
                    wrap(loaded, name)
        return result
    builtins.__import__ = importer
