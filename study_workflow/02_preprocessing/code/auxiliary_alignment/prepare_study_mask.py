"""Validate study masks on the 0.1-degree reference grid."""
from pathlib import Path
import math, sys
import numpy as np
import rasterio
from rasterio.transform import Affine
ROOT = next((p for p in Path(__file__).resolve().parents if (p / '07_environment').is_dir()))
sys.path.insert(0, str(ROOT / '07_environment'))
from project_paths import VRT_ROOT, raster_source, PROJECT_ROOT
ROOT = PROJECT_ROOT

def main():
    source = raster_source(VRT_ROOT / 'auxiliary_sources/study_mask_native', 'arid_region_mask')
    with rasterio.open(source) as src:
        gt = src.transform
        if gt.b or gt.d or gt.a <= 0 or (gt.e >= 0):
            raise ValueError('North-up source grid required')
        width = math.ceil(src.width * gt.a / 0.1 - 1e-10)
        height = math.ceil(src.height * -gt.e / 0.1 - 1e-10)
        transform = Affine(0.1, 0, gt.c, 0, -0.1, gt.f)
        crs = src.crs
    for name in ('arid_region_mask.tif', 'arid_region_mask_a.tif'):
        path = ROOT / '02_preprocessing/data/auxiliary/masks_0p1' / name
        if not path.is_file():
            raise FileNotFoundError('Required fixed auxiliary mask: ' + str(path))
        with rasterio.open(path) as mask:
            if mask.shape != (height, width) or mask.transform != transform or mask.crs != crs or (mask.count != 1) or (mask.nodata != 0):
                raise ValueError('Fixed mask does not match the VRT-derived grid: ' + str(path))
            values = mask.read(1)
            if not set(np.unique(values)).issubset({0, 1}):
                raise ValueError('Mask must contain only 0 and 1')
        print('Verified fixed auxiliary mask:', path, flush=True)
    print('The supplied masks define the established analysis domain; they are not regenerated from a different boundary or resampling rule.', flush=True)
if __name__ == '__main__':
    main()
