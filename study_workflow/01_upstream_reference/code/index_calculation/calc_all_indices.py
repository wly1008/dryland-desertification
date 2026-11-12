"""Calculate reflectance-derived remote-sensing indices."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, VRT_ROOT, UPSTREAM_DATA_ROOT, raster_source
import os
import queue
import threading
import numpy as np
import rasterio
from tqdm import tqdm
from rasterio.windows import Window
vrt_root = VRT_ROOT / 'reflectance'
out_root = UPSTREAM_DATA_ROOT / 'rs_indices'
INDICES = ['MSAVI', 'TGSI', 'Albedo', 'Wet', 'DFI', 'SMI']
SCALE = 0.0001
OUT_SCALE = 10000.0
NODATA = 32767
BLOCK = 3072
QUEUE_DEPTH = 2
DFI_NAME = 'DFI'

def index_out_path(name, year, month):
    d = os.path.join(out_root, name, str(year))
    return os.path.join(d, '%s_%d_%02d.tif' % (name, year, month))
tasks = [(y, m) for y in range(2001, 2023) for m in range(1, 13)]
for year, month in tqdm(tasks, desc='RS-indices', unit='month'):
    vrt_path = raster_source(vrt_root / str(year), 'MCD43A4_%d_%02d' % (year, month))
    todo = [name for name in INDICES if not os.path.exists(index_out_path(name, year, month))]
    if not todo:
        continue
    for name in todo:
        os.makedirs(os.path.dirname(index_out_path(name, year, month)), exist_ok=True)
    stop = threading.Event()
    tmp_paths = {}
    dst_map = {}
    try:
        with rasterio.open(vrt_path) as src:
            base = src.profile.copy()
            xxx = src.read(1, window=Window(0, 0, 256, 256))
            base.update(driver='GTiff', count=1, compress='lzw', tiled=True, blockxsize=256, blockysize=256, BIGTIFF='YES')
            profile_i16 = base.copy()
            profile_i16.update(dtype='int16', nodata=NODATA)
            profile_f32 = base.copy()
            profile_f32.update(dtype='float32', nodata=float('nan'))
            height = src.height
            width = src.width
            for name in todo:
                final_path = index_out_path(name, year, month)
                tmp_path = final_path + '.tmp'
                tmp_paths[name] = tmp_path
                prof = profile_f32 if name == DFI_NAME else profile_i16
                dst_map[name] = rasterio.open(tmp_path, 'w', **prof)
            read_q = queue.Queue(maxsize=QUEUE_DEPTH)
            write_q = queue.Queue(maxsize=QUEUE_DEPTH)

            def reader():
                for r in range(0, height, BLOCK):
                    if stop.is_set():
                        break
                    h = min(BLOCK, height - r)
                    w = Window(0, r, width, h)
                    bands = src.read((1, 2, 3, 4, 5, 6, 7), window=w)
                    while not stop.is_set():
                        try:
                            read_q.put((r, h, bands), timeout=0.5)
                            break
                        except queue.Full:
                            continue
                read_q.put(None)

            def writer():
                while True:
                    try:
                        item = write_q.get(timeout=0.5)
                    except queue.Empty:
                        if stop.is_set():
                            break
                        continue
                    if item is None:
                        break
                    r, h, outs = item
                    win = Window(0, r, width, h)
                    for name, arr in outs.items():
                        dst_map[name].write(arr, 1, window=win)
            t_read = threading.Thread(target=reader)
            t_write = threading.Thread(target=writer)
            t_read.start()
            t_write.start()
            try:
                while True:
                    try:
                        item = read_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    r, h, bands = item
                    b1_dn, b2_dn, b3_dn, b4_dn, b5_dn, b6_dn, b7_dn = bands
                    b1 = b1_dn.astype(np.float32) * SCALE
                    b2 = b2_dn.astype(np.float32) * SCALE
                    b3 = b3_dn.astype(np.float32) * SCALE
                    b4 = b4_dn.astype(np.float32) * SCALE
                    b5 = b5_dn.astype(np.float32) * SCALE
                    b6 = b6_dn.astype(np.float32) * SCALE
                    b7 = b7_dn.astype(np.float32) * SCALE
                    v1 = b1_dn != 0
                    v2 = b2_dn != 0
                    v3 = b3_dn != 0
                    v4 = b4_dn != 0
                    v5 = b5_dn != 0
                    v6 = b6_dn != 0
                    v7 = b7_dn != 0
                    outs = {}
                    if 'MSAVI' in todo:
                        valid = v1 & v2
                        term = 2.0 * b2 + 1.0
                        inner = term * term - 8.0 * (b2 - b1)
                        inner = np.clip(inner, 0.0, None)
                        msavi = (term - np.sqrt(inner)) / 2.0
                        out = np.full((h, width), NODATA, dtype=np.int16)
                        s = np.clip(np.round(msavi[valid] * OUT_SCALE), -32768, 32766)
                        out[valid] = s.astype(np.int16)
                        outs['MSAVI'] = out
                    if 'TGSI' in todo:
                        valid = v1 & v3 & v4
                        denom = b1 + b3 + b4
                        safe = valid & (denom != 0)
                        tgsi = np.zeros((h, width), dtype=np.float32)
                        tgsi[safe] = (b1[safe] - b3[safe]) / denom[safe]
                        out = np.full((h, width), NODATA, dtype=np.int16)
                        s = np.clip(np.round(tgsi[safe] * OUT_SCALE), -32768, 32766)
                        out[safe] = s.astype(np.int16)
                        outs['TGSI'] = out
                    if 'Albedo' in todo:
                        valid = v1 & v2 & v3 & v4 & v5 & v7
                        albedo = 0.16 * b1 + 0.291 * b2 + 0.243 * b3 + 0.116 * b4 + 0.112 * b5 + 0.081 * b7 - 0.0015
                        out = np.full((h, width), NODATA, dtype=np.int16)
                        s = np.clip(np.round(albedo[valid] * OUT_SCALE), -32768, 32766)
                        out[valid] = s.astype(np.int16)
                        outs['Albedo'] = out
                    if 'Wet' in todo:
                        valid = v1 & v2 & v3 & v4 & v5 & v6 & v7
                        wet = 0.1147 * b1 + 0.2489 * b2 + 0.2408 * b3 + 0.3132 * b4 - 0.3122 * b5 - 0.6416 * b6 - 0.5087 * b7
                        out = np.full((h, width), NODATA, dtype=np.int16)
                        s = np.clip(np.round(wet[valid] * OUT_SCALE), -32768, 32766)
                        out[valid] = s.astype(np.int16)
                        outs['Wet'] = out
                    if 'DFI' in todo:
                        valid = v1 & v2 & v6 & v7
                        out = np.full((h, width), np.nan, dtype=np.float32)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            dfi_val = (100.0 * (1.0 - b7 / b6) * (b1 / b2)).astype(np.float32)
                        out[valid] = dfi_val[valid]
                        out[~np.isfinite(out)] = np.nan
                        outs['DFI'] = out
                    if 'SMI' in todo:
                        valid = v1 & v2 & v3
                        term = 2.0 * b2 + 1.0
                        inner = term * term - 8.0 * (b2 - b1)
                        inner = np.clip(inner, 0.0, None)
                        msavi_s = (term - np.sqrt(inner)) / 2.0
                        si = np.sqrt(np.clip(b3 * b1, 0.0, None))
                        smi = np.sqrt((msavi_s - 1.0) ** 2 + si ** 2)
                        out = np.full((h, width), NODATA, dtype=np.int16)
                        s = np.clip(np.round(smi[valid] * OUT_SCALE), -32768, 32766)
                        out[valid] = s.astype(np.int16)
                        outs['SMI'] = out
                    while not stop.is_set():
                        try:
                            write_q.put((r, h, outs), timeout=0.5)
                            break
                        except queue.Full:
                            continue
                write_q.put(None)
                t_read.join()
                t_write.join()
            except BaseException:
                stop.set()
                for q in (read_q, write_q):
                    try:
                        while True:
                            q.get_nowait()
                    except queue.Empty:
                        pass
                t_read.join(timeout=5)
                t_write.join(timeout=5)
                raise
        for name in list(dst_map.keys()):
            dst_map[name].close()
        dst_map.clear()
        for name in todo:
            os.replace(tmp_paths[name], index_out_path(name, year, month))
    except BaseException:
        for name in list(dst_map.keys()):
            try:
                dst_map[name].close()
            except Exception:
                pass
        for name, tp in tmp_paths.items():
            if os.path.exists(tp):
                try:
                    os.remove(tp)
                except OSError:
                    pass
        raise
print('All indicator calculations completed.')
