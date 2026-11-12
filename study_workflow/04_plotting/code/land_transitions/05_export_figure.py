"""Export land-cover transition figures."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, save_main_figure, publication_font
import json
import warnings
import numpy as np
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.font_manager import findfont
font_family = publication_font()
font_file = findfont(font_family, fallback_to_default=False)
warnings.filterwarnings('error', message='Glyph .* missing from font')
for legend in fig.legends:
    legend.get_title().set_gid('legend_title')
for axes in fig.axes:
    if axes.get_legend() is not None:
        axes.get_legend().get_title().set_gid('legend_title')
for item in fig.findobj(match=Text):
    if item.get_text().startswith(('(a)', '(b)', '(c)')):
        item.set_gid('subplot_title')
    item.set_fontfamily(font_family)
    item.set_fontweight('bold' if item.get_gid() in ('subplot_title', 'legend_title') else 'normal')
    item.set_fontstyle('normal')
for crop_iteration in range(14):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    content_boxes = []
    for axes in fig.axes:
        if axes.get_visible() and axes.axison:
            content_boxes.append(axes.get_tightbbox(renderer))
    for item in fig.findobj(match=Text):
        if item.get_visible() and item.get_text().strip():
            content_boxes.append(item.get_window_extent(renderer))
    content_boxes = [box for box in content_boxes if box is not None and np.isfinite(box.extents).all() and (box.width > 0) and (box.height > 0)]
    crop_box = Bbox.union(content_boxes).transformed(fig.dpi_scale_trans.inverted()).padded(0.035)
    if target_width_cm is None or abs(crop_box.width * 2.54 - target_width_cm) < 0.0001:
        break
    old_width, old_height = fig.get_size_inches()
    fig.set_size_inches(old_width * (target_width_cm / 2.54) / crop_box.width, old_height)
font_counts = {}
weight_counts = {'normal': 0, 'bold': 0}
text_specs = []
for item in fig.findobj(match=Text):
    if item.get_visible() and item.get_text().strip():
        point_size = round(item.get_fontsize(), 3)
        assert point_size in (6, 7, 8), (base, item.get_text(), point_size)
        role = item.get_gid() if item.get_gid() in ('subplot_title', 'legend_title', 'colorbar_title') else 'body'
        expected_weight = 'bold' if role in ('subplot_title', 'legend_title') else 'normal'
        assert item.get_fontweight() == expected_weight
        assert role != 'subplot_title' or point_size == 8
        assert role not in ('legend_title', 'colorbar_title') or point_size == 6
        font_counts[str(point_size)] = font_counts.get(str(point_size), 0) + 1
        weight_counts[expected_weight] += 1
        text_specs.append({'text': item.get_text(), 'size_pt': point_size, 'role': role, 'gid': item.get_gid(), 'anchor_display_px': item.get_transform().transform(item.get_position()).tolist(), 'horizontal_alignment': item.get_ha(), 'family': item.get_fontfamily(), 'weight': item.get_fontweight(), 'style': item.get_fontstyle(), 'resolved_font': findfont(item.get_fontproperties(), fallback_to_default=False)})
alignment_error = None
if hasattr(fig, '_lucc_alignment'):
    network_axes, heatmap_axes, network_y = fig._lucc_alignment
    for i, ypos in enumerate(network_y):
        error = abs(network_axes.transData.transform((0, ypos))[1] - heatmap_axes.transData.transform((0, i))[1])
        alignment_error = max(alignment_error or 0.0, float(error))
    assert alignment_error < 1e-06, alignment_error
save_main_figure(fig, int(__import__('re').match('Figure([0-9]+)', base)[1]), bbox_inches=crop_box, pad_inches=0, facecolor='white', pil_kwargs={'quality': 97, 'subsampling': 0})
manifest_path = PROJECT_ROOT / '06_verification/figure_export_manifest.json'
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
manifest[base] = {'schema_version': 5, 'formats': ['JPEG', 'TIFF'], 'dpi': 400, 'font_family': font_family, 'font_file': font_file, 'font_weight': 'role-based', 'font_weight_policy': {'subplot_title': 'bold', 'legend_title': 'bold', 'colorbar_title': 'normal', 'body': 'normal'}, 'font_weight_counts': weight_counts, 'font_sizes_pt_counts': font_counts, 'width_cm': round(crop_box.width * 2.54, 5), 'height_cm': round(crop_box.height * 2.54, 5), 'outer_padding_inches': 0.035, 'text_specs': text_specs, 'LUCC_shared_row_alignment_max_pixel_error': alignment_error}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Exported {base}: {crop_box.width * 2.54:.2f} × {crop_box.height * 2.54:.2f} cm; JPEG + TIFF 400 dpi.', flush=True)
