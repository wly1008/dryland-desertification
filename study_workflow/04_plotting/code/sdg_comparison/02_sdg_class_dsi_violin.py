"""Plot DSI distributions by SDG class (Figure 4)."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, save_main_figure
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Rectangle
SCRIPT_DIR = CODE_ROOT / '03_analysis/03_sdg_comparison/code'
WORKING_ROOT = PROJECT_ROOT / '03_analysis/03_sdg_comparison'
VERIFY_DIR = PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1'
DSI_DIR = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'
DSI_PATHS = {'2004_2019': PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_2004_2019.tif', 'p_lt_0.05_2004_2019': PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_p_lt_0.05_2004_2019.tif'}
SDG_PATHS = {'FAO-WOCAT': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_FAO-WOCAT/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif', 'JRC': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_JRC/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif', 'Trends.Earth': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_Trends.Earth/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif'}
OUTPUT_DIR = FIGURES_ROOT / 'jpg'
OUTPUT_JPG = FIGURES_ROOT / 'jpg/Fig4.jpg'
VERSION_ORDER = ['2004_2019', 'p_lt_0.05_2004_2019']
PRODUCT_ORDER = ['FAO-WOCAT', 'JRC', 'Trends.Earth']
ROW_TITLES = ['All trends', 'Significant trends\n($P < 0.05$)']
CLASS_NAMES = ['Degraded', 'Intermediate', 'Improved']
CLASS_CODES = {'Degraded': 1, 'Intermediate': 2, 'Improved': 3}
CLASS_POSITIONS = [3, 2, 1]
CLASS_COLORS = {'Degraded': '#C84F5E', 'Intermediate': '#C7CBD1', 'Improved': '#4F78A8'}
THRESHOLD_FRACTION = 0.1
X_DISPLAY_QUANTILES = (0.005, 0.995)
PROPORTION_BAR_FIX_SPLIT_AT_ZERO = False
PROPORTION_BAR_MIN_LABEL_WIDTH_PERCENT = 5.0
SAMPLE_VIOLIN = False
MAX_VIOLIN_POINTS_PER_GROUP = 50000
VIOLIN_POINTS = 160
RANDOM_SEED = 20260818
FIGURE_DPI = 400
FIGURE_WIDTH_CM = 17.0
FIGURE_HEIGHT_CM = 10.0
CM_PER_INCH = 2.54
BASE_FONT_SIZE = 7.0
AUXILIARY_FONT_SIZE = 6.0
PROPORTION_VALUE_FONT_SIZE = 6.0
PROPORTION_DIRECTION_FONT_SIZE = 6.0
CATEGORY_TICK_FONT_SIZE = 6.0
PANEL_LABEL_FONT_SIZE = 8.0
PRODUCT_TITLE_FONT_SIZE = 8.0
ROW_TITLE_FONT_SIZE = 8.0
GLOBAL_LABEL_FONT_SIZE = 7.0
MEDIAN_MARKER_AREA = 10.0
AXIS_LINE_WIDTH = 0.6
TICK_LINE_WIDTH = 0.6
BAR_SEGMENT_EDGE_WIDTH = 0.45
BAR_OUTLINE_WIDTH = 0.45
BAR_SPLIT_WIDTH = 0.6
VIOLIN_EDGE_WIDTH = 0.45
BOX_LINE_WIDTH = 0.6
MEDIAN_LINE_WIDTH = 1.0
ZERO_REFERENCE_LINE_WIDTH = 0.55
for required_path in list(DSI_PATHS.values()) + list(SDG_PATHS.values()):
    if not required_path.exists():
        raise FileNotFoundError(f'Missing required raster: {required_path}')
dsi_inputs = {}
reference_shape = None
reference_transform = None
reference_crs = None
x_quantile_limits = []
for version_name in VERSION_ORDER:
    dsi_path = DSI_PATHS[version_name]
    with rasterio.open(dsi_path) as src:
        masked = src.read(1, masked=True)
        data = np.asarray(masked.data, dtype=np.float64)
        valid = ~np.ma.getmaskarray(masked) & np.isfinite(data)
        if reference_shape is None:
            reference_shape = src.shape
            reference_transform = src.transform
            reference_crs = src.crs
        else:
            if src.shape != reference_shape:
                raise ValueError(f'Grid shape mismatch for DSI {version_name}')
            if not src.transform.almost_equals(reference_transform):
                raise ValueError(f'Grid transform mismatch for DSI {version_name}')
            if src.crs != reference_crs:
                raise ValueError(f'CRS mismatch for DSI {version_name}')
    if not np.any(valid):
        raise ValueError(f'No valid DSI pixels: {dsi_path}')
    dsi_inputs[version_name] = {'data': data, 'valid': valid}
    x_quantile_limits.append(np.quantile(data[valid] * 1000.0, X_DISPLAY_QUANTILES))
x_min = float(min((bounds[0] for bounds in x_quantile_limits)))
x_max = float(max((bounds[1] for bounds in x_quantile_limits)))
sdg_inputs = {}
for product_name in PRODUCT_ORDER:
    sdg_path = SDG_PATHS[product_name]
    with rasterio.open(sdg_path) as src:
        if src.shape != reference_shape:
            raise ValueError(f'Grid shape mismatch for {product_name}')
        if not src.transform.almost_equals(reference_transform):
            raise ValueError(f'Grid transform mismatch for {product_name}')
        if src.crs != reference_crs:
            raise ValueError(f'CRS mismatch for {product_name}')
        masked = src.read(1, masked=True)
        data = np.asarray(masked.data, dtype=np.float64)
        valid = ~np.ma.getmaskarray(masked) & np.isfinite(data)
    valid_values = data[valid]
    sdg_min = float(np.min(valid_values))
    sdg_max = float(np.max(valid_values))
    if not sdg_min < 0.0 < sdg_max:
        raise ValueError(f'{product_name} does not span negative and positive values')
    threshold_low = THRESHOLD_FRACTION * sdg_min
    threshold_high = THRESHOLD_FRACTION * sdg_max
    class_grid = np.zeros(reference_shape, dtype=np.uint8)
    class_grid[valid & (data < threshold_low)] = CLASS_CODES['Degraded']
    class_grid[valid & (data >= threshold_low) & (data <= threshold_high)] = CLASS_CODES['Intermediate']
    class_grid[valid & (data > threshold_high)] = CLASS_CODES['Improved']
    if np.count_nonzero(class_grid) != np.count_nonzero(valid):
        raise ValueError(f'SDG classes are not exhaustive for {product_name}')
    sdg_inputs[product_name] = {'valid': valid, 'class_grid': class_grid}
all_plot_data = {}
for version_name in VERSION_ORDER:
    dsi_item = dsi_inputs[version_name]
    version_data = []
    for product_name in PRODUCT_ORDER:
        sdg_item = sdg_inputs[product_name]
        paired_valid = dsi_item['valid'] & sdg_item['valid']
        x = dsi_item['data'][paired_valid] * 1000.0
        paired_classes = sdg_item['class_grid'][paired_valid]
        class_values = {}
        for class_name in CLASS_NAMES:
            values = x[paired_classes == CLASS_CODES[class_name]]
            if values.size == 0:
                raise ValueError(f'Empty class for {version_name}, {product_name}, {class_name}')
            class_values[class_name] = values
        version_data.append({'name': product_name, 'class_values': class_values})
    all_plot_data[version_name] = version_data
plt.rcParams.update({'font.family': 'Arial', 'font.size': BASE_FONT_SIZE, 'axes.linewidth': AXIS_LINE_WIDTH, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.width': TICK_LINE_WIDTH, 'ytick.major.width': TICK_LINE_WIDTH, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig = plt.figure(figsize=(FIGURE_WIDTH_CM / CM_PER_INCH, FIGURE_HEIGHT_CM / CM_PER_INCH))
layout = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.0), height_ratios=(1.0, 1.0), left=0.09, right=0.94, bottom=0.145, top=0.925, wspace=0.11, hspace=0.12)
proportion_axes = np.empty((2, 3), dtype=object)
violin_axes = np.empty((2, 3), dtype=object)
reference_x_ax = None
reference_y_ax = None
for row_index in range(2):
    for column_index in range(3):
        cell = layout[row_index, column_index].subgridspec(2, 1, height_ratios=(0.18, 1.0), hspace=0.02)
        proportion_axes[row_index, column_index] = fig.add_subplot(cell[0, 0], sharex=reference_x_ax if PROPORTION_BAR_FIX_SPLIT_AT_ZERO else None)
        violin_axes[row_index, column_index] = fig.add_subplot(cell[1, 0], sharex=reference_x_ax, sharey=reference_y_ax)
        if reference_x_ax is None:
            reference_x_ax = violin_axes[row_index, column_index]
        if reference_y_ax is None:
            reference_y_ax = violin_axes[row_index, column_index]
panel_labels = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]
for row_index, version_name in enumerate(VERSION_ORDER):
    for column_index, item in enumerate(all_plot_data[version_name]):
        ax = violin_axes[row_index, column_index]
        proportion_ax = proportion_axes[row_index, column_index]
        plot_values = []
        box_stats = []
        class_counts = []
        for class_index, class_name in enumerate(CLASS_NAMES):
            values = item['class_values'][class_name]
            class_counts.append(values.size)
            if SAMPLE_VIOLIN and values.size > MAX_VIOLIN_POINTS_PER_GROUP:
                group_rng = np.random.default_rng(RANDOM_SEED + row_index * 100 + column_index * 10 + class_index)
                indices = group_rng.choice(values.size, size=MAX_VIOLIN_POINTS_PER_GROUP, replace=False)
                plot_values.append(values[indices])
            else:
                plot_values.append(values)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            iqr = q3 - q1
            box_stats.append({'label': class_name, 'med': float(median), 'q1': float(q1), 'q3': float(q3), 'whislo': float(np.min(values[values >= q1 - 1.5 * iqr])), 'whishi': float(np.max(values[values <= q3 + 1.5 * iqr])), 'fliers': []})
        negative_counts = np.asarray([np.count_nonzero(item['class_values'][name] < 0.0) for name in CLASS_NAMES], dtype=np.float64)
        positive_counts = np.asarray([np.count_nonzero(item['class_values'][name] > 0.0) for name in CLASS_NAMES], dtype=np.float64)
        if negative_counts.sum() <= 0.0 or positive_counts.sum() <= 0.0:
            raise ValueError(f"Missing positive or negative values for {version_name}, {item['name']}")
        negative_proportions = negative_counts / negative_counts.sum() * 100.0
        positive_proportions = positive_counts / positive_counts.sum() * 100.0
        negative_total = float(negative_counts.sum())
        positive_total = float(positive_counts.sum())
        nonzero_total = negative_total + positive_total
        if PROPORTION_BAR_FIX_SPLIT_AT_ZERO:
            bar_x_min, bar_x_max = (x_min, x_max)
            negative_start, negative_width = (x_min, -x_min)
            positive_start, positive_width = (0.0, x_max)
            split_position = 0.0
        else:
            bar_x_min, bar_x_max = (0.0, 100.0)
            negative_start = 0.0
            negative_width = negative_total / nonzero_total * 100.0
            positive_start = negative_width
            positive_width = positive_total / nonzero_total * 100.0
            split_position = negative_width
        full_bar_width = bar_x_max - bar_x_min
        left_edge = negative_start
        for class_name, class_proportion in zip(CLASS_NAMES, negative_proportions):
            segment_width = negative_width * class_proportion / 100.0
            proportion_ax.barh(0.0, segment_width, left=left_edge, height=0.72, color=CLASS_COLORS[class_name], edgecolor='white', linewidth=BAR_SEGMENT_EDGE_WIDTH)
            if segment_width / full_bar_width * 100.0 >= PROPORTION_BAR_MIN_LABEL_WIDTH_PERCENT:
                proportion_ax.text(left_edge + 0.5 * segment_width, 0.0, f'{class_proportion:.0f}', ha='center', va='center', fontsize=PROPORTION_VALUE_FONT_SIZE, color='black', fontweight='normal')
            left_edge += segment_width
        right_edge = positive_start
        for class_name, class_proportion in zip(CLASS_NAMES, positive_proportions):
            segment_width = positive_width * class_proportion / 100.0
            proportion_ax.barh(0.0, segment_width, left=right_edge, height=0.72, color=CLASS_COLORS[class_name], edgecolor='white', linewidth=BAR_SEGMENT_EDGE_WIDTH)
            if segment_width / full_bar_width * 100.0 >= PROPORTION_BAR_MIN_LABEL_WIDTH_PERCENT:
                proportion_ax.text(right_edge + 0.5 * segment_width, 0.0, f'{class_proportion:.0f}', ha='center', va='center', fontsize=PROPORTION_VALUE_FONT_SIZE, color='black', fontweight='normal')
            right_edge += segment_width
        for start, width in ((negative_start, negative_width), (positive_start, positive_width)):
            proportion_ax.add_patch(Rectangle((start, -0.36), width, 0.72, fill=False, edgecolor='0.25', linewidth=BAR_OUTLINE_WIDTH))
        proportion_ax.axvline(split_position, color='0.15', linewidth=BAR_SPLIT_WIDTH, zorder=5)
        proportion_ax.text(negative_start + 0.5 * negative_width, 0.42, 'DSI trend < 0', ha='center', va='bottom', fontsize=PROPORTION_DIRECTION_FONT_SIZE, color='0.25')
        proportion_ax.text(positive_start + 0.5 * positive_width, 0.42, 'DSI trend > 0', ha='center', va='bottom', fontsize=PROPORTION_DIRECTION_FONT_SIZE, color='0.25')
        proportion_ax.set_xlim(bar_x_min, bar_x_max)
        proportion_ax.set_ylim(-0.5, 0.5)
        proportion_ax.axis('off')
        violin_parts = ax.violinplot(plot_values, positions=CLASS_POSITIONS, vert=False, widths=0.78, showmeans=False, showmedians=False, showextrema=False, points=VIOLIN_POINTS, bw_method='scott')
        for body, class_name in zip(violin_parts['bodies'], CLASS_NAMES):
            body.set_facecolor(CLASS_COLORS[class_name])
            body.set_edgecolor('0.25')
            body.set_linewidth(VIOLIN_EDGE_WIDTH)
            body.set_alpha(0.78)
        ax.bxp(box_stats, positions=CLASS_POSITIONS, vert=False, widths=0.18, showfliers=False, patch_artist=True, boxprops={'facecolor': 'white', 'edgecolor': '0.18', 'linewidth': BOX_LINE_WIDTH}, whiskerprops={'color': '0.18', 'linewidth': BOX_LINE_WIDTH}, capprops={'color': '0.18', 'linewidth': BOX_LINE_WIDTH}, medianprops={'color': 'black', 'linewidth': MEDIAN_LINE_WIDTH})
        ax.scatter([entry['med'] for entry in box_stats], CLASS_POSITIONS, s=MEDIAN_MARKER_AREA, color='black', zorder=5)
        ax.axvline(0.0, color='0.28', linewidth=ZERO_REFERENCE_LINE_WIDTH, linestyle='--', zorder=0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.5, 3.5)
        ax.set_yticks(CLASS_POSITIONS)
        ax.set_yticklabels(CLASS_NAMES)
        ax.tick_params(axis='y', labelleft=column_index == 0, labelsize=CATEGORY_TICK_FONT_SIZE)
        ax.tick_params(axis='x', labelbottom=row_index == 1, labelsize=BASE_FONT_SIZE)
        ax.grid(False)
        ax.text(0.02, 0.98, panel_labels[row_index][column_index], transform=ax.transAxes, ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold', bbox={'boxstyle': 'square,pad=0.08', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72})
        for y_position, count in zip(CLASS_POSITIONS, class_counts):
            ax.text(0.015, y_position - 0.18, f'n={count:,}', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=AUXILIARY_FONT_SIZE)
for column_index, product_name in enumerate(PRODUCT_ORDER):
    proportion_axes[0, column_index].text(0.5, 1.62, product_name, transform=proportion_axes[0, column_index].transAxes, ha='center', va='bottom', fontsize=PRODUCT_TITLE_FONT_SIZE, fontweight='bold', clip_on=False)
for row_index, row_title in enumerate(ROW_TITLES):
    violin_axes[row_index, 2].text(1.07, 0.5, row_title, transform=violin_axes[row_index, 2].transAxes, ha='center', va='center', rotation=270, fontsize=ROW_TITLE_FONT_SIZE, fontweight='normal', clip_on=False)
fig.supxlabel('Improvement  $\\leftarrow$  DSI trend ($\\times 10^{-3}$ yr$^{-1}$)  $\\rightarrow$  Degradation', x=0.5, y=0.04, fontsize=GLOBAL_LABEL_FONT_SIZE)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_main_figure(fig, 4, facecolor='white', bbox_inches='tight', pad_inches=0.03, pil_kwargs={'quality': 95, 'subsampling': 0})
plt.close(fig)
print(f'Saved: {OUTPUT_JPG}')
print(f'Shared DSI x-axis: {x_min:.6f} to {x_max:.6f}')
print('Proportion bar mode: ' + ('fixed at DSI=0' if PROPORTION_BAR_FIX_SPLIT_AT_ZERO else 'actual negative/positive ratio'))
