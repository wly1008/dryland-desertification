"""Plot DSI-SDG density distributions and regressions (Figure 3)."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, save_main_figure
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Patch
from scipy import stats
from scipy.ndimage import gaussian_filter1d
SCRIPT_DIR = CODE_ROOT / '03_analysis/03_sdg_comparison/code'
WORKING_ROOT = PROJECT_ROOT / '03_analysis/03_sdg_comparison'
VERIFY_DIR = PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1'
DSI_PATHS = {'2004_2019': PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_2004_2019.tif', 'p_lt_0.05_2004_2019': PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_p_lt_0.05_2004_2019.tif'}
SDG_PATHS = {'FAO-WOCAT': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_FAO-WOCAT/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif', 'JRC': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_JRC/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif', 'Trends.Earth': PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum/TrendsEarth_SDG15.3.1_2000-2023_Trends.Earth/SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif'}
OUTPUT_ROOT = FIGURES_ROOT / 'jpg'
DENSITY_BINS_X = 260
DENSITY_BINS_Y = 220
DENSITY_USE_LOG_SCALE = True
DENSITY_LOG_VMIN = 1.0
DENSITY_LOG_VMAX = 10000.0
DENSITY_LOG_TICKS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
MARGINAL_BINS_X = 180
MARGINAL_BINS_Y = 160
MARGINAL_SMOOTH_SIGMA_BINS = 2.0
TOP_MARGINAL_FREQUENCY_MAX = 6.0
RIGHT_MARGINAL_FREQUENCY_MAX = 6.0
TOP_MARGINAL_PANEL_SIZE_RATIO = 0.44
RIGHT_MARGINAL_PANEL_SIZE_RATIO = 0.38
X_DISPLAY_QUANTILES = (0.005, 0.995)
Y_DISPLAY_LIMITS = (-1.0, 1.0)
Y_AXIS_LIMITS = (-1.0, 1.0)
VERSION_COLORS = {'2004_2019': '#8C8C8C', 'p_lt_0.05_2004_2019': '#D95F5F'}
PRODUCT_COLORS = {'FAO-WOCAT': '#4C78A8', 'JRC': '#F28E2B', 'Trends.Earth': '#59A14F'}
ACTIVE_DENSITY_PALETTE = 'c1'
REVERSE_DENSITY_PALETTE = 0
DENSITY_PALETTES = {'purple_orange_reference': ('#A8736B', '#EFB285', '#FAC897', '#F1C5A8', '#E4BEBB', '#D6B7C7', '#C7B3CC', '#B6A7D2', '#9692C5', '#8086BA', '#5E71A9', '#4B6099'), 'red_yellow_blue': ('#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8', '#FFFFBF', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026'), 'red_yellow_blue_discrete_reference': ('#354BA0', '#578DC0', '#8FC2DE', '#CAE8F1', '#F3FAD4', '#FEE99E', '#FDB96B', '#F57645', '#D83022', '#A60021'), 'teal_red_reference': ('#972314', '#C13627', '#E9495F', '#FF857D', '#FECFA9', '#9AB999', '#009976', '#253338'), 'c1': ('#4B6099', '#5E71A9', '#8086BA', '#9692C5', '#FF857D', '#E9495F', '#C13627', '#972314')}
DENSITY_COLORS = DENSITY_PALETTES[ACTIVE_DENSITY_PALETTE]
if REVERSE_DENSITY_PALETTE:
    DENSITY_COLORS = tuple(reversed(DENSITY_COLORS))
DENSITY_CMAP = LinearSegmentedColormap.from_list(f'density_{ACTIVE_DENSITY_PALETTE}', DENSITY_COLORS, N=256)
PLOT_SAMPLING_ENABLED = False
PLOT_SAMPLE_SIZE = 220000
RANDOM_SEED = 20260818
FIGURE_DPI = 400
FIGURE_WIDTH_CM = 17.0
FIGURE_HEIGHT_CM = 12.0
CM_PER_INCH = 2.54
BASE_FONT_SIZE = 7.0
SMALL_FONT_SIZE = 6.0
PANEL_LABEL_SIZE = 8.0
TITLE_FONT_SIZE = 8.0
dsi_inputs = {}
x_quantile_limits = []
for version_name, dsi_path in DSI_PATHS.items():
    with rasterio.open(dsi_path) as dsi_src:
        dsi_masked = dsi_src.read(1, masked=True)
        dsi_data = np.asarray(dsi_masked.data, dtype=np.float64)
        dsi_valid = ~np.ma.getmaskarray(dsi_masked) & np.isfinite(dsi_data)
        dsi_shape = dsi_src.shape
        dsi_transform = dsi_src.transform
        dsi_crs = dsi_src.crs
    if not np.any(dsi_valid):
        raise ValueError(f'No valid DSI pixels: {dsi_path}')
    dsi_display = dsi_data[dsi_valid] * 1000.0
    version_limits = np.quantile(dsi_display, X_DISPLAY_QUANTILES)
    x_quantile_limits.append(version_limits)
    dsi_inputs[version_name] = {'path': dsi_path, 'data': dsi_data, 'valid': dsi_valid, 'shape': dsi_shape, 'transform': dsi_transform, 'crs': dsi_crs}
x_min = float(min((item[0] for item in x_quantile_limits)))
x_max = float(max((item[1] for item in x_quantile_limits)))
reference = dsi_inputs['2004_2019']
sdg_inputs = {}
for product_name, sdg_path in SDG_PATHS.items():
    with rasterio.open(sdg_path) as sdg_src:
        if sdg_src.shape != reference['shape']:
            raise ValueError(f'Grid shape mismatch for {product_name}')
        if not sdg_src.transform.almost_equals(reference['transform']):
            raise ValueError(f'Grid transform mismatch for {product_name}')
        if sdg_src.crs != reference['crs']:
            raise ValueError(f'CRS mismatch for {product_name}')
        sdg_masked = sdg_src.read(1, masked=True)
        sdg_data = np.asarray(sdg_masked.data, dtype=np.float64)
        sdg_valid = ~np.ma.getmaskarray(sdg_masked) & np.isfinite(sdg_data)
    sdg_max = float(np.max(sdg_data[sdg_valid]))
    if not np.isfinite(sdg_max) or sdg_max <= 0:
        raise ValueError(f'The valid maximum of {product_name} is not positive')
    sdg_inputs[product_name] = {'path': sdg_path, 'data': sdg_data, 'valid': sdg_valid, 'max': sdg_max}
all_plot_data = {}
all_positive_counts = []
for version_name, dsi_item in dsi_inputs.items():
    version_plot_data = []
    for product_name, sdg_item in sdg_inputs.items():
        paired_valid = dsi_item['valid'] & sdg_item['valid']
        x = dsi_item['data'][paired_valid] * 1000.0
        y = sdg_item['data'][paired_valid] / sdg_item['max']
        ols = stats.linregress(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        visible = (x >= x_min) & (x <= x_max) & (y >= Y_DISPLAY_LIMITS[0]) & (y <= Y_DISPLAY_LIMITS[1])
        x_visible = x[visible]
        y_visible = y[visible]
        density_counts, x_edges, y_edges = np.histogram2d(x_visible, y_visible, bins=(DENSITY_BINS_X, DENSITY_BINS_Y), range=((x_min, x_max), Y_DISPLAY_LIMITS))
        x_index = np.searchsorted(x_edges, x_visible, side='right') - 1
        y_index = np.searchsorted(y_edges, y_visible, side='right') - 1
        x_index = np.clip(x_index, 0, DENSITY_BINS_X - 1)
        y_index = np.clip(y_index, 0, DENSITY_BINS_Y - 1)
        point_count = density_counts[x_index, y_index]
        positive_counts = density_counts[density_counts > 0]
        all_positive_counts.append(positive_counts)
        version_plot_data.append({'name': product_name, 'x_visible': x_visible, 'y_visible': y_visible, 'count': point_count, 'n': int(x.size), 'visible_n': int(x_visible.size), 'ols': ols, 'spearman_rho': float(spearman_rho), 'spearman_p': float(spearman_p)})
    all_plot_data[version_name] = version_plot_data
all_positive_counts = np.concatenate(all_positive_counts)
density_count_min = float(np.min(all_positive_counts))
density_count_max = float(np.max(all_positive_counts))
if DENSITY_USE_LOG_SCALE:
    shared_norm = LogNorm(vmin=DENSITY_LOG_VMIN, vmax=DENSITY_LOG_VMAX)
    density_scale_label = 'log scale'
else:
    shared_norm = Normalize(vmin=density_count_min, vmax=density_count_max)
    density_scale_label = 'linear scale'
plt.rcParams.update({'font.family': 'Arial', 'font.size': BASE_FONT_SIZE, 'axes.linewidth': 0.8, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
rng = np.random.default_rng(RANDOM_SEED)
top_frequency_panel_labels = ['(a)', '(b)', '(c)']
scatter_panel_labels = [['(d)', '(e)', '(f)'], ['(h)', '(i)', '(j)']]
right_frequency_panel_labels = ['(g)', '(k)']
version_order = ['2004_2019', 'p_lt_0.05_2004_2019']
row_labels = ['All DSI trends', 'Significant DSI trends ($p < 0.05$)']
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
output_jpg = FIGURES_ROOT / 'jpg/Fig3.jpg'
fig = plt.figure(figsize=(FIGURE_WIDTH_CM / CM_PER_INCH, FIGURE_HEIGHT_CM / CM_PER_INCH))
layout = fig.add_gridspec(3, 4, width_ratios=(1.0, 1.0, 1.0, RIGHT_MARGINAL_PANEL_SIZE_RATIO), height_ratios=(TOP_MARGINAL_PANEL_SIZE_RATIO, 1.0, 1.0), left=0.122, right=0.91, bottom=0.215, top=0.94, wspace=0.08, hspace=0.12)
axes = np.empty((2, 3), dtype=object)
reference_scatter_ax = None
for row_index in range(2):
    for column_index in range(3):
        axes[row_index, column_index] = fig.add_subplot(layout[row_index + 1, column_index], sharex=reference_scatter_ax, sharey=reference_scatter_ax)
        if reference_scatter_ax is None:
            reference_scatter_ax = axes[row_index, column_index]
top_frequency_axes = [fig.add_subplot(layout[0, column_index], sharex=axes[0, column_index]) for column_index in range(3)]
right_frequency_axes = [fig.add_subplot(layout[row_index + 1, 3], sharey=axes[row_index, 0]) for row_index in range(2)]
source_legend_ax = fig.add_subplot(layout[0, 3])
source_legend_ax.axis('off')
scatter_artist = None
for row_index, version_name in enumerate(version_order):
    version_plot_data = all_plot_data[version_name]
    for column_index, item in enumerate(version_plot_data):
        ax = axes[row_index, column_index]
        panel_label = scatter_panel_labels[row_index][column_index]
        x_visible = item['x_visible']
        y_visible = item['y_visible']
        count = item['count']
        if PLOT_SAMPLING_ENABLED and x_visible.size > PLOT_SAMPLE_SIZE:
            display_index = rng.choice(x_visible.size, size=PLOT_SAMPLE_SIZE, replace=False)
        else:
            display_index = np.arange(x_visible.size)
        display_index = display_index[np.argsort(count[display_index])]
        scatter_artist = ax.scatter(x_visible[display_index], y_visible[display_index], c=count[display_index], s=1.8, cmap=DENSITY_CMAP, norm=shared_norm, linewidths=0, alpha=0.72, rasterized=True)
        line_x = np.linspace(x_min, x_max, 400)
        line_y = item['ols'].intercept + item['ols'].slope * line_x
        ax.plot(line_x, line_y, color='black', linewidth=0.8, zorder=4)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(*Y_AXIS_LIMITS)
        ax.grid(False)
        ax.tick_params(axis='x', labelbottom=row_index == 1)
        ax.tick_params(axis='y', labelleft=column_index == 0)
        ax.text(0.98, 0.98, panel_label, transform=ax.transAxes, ha='right', va='top', fontsize=PANEL_LABEL_SIZE, fontweight='bold', bbox={'boxstyle': 'square,pad=0.08', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72})
        if item['ols'].pvalue < 0.001:
            p_annotation = '$P < 0.001$'
        else:
            p_annotation = f"$P = {item['ols'].pvalue:.3f}$"
        annotation = f"$b$ = {item['ols'].slope:.3f}" + f"\n$r$ = {item['ols'].rvalue:.3f}" + f'\n{p_annotation}'
        ax.text(0.02, 0.025, annotation, transform=ax.transAxes, ha='left', va='bottom', fontsize=SMALL_FONT_SIZE, bbox={'boxstyle': 'square,pad=0.12', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72})
for column_index, top_ax in enumerate(top_frequency_axes):
    all_pixel_count = all_plot_data['2004_2019'][column_index]['n']
    for version_name in version_order:
        item = all_plot_data[version_name][column_index]
        counts, edges = np.histogram(item['x_visible'], bins=MARGINAL_BINS_X, range=(x_min, x_max))
        frequency_percent = counts / all_pixel_count * 100.0
        frequency_percent = gaussian_filter1d(frequency_percent, sigma=MARGINAL_SMOOTH_SIGMA_BINS, mode='nearest')
        centers = 0.5 * (edges[:-1] + edges[1:])
        color = VERSION_COLORS[version_name]
        top_ax.fill_between(centers, frequency_percent, color=color, alpha=0.25, linewidth=0)
        top_ax.plot(centers, frequency_percent, color=color, linewidth=1.0)
    top_ax.set_xlim(x_min, x_max)
    top_ax.set_title(all_plot_data[version_order[0]][column_index]['name'], fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=4)
    top_ax.text(0.98, 0.95, top_frequency_panel_labels[column_index], transform=top_ax.transAxes, ha='right', va='top', fontsize=PANEL_LABEL_SIZE, fontweight='bold', bbox={'boxstyle': 'square,pad=0.08', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72})
    top_ax.tick_params(axis='x', labelbottom=False, length=0)
    top_ax.tick_params(axis='y', labelsize=BASE_FONT_SIZE)
    top_ax.grid(axis='y', color='0.88', linewidth=0.45, linestyle='--')
    if column_index == 0:
        top_ax.set_ylabel('Frequency (%)', fontsize=BASE_FONT_SIZE)
    else:
        top_ax.tick_params(axis='y', labelleft=False)
for top_ax in top_frequency_axes:
    top_ax.set_ylim(0.0, TOP_MARGINAL_FREQUENCY_MAX)
    top_ax.set_yticks([0.0, 3.0, 6.0])
source_legend_labels = ['All', 'Significant']
source_legend_handles = [Patch(facecolor=VERSION_COLORS[version_name], edgecolor=VERSION_COLORS[version_name], alpha=0.45, label=source_legend_labels[version_index]) for version_index, version_name in enumerate(version_order)]
top_frequency_axes[0].legend(handles=source_legend_handles, loc='upper left', frameon=False, fontsize=SMALL_FONT_SIZE, handlelength=0.9, handletextpad=0.35, labelspacing=0.25, borderaxespad=0.25)
for row_index, right_ax in enumerate(right_frequency_axes):
    for column_index, product_name in enumerate(SDG_PATHS):
        item = all_plot_data[version_order[row_index]][column_index]
        all_pixel_count = all_plot_data['2004_2019'][column_index]['n']
        counts, edges = np.histogram(item['y_visible'], bins=MARGINAL_BINS_Y, range=Y_DISPLAY_LIMITS)
        frequency_percent = counts / all_pixel_count * 100.0
        frequency_percent = gaussian_filter1d(frequency_percent, sigma=MARGINAL_SMOOTH_SIGMA_BINS, mode='nearest')
        centers = 0.5 * (edges[:-1] + edges[1:])
        color = PRODUCT_COLORS[product_name]
        right_ax.fill_betweenx(centers, 0.0, frequency_percent, color=color, alpha=0.18, linewidth=0)
        right_ax.plot(frequency_percent, centers, color=color, linewidth=1.0)
    right_ax.set_ylim(*Y_AXIS_LIMITS)
    right_ax.tick_params(axis='y', labelleft=False, length=0)
    right_ax.tick_params(axis='x', labelsize=BASE_FONT_SIZE, labelbottom=row_index == 1)
    right_ax.grid(axis='x', color='0.88', linewidth=0.45, linestyle='--')
    if row_index == 1:
        right_ax.set_xlabel('Frequency (%)', fontsize=BASE_FONT_SIZE)
    right_title = 'All trends' if row_index == 0 else 'Significant trends\n($P < 0.05$)'
    right_ax.text(1.32, 0.5, right_title, transform=right_ax.transAxes, ha='center', va='center', rotation=270, fontsize=TITLE_FONT_SIZE, fontweight='normal', clip_on=False)
    right_ax.text(0.96, 0.97, right_frequency_panel_labels[row_index], transform=right_ax.transAxes, ha='right', va='top', fontsize=PANEL_LABEL_SIZE, fontweight='bold', bbox={'boxstyle': 'square,pad=0.08', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72})
for right_ax in right_frequency_axes:
    right_ax.set_xlim(0.0, RIGHT_MARGINAL_FREQUENCY_MAX)
    right_ax.set_xticks([0.0, 3.0, 6.0])
fig.supxlabel('Improvement  $\\leftarrow$  DSI trend ($\\times 10^{-3}$ yr$^{-1}$)  $\\rightarrow$  Degradation', x=0.46, y=0.125, fontsize=BASE_FONT_SIZE)
fig.text(0.05, 0.5, 'Degradation  $\\leftarrow$  Scaled SDG sum  $\\rightarrow$  Improvement', ha='center', va='center', rotation=90, fontsize=BASE_FONT_SIZE)
source_legend_ax.legend(handles=[Patch(facecolor=PRODUCT_COLORS[product_name], edgecolor=PRODUCT_COLORS[product_name], alpha=0.42, label=product_name) for product_name in SDG_PATHS], loc='center', frameon=False, fontsize=SMALL_FONT_SIZE, handlelength=0.8, handletextpad=0.25, labelspacing=0.25, borderaxespad=0.0)
left_three_columns_x0 = axes[0, 0].get_position().x0
left_three_columns_x1 = axes[0, 2].get_position().x1
cbar_ax = fig.add_axes([left_three_columns_x0, 0.062, left_three_columns_x1 - left_three_columns_x0, 0.018])
cbar = fig.colorbar(scatter_artist, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(axis='x', labelsize=SMALL_FONT_SIZE, length=2.5, pad=1.5)
if DENSITY_USE_LOG_SCALE:
    cbar.set_ticks(DENSITY_LOG_TICKS)
    cbar.set_ticklabels(('$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'))
    cbar.minorticks_off()
cbar.ax.set_title(f'Pixel count per bin ({density_scale_label})', fontsize=SMALL_FONT_SIZE, pad=4.0)
save_main_figure(fig, 3, facecolor='white', bbox_inches='tight', pad_inches=0.03, pil_kwargs={'quality': 95, 'subsampling': 0})
plt.close(fig)
print(f'Saved: {output_jpg}')
print(f'Common x-axis: {x_min:.6f} to {x_max:.6f}')
print(f'Common y-axis: {Y_DISPLAY_LIMITS[0]:.1f} to {Y_DISPLAY_LIMITS[1]:.1f}')
print(f'Common density scale: {shared_norm.vmin:.8g} to {shared_norm.vmax:.8g} pixels per bin ({density_scale_label})')
