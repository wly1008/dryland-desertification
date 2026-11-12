"""Plot bivariate DSI-SDG LISA results (Figure 5)."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, study_boundary, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, save_main_figure
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator
SCRIPT_DIR = CODE_ROOT / '03_analysis/03_sdg_comparison/code'
WORKING_ROOT = PROJECT_ROOT / '03_analysis/03_sdg_comparison'
VERIFY_DIR = PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1'
SOURCE_RASTER_DIR = PROJECT_ROOT / '03_analysis/03_sdg_comparison/data/03_bivariate_moran_lisa_direct_dsi/2004_2019/rasters'
OUTPUT_DIR = FIGURES_ROOT / 'jpg'
STUDY_AREA_SHAPEFILE = study_boundary()
PRODUCTS = ['FAO-WOCAT', 'JRC', 'Trends.Earth']
PANEL_LABELS = ['(a)', '(b)', '(c)']
FIGURE_WIDTH_CM = 17.0
FIGURE_HEIGHT_CM = 10.0
BASE_FONT_SIZE = 7.0
FINAL_DPI = 400
MAP_PROJECTION = ccrs.PlateCarree()
DATA_CRS = ccrs.PlateCarree()
STUDY_AREA_OUTLINE_COLOR = '#303030'
STUDY_AREA_OUTLINE_WIDTH = 0.175
GRIDLINE_COLOR = '#777777'
GRIDLINE_WIDTH = 0.125
GRIDLINE_ALPHA = 0.3
LONGITUDE_TICK_INTERVAL = 50.0
LATITUDE_TICK_INTERVAL = 20.0
SUBPLOT_VERTICAL_SPACE = 0.005
COMPOSITION_INSET_BOUNDS = [0.022, 0.115, 0.16, 0.205]
COMPOSITION_BAR_X_START = 8.0
COMPOSITION_LABEL_TO_BAR_GAP = 3.0
COMPOSITION_BAR_HEIGHT = 0.34
COMPOSITION_MIN_LABEL_PERCENT = 8.0
COMPOSITION_TICK_FONT_SIZE = 6.0
COMPOSITION_LABEL_FONT_SIZE = 6.0
COMPOSITION_VALUE_FONT_SIZE = 5.0
PANEL_TITLE_FONT_SIZE = 8.0
SIGNIFICANCE_FONT_SIZE = 6.0
LEGEND_FONT_SIZE = 6.0
AGREEMENT_NOTE_FONT_SIZE = 7.0
BOTTOM_LEGEND_GAP_POINTS = 4.0
BOTTOM_LEGEND_HANDLE_LENGTH = 3.1
BOTTOM_LEGEND_HANDLE_HEIGHT = 1.15
SHOW_FDR_SIGNIFICANT_PROPORTION = False
SIGNIFICANCE_TEXT_FORMAT = 'FDR-significant: {value:.1%}'
SOURCE_CODES_USE_NEGATED_DSI = False
NEGATED_TO_DIRECT_DSI_CODE = {0: 0, 1: 4, 2: 3, 3: 2, 4: 1}
LISA_LABELS = {0: 'Not significant', 1: 'HH', 2: 'LL', 3: 'HL', 4: 'LH'}
LISA_COLORS = {0: '#D8D6D2', 1: '#B18B60', 2: '#657F9F', 3: '#B86670', 4: '#668674'}

def cm_to_inches(value_cm):
    return value_cm / 2.54

def product_file_stem(product_name):
    return product_name.replace('.', '_').replace('-', '_')

def read_study_area_geometries():
    required_parts = [STUDY_AREA_SHAPEFILE, STUDY_AREA_SHAPEFILE.with_suffix('.shx'), STUDY_AREA_SHAPEFILE.with_suffix('.dbf'), STUDY_AREA_SHAPEFILE.with_suffix('.prj')]
    for vector_part in required_parts:
        if not vector_part.exists():
            raise FileNotFoundError(f'Missing study-area vector component: {vector_part}')
    projection_text = STUDY_AREA_SHAPEFILE.with_suffix('.prj').read_text(encoding='utf-8').upper()
    if 'WGS_1984' not in projection_text and 'WGS 84' not in projection_text:
        raise ValueError(f'The study-area outline must use WGS84 geographic coordinates: {STUDY_AREA_SHAPEFILE}')
    geometries = list(shpreader.Reader(str(STUDY_AREA_SHAPEFILE)).geometries())
    if not geometries:
        raise ValueError(f'Study-area vector contains no geometry: {STUDY_AREA_SHAPEFILE}')
    return geometries

def read_lisa_rasters():
    if not SOURCE_RASTER_DIR.exists():
        raise FileNotFoundError(f'Missing LISA raster directory: {SOURCE_RASTER_DIR}')
    items = []
    shared_shape = None
    shared_transform = None
    shared_crs = None
    shared_bounds = None
    for product_name in PRODUCTS:
        raster_path = SOURCE_RASTER_DIR / f'{product_file_stem(product_name)}_continuous_LISA_cluster.tif'
        if not raster_path.exists():
            raise FileNotFoundError(f'Missing LISA raster: {raster_path}')
        with rasterio.open(raster_path) as source:
            grid = source.read(1)
            shape = source.shape
            transform = source.transform
            crs = source.crs
            bounds = source.bounds
            nodata = source.nodata
        if nodata != 255:
            raise ValueError(f'Unexpected NoData value in {raster_path}: {nodata}')
        if crs is None or crs.to_epsg() != 4326:
            raise ValueError(f'Expected EPSG:4326 raster: {raster_path}')
        unique_codes = set(np.unique(grid).tolist())
        if not unique_codes.issubset({0, 1, 2, 3, 4, 255}):
            raise ValueError(f'Unexpected LISA codes in {raster_path}: {unique_codes}')
        if SOURCE_CODES_USE_NEGATED_DSI:
            source_grid = grid
            grid = source_grid.copy()
            for source_code, direct_dsi_code in NEGATED_TO_DIRECT_DSI_CODE.items():
                grid[source_grid == source_code] = direct_dsi_code
        if shared_shape is None:
            shared_shape = shape
            shared_transform = transform
            shared_crs = crs
            shared_bounds = bounds
        else:
            if shape != shared_shape:
                raise ValueError(f'Raster shape mismatch: {raster_path}')
            if not transform.almost_equals(shared_transform):
                raise ValueError(f'Raster transform mismatch: {raster_path}')
            if crs != shared_crs:
                raise ValueError(f'Raster CRS mismatch: {raster_path}')
            if bounds != shared_bounds:
                raise ValueError(f'Raster bounds mismatch: {raster_path}')
        analysis_mask = grid != 255
        valid_count = int(np.count_nonzero(analysis_mask))
        if valid_count == 0:
            raise ValueError(f'Raster contains no analyzed pixels: {raster_path}')
        significant_count = int(np.count_nonzero((grid >= 1) & (grid <= 4)))
        category_counts = {code: int(np.count_nonzero(grid == code)) for code in range(5)}
        significant_relative_proportions = {code: category_counts[code] / significant_count for code in range(1, 5)}
        all_relative_proportions = {code: category_counts[code] / valid_count for code in range(5)}
        if not np.isclose(sum(significant_relative_proportions.values()), 1.0):
            raise ValueError(f'Significant composition does not sum to one: {raster_path}')
        if not np.isclose(sum(all_relative_proportions.values()), 1.0):
            raise ValueError(f'All-pixel composition does not sum to one: {raster_path}')
        items.append({'product': product_name, 'path': raster_path, 'grid': grid, 'significant_proportion': significant_count / valid_count, 'significant_relative_proportions': significant_relative_proportions, 'all_relative_proportions': all_relative_proportions})
    return (items, shared_bounds)

def draw_composition_inset(parent_axis, plot_item):
    inset_axis = parent_axis.inset_axes(COMPOSITION_INSET_BOUNDS, zorder=6)
    inset_axis.set_facecolor('none')
    inset_axis.set_xlim(0.0, COMPOSITION_BAR_X_START + 100.0)
    inset_axis.set_ylim(-0.05, 1.05)
    inset_axis.set_xticks([COMPOSITION_BAR_X_START, COMPOSITION_BAR_X_START + 50.0, COMPOSITION_BAR_X_START + 100.0])
    inset_axis.set_xticklabels(['0', '50', '100'])
    inset_axis.set_yticks([])
    inset_axis.tick_params(axis='x', which='major', labelsize=COMPOSITION_TICK_FONT_SIZE, length=1.8, width=0.225, pad=0.8, bottom=True, top=False)
    inset_axis.spines['top'].set_visible(False)
    inset_axis.spines['left'].set_visible(False)
    inset_axis.spines['right'].set_visible(False)
    inset_axis.spines['bottom'].set_color('#666666')
    inset_axis.spines['bottom'].set_linewidth(0.225)
    row_definitions = (('Sig.', 0.72, range(1, 5), plot_item['significant_relative_proportions']), ('All', 0.27, range(5), plot_item['all_relative_proportions']))
    for row_label, y_position, codes, proportions in row_definitions:
        inset_axis.text(COMPOSITION_BAR_X_START - COMPOSITION_LABEL_TO_BAR_GAP, y_position, row_label, ha='right', va='center', fontsize=COMPOSITION_LABEL_FONT_SIZE, color='black')
        left_edge = COMPOSITION_BAR_X_START
        for code in codes:
            width = proportions[code] * 100.0
            inset_axis.barh(y_position, width, left=left_edge, height=COMPOSITION_BAR_HEIGHT, color=LISA_COLORS[code], edgecolor='white', linewidth=0.175)
            if width >= COMPOSITION_MIN_LABEL_PERCENT:
                inset_axis.text(left_edge + 0.5 * width, y_position, f'{width:.0f}', ha='center', va='center', fontsize=COMPOSITION_VALUE_FONT_SIZE, color='black', fontweight='normal')
            left_edge += width
        inset_axis.add_patch(Rectangle((COMPOSITION_BAR_X_START, y_position - 0.5 * COMPOSITION_BAR_HEIGHT), 100.0, COMPOSITION_BAR_HEIGHT, facecolor='none', edgecolor='#666666', linewidth=0.225))

def add_bottom_horizontal_legend(figure, legend_handles, map_axes):
    """Distribute the six legend items across the exact width of the map frame."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    map_left = min((axis.get_position().x0 for axis in map_axes))
    map_right = max((axis.get_position().x1 for axis in map_axes))
    bottom_content_bbox = map_axes[-1].get_tightbbox(renderer).transformed(figure.transFigure.inverted())
    gap_fraction = BOTTOM_LEGEND_GAP_POINTS / (figure.get_figheight() * 72.0)
    legend_top = bottom_content_bbox.y0 - gap_fraction
    figure.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(map_left, legend_top, map_right - map_left, 0.0), bbox_transform=figure.transFigure, ncol=len(legend_handles), mode='expand', frameon=False, fontsize=LEGEND_FONT_SIZE, handlelength=BOTTOM_LEGEND_HANDLE_LENGTH, handleheight=BOTTOM_LEGEND_HANDLE_HEIGHT, handletextpad=0.45, columnspacing=0.0, borderpad=0.0, borderaxespad=0.0)

def draw_figure(plot_items, shared_bounds, study_area_geometries):
    lisa_color_list = [LISA_COLORS[code] for code in range(5)]
    lisa_cmap = ListedColormap(lisa_color_list)
    lisa_cmap.set_bad('white')
    lisa_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], lisa_cmap.N)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': BASE_FONT_SIZE, 'axes.linewidth': 0.35, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(3, 1, figsize=(cm_to_inches(FIGURE_WIDTH_CM), cm_to_inches(FIGURE_HEIGHT_CM)), subplot_kw={'projection': MAP_PROJECTION})
    for row_index, (panel_label, axis, plot_item) in enumerate(zip(PANEL_LABELS, axes, plot_items)):
        masked_grid = np.ma.masked_equal(plot_item['grid'], 255)
        axis.imshow(masked_grid, cmap=lisa_cmap, norm=lisa_norm, interpolation='nearest', extent=[shared_bounds.left, shared_bounds.right, shared_bounds.bottom, shared_bounds.top], origin='upper', transform=DATA_CRS, zorder=1)
        axis.add_geometries(study_area_geometries, crs=DATA_CRS, facecolor='none', edgecolor=STUDY_AREA_OUTLINE_COLOR, linewidth=STUDY_AREA_OUTLINE_WIDTH, zorder=3)
        axis.set_extent([shared_bounds.left, shared_bounds.right, shared_bounds.bottom, shared_bounds.top], crs=DATA_CRS)
        gridliner = axis.gridlines(crs=DATA_CRS, draw_labels=True, linewidth=GRIDLINE_WIDTH, color=GRIDLINE_COLOR, alpha=GRIDLINE_ALPHA, linestyle=':', x_inline=False, y_inline=False, zorder=2)
        gridliner.top_labels = False
        gridliner.right_labels = False
        gridliner.bottom_labels = row_index == len(axes) - 1
        gridliner.left_labels = True
        gridliner.xlocator = FixedLocator(np.arange(-200.0, 200.0 + LONGITUDE_TICK_INTERVAL, LONGITUDE_TICK_INTERVAL))
        gridliner.ylocator = FixedLocator(np.arange(-100.0, 100.0 + LATITUDE_TICK_INTERVAL, LATITUDE_TICK_INTERVAL))
        gridliner.xlabel_style = {'size': BASE_FONT_SIZE}
        gridliner.ylabel_style = {'size': BASE_FONT_SIZE}
        axis.text(0.012, 0.965, f"{panel_label} {plot_item['product']}", transform=axis.transAxes, ha='left', va='top', fontsize=PANEL_TITLE_FONT_SIZE, fontweight='bold', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'pad': 1.0}, zorder=5)
        if SHOW_FDR_SIGNIFICANT_PROPORTION:
            axis.text(0.99, 0.025, SIGNIFICANCE_TEXT_FORMAT.format(value=plot_item['significant_proportion']), transform=axis.transAxes, ha='right', va='bottom', fontsize=SIGNIFICANCE_FONT_SIZE, bbox={'facecolor': 'white', 'edgecolor': '#999999', 'linewidth': 0.2, 'alpha': 0.88, 'pad': 1.5}, zorder=5)
        if row_index == len(axes) - 1:
            axis.text(0.988, 0.035, 'HL/LH = agreement', transform=axis.transAxes, ha='right', va='bottom', fontsize=AGREEMENT_NOTE_FONT_SIZE, fontweight='normal', color='black', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1.0}, zorder=6)
        draw_composition_inset(axis, plot_item)
    legend_handles = [Patch(facecolor=LISA_COLORS[code], edgecolor='#666666', linewidth=0.3, label=LISA_LABELS[code]) for code in range(5)]
    legend_handles.append(Patch(facecolor='white', edgecolor='#666666', linewidth=0.3, label='No data'))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.98, hspace=SUBPLOT_VERTICAL_SPACE)
    add_bottom_horizontal_legend(fig, legend_handles, axes)
    return fig

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study_area_geometries = read_study_area_geometries()
    plot_items, shared_bounds = read_lisa_rasters()
    figure = draw_figure(plot_items, shared_bounds, study_area_geometries)
    output_jpg = FIGURES_ROOT / 'jpg/Fig5.jpg'
    save_main_figure(figure, 5, facecolor='white', bbox_inches='tight', pad_inches=0.02, pil_kwargs={'quality': 95, 'subsampling': 0})
    plt.close(figure)
    print(f'Saved: {output_jpg}')
    for item in plot_items:
        print(f"{item['product']}: FDR-significant {item['significant_proportion']:.1%}")
if __name__ == '__main__':
    main()
