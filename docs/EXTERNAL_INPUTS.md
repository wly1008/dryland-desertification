# Input requirements

Use `STUDY_WORK_ROOT` for input data, intermediates, and outputs. Source code
and bundled configurations remain in `study_workflow`.

Execution requires the following inputs, depending on the selected task.

| Stage | Inputs |
|---|---|
| Indices and normalization | Reflectance/LST rasters or resolvable VRTs, percentile statistics, fixed bounds, and masks |
| 0.1-degree preprocessing | Indices, cover components, land cover, SDG products, target grid, and study boundary |
| Seasonal DSI | Direction-unified normalized indicators for October-April and validity masks |
| Trends and SDG comparison | Annual DSI, aligned reference products, boundaries, and analysis masks |
| Cover and land transitions | Component and land-cover series, DSI trends, and paired statistical intermediates |
| Figures | Derived results, figure-specific intermediates, plotting libraries, and installed fonts |

Set paths in `study_workflow/07_environment` and the selected module.
`STUDY_WORK_ROOT` controls the shared input root; code entry points remain in
the source tree. Input filenames, schemas, grids, and masks must match the module.

The companion data archive contains derived results, not every input required
by these tasks. Its topic-based directories need mapping to workflow paths.
VRT utilities require accessible source rasters; VRT files and raw source
products are not included.

## Workspace layout

```text
01_upstream_reference/vrt/           # reflectance, LST, SMA, indices, auxiliary sources
02_preprocessing/data/              # aligned inputs, masks, boundaries, seasonal means
03_analysis/                       # DSI and downstream results in stage-specific data folders
04_plotting/intermediate/           # study-area display and figure-specific inputs
05_outputs/                        # generated figures and tables
06_verification/                   # execution logs and completion records
```

The study boundary basename is `dryland_boundary_buffer_0p1_northern_hemisphere`.
Keep its SHP, SHX, DBF, PRJ, and any encoding files together under
`02_preprocessing/data/auxiliary/study_boundary`.

The seven `01_upstream_reference/statistics/*_stats.xlsx` inputs and
`04_plotting/intermediate/publication_tables/table_layouts.json` are bundled
with the code. Workspace copies take precedence. Fixed bounds are reconstructed
from these statistics, rather than re-estimated from raw rasters by the launcher.
The two fixed 0.1-degree masks and study-area display raster remain required inputs.

Run `--task check_inputs` before preprocessing. This checks the expected input
families; downstream intermediate requirements are checked by individual tasks.
Raw input means the existing reflectance/LST/SMA VRT level; pre-VRT satellite
processing, unmixing, and tile mosaicking are outside this workflow.

For relocated historical metadata, `STUDY_PATH_MAP` may point to a local JSON
list of `[old_prefix, workspace_relative_path]` pairs. It is optional for newly
prepared inputs; private path mappings are not distributed.

`STUDY_LISA_JOBS` sets local LISA workers (default: 1).

`STUDY_LISA_PRODUCTS` optionally selects comma-separated product names
(FAO-WOCAT, JRC, Trends.Earth). Selected runs update raster outputs only;
the consolidated LISA workbook is produced by a complete three-product run.

`STUDY_BOUNDARY` optionally sets a boundary path inside the workspace.
A unique legacy buffered boundary is also recognized without renaming inputs.
Fixed-bound calculation accepts both the English statistics headers and the
original seven-column statistics layout; generated workbooks use English headers.
