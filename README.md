# Dryland desertification analysis

Python code for data preprocessing, DSI construction, trend analysis, comparison
with SDG 15.3.1 products, and figure generation.

Version: 1.0.0. Code license: MIT. Fixed-parameter workbooks: CC BY 4.0.

## Contents

| Directory in `study_workflow` | Purpose |
|---|---|
| `01_upstream_reference` | Indices, normalization, VRT utilities, and resampling |
| `02_preprocessing` | Grid alignment, indicator directions, and seasonal aggregation |
| `03_analysis` | DSI, OLS trends, LISA, paired bootstrap, cover combinations, land transitions, and sensitivity analyses |
| `04_plotting` | Main figures, Table 2, and supplementary tables |
| `07_environment` | Paths, input checks, and task configuration |

The paired bootstrap uses 5-degree blocks and 1,999 resamples. Cover-change
classification uses P < 0.05. Figure 2 and Table 1 are available as finished
outputs in the data archive.

## Environment and use

Python 3.13.9; dependencies are specified in `environment.yml`.

```sh
conda env create -f environment.yml
conda activate desertification-repro
python study_workflow/run_pipeline.py --list
python study_workflow/run_pipeline.py --task all --plan
```

The last two commands list tasks and their execution order without calculation.
Tasks use source data and their upstream results as inputs. The repository
contains fixed normalization-statistics workbooks and configuration for table
captions, headers, and notes. Raw rasters and other analysis data are external
inputs.

`STUDY_WORK_ROOT` defines the data workspace for inputs, intermediate results,
and outputs; the directory requires write access. Code entry points are in
`study_workflow`. See [input requirements and directory layout](docs/EXTERNAL_INPUTS.md)
and [analysis methods](docs/ALGORITHM_NOTES.md).

The companion data archive contains derived products organized for reuse;
it does not supply every workflow input. Plotting uses installed Arial or
the configured Matplotlib fallback.

## Sensitivity analyses

Weighting scenarios are G10, G15, G20, and equal weights. Trend sensitivity
compares OLS with Theil-Sen/Mann-Kendall/BH. Additional checks vary cover
significance thresholds, SDG thresholds, and bootstrap block sizes.

```sh
python study_workflow/run_pipeline.py --task check_inputs --source-level normalized
python study_workflow/run_pipeline.py --task dsi
python study_workflow/run_pipeline.py --task trends
python study_workflow/run_pipeline.py --task analysis
python study_workflow/run_pipeline.py --task sensitivity
python study_workflow/run_pipeline.py --task plots --verify
python study_workflow/run_pipeline.py --task tables
```

These stage commands require the upstream inputs for each stage. Use `--task all`
to include preprocessing from the selected VRT input level. Existing sensitivity
outputs are reused only after code and input-content checks. Add
`--recompute-sensitivity` to recompute them explicitly.
