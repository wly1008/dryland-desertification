# Analysis methods

## Preprocessing

Reflectance-derived indices are normalized using fixed bounds and aligned to
a 0.1-degree grid. Categorical land cover uses nearest-neighbour resampling.
PV and NPV directions are unified for DSI. Each non-growing season spans
October of the labeled year through April of the next year.

## DSI

Annual common-valid masks use the configured set of 12 preprocessing indicators.
Valid normalized values are divided by 10000 and clipped to [0,1].

Knowledge granulation uses 15 equal-width classes (1-15; 0 is NoData).
Study-period class frequencies determine granularity, redundancy, and entropy
weights. The with-NPV group contains Albedo, SMI, TGSI, Wet, LST, PV, and NPV;
the without-NPV group contains the first six. Fixed weights sum to one within
each group. DSI is the weighted sum of continuous normalized values.

## Trends and SDG comparison

OLS slopes use two-sided nominal Student-t P values; significance is P < 0.05.
SDG products are normalized using their full valid-product ranges.

Bivariate LISA uses direct DSI slopes, row-standardized Queen contiguity,
and paired valid cells with islands excluded. Global and local Moran analyses
use 999 permutations. Local classes use per-product Benjamini-Hochberg q < 0.05.
DSI and SDG improvement directions must be considered when interpreting classes.

## NPV comparison

State differences are with-NPV minus without-NPV on the common complete-year
domain. State summaries use WGS84 ellipsoidal pixel areas; external agreement
uses equal pixel weights on its common comparison mask.

The paired bootstrap jointly resamples occupied 5-degree spatial blocks for
both DSI variants, using 1,999 replicates to estimate paired-difference intervals.
The two variants have separately normalized indicator weights.

## Cover combinations and land transitions

Physical PV, NPV, and bare-soil trends are measured in percentage points per
year, without enforced ternary closure. At P < 0.05, states are decrease=0,
not significant=1, and increase=2. Combination code is 9*PV+3*NPV+BS+1 (1-27).
Full-cover and paired cover-DSI domains have distinct masks.

Land-cover endpoints use majority classes in 2001-2003 and 2019-2021, requiring
at least two matching years. Uncertain endpoints are excluded. Transition code
is 6*(origin-1)+destination (1-36). Summaries use WGS84 areas and area-weighted
quantiles. Displayed transitions are selected by area.

## Figures and tables

Figures are exported as 400 dpi JPG/TIFF. Figure 2 and Table 1 are static
outputs in the data archive. Table 2 presents the fixed indicator weights.

## Sensitivity analyses

Weighting sensitivity recalculates study-period entropy weights with G=10,
G=15, and G=20 and compares equal weights. Each without-NPV weight vector is
renormalized. Normalization, annual masks, periods, and OLS tests are held fixed.

Trend sensitivity compares A (OLS, nominal P < 0.05) with D (Theil-Sen slope,
two-sided Mann-Kendall with ties and continuity correction, and per-domain BH
q < 0.05). D jointly changes estimation, testing, and multiplicity treatment;
the comparison does not isolate their individual effects or correct temporal
autocorrelation. D is evaluated for fixed G15 weights only.

Cover thresholds are 0.01, 0.05, and 0.10. NPV comparisons include alternative
SDG classification thresholds and paired-bootstrap block sizes of 2.5, 5,
and 10 degrees. Supplementary Tables S1-S2 summarize weighting sensitivity;
S3 summarizes the A-versus-D comparison.

Local permutation seeds are 20260821 (FAO-WOCAT), 20261821 (JRC), and
20260821 (Trends.Earth). Seeds are bound to product names and do not change
when products are selected individually. Local permutation P values use
the esda `directed` convention before BH correction.
