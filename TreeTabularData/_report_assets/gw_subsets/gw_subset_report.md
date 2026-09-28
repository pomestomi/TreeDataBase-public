# GROUNDWATER SUBSET ANALYSIS -- GI Manuscript

## G1  Subset composition

**Subset A** -- 279 trees with non-null DTGW in 16 cities: Aachen (2); Bamberg (18); Berlin (15); Bremen (19); Dresden (27); Erfurt (3); Hagen (30); Ingolstadt (2); Leipzig (3); Lüdenscheid (2); Pappenheim (2); Pillnitz (1); Pirmasens (3); Potsdam (134); Salzburg (3); Vienna (15).

**Subset B** -- 149 trees from Potsdam and Berlin.

### SM BSC_XX11_Q12 event counts (CHANGE 1 active)

| Subset | Depth | N total (incl. y<=0) | N y>0 |
|--------|-------|----------------------|-------|
| A | 30 cm | 2,159 | 1,571 |
| A | 60 cm | 2,168 | 1,615 |
| A | 90 cm | 2,155 | 1,531 |
| B | 30 cm | 934 | 569 |
| B | 60 cm | 932 | 549 |
| B | 90 cm | 932 | 535 |

### DT BSC_00XX_Q34 event counts (CHANGE 1 + CHANGE 2 active)

| Subset | Depth | N y>0 (pre-excl.) | N excl. CHANGE 2 | N training |
|--------|-------|-------------------|------------------|------------|
| A | 30 cm | 627 | 113 | 514 |
| A | 60 cm | 705 | 115 | 590 |
| A | 90 cm | 643 | 105 | 538 |
| B | 30 cm | 277 | 64 | 213 |
| B | 60 cm | 278 | 53 | 225 |
| B | 90 cm | 239 | 53 | 186 |

**Reconciliation with manuscript event counts:** The manuscript cited 5,633 SM and 3,390 DT for subset A, and 3,495 SM and 2,188 DT for subset B. These figures were extracted before CHANGE 2 (the 5 mm post-event rain exclusion for DT) and before the dataset freeze of 2026-08-30. CHANGE 2 affects only DT models and is applied per depth independently. SM figures above are post-CHANGE-1 (rain_post24h/72h removed from predictors) and match the y>0 filtered training counts. Differences in SM counts from the cited numbers reflect dataset updates since the original extraction.

## G2  Performance per model

| Subset | Model | Depth | n | Trees | Feats (Boruta) | Train R2 | CV R2 +/- sigma | GKF R2 +/- sigma | delta(GKF-CV) |
|--------|-------|-------|---|-------|----------------|----------|-----------------|------------------|---------------|
| A | SM | 30 cm | 1,571 | 210 | 23 (23 sel.) | 0.859 | 0.570 +/- 0.030 | 0.531 +/- 0.005 | -0.038 |
| A | SM | 60 cm | 1,615 | 216 | 22 (22 sel.) | 0.846 | 0.544 +/- 0.031 | 0.505 +/- 0.022 | -0.040 |
| A | SM | 90 cm | 1,531 | 212 | 23 (23 sel.) | 0.832 | 0.498 +/- 0.017 | 0.429 +/- 0.055 | -0.069 |
| A | DT | 30 cm | 514 | 203 | 13 (13 sel.) | 0.779 | 0.418 +/- 0.111 | 0.427 +/- 0.017 | +0.009 |
| A | DT | 60 cm | 590 | 199 | 10 (10 sel.) | 0.746 | 0.356 +/- 0.018 | 0.371 +/- 0.056 | +0.015 |
| A | DT | 90 cm | 538 | 190 | 8 (8 sel.) | 0.692 | 0.283 +/- 0.030 | 0.329 +/- 0.082 | +0.046 |
| B | SM | 30 cm | 569 | 113 | 30 (30 sel.) | 0.873 | 0.604 +/- 0.031 | 0.357 +/- 0.310 | -0.247 |
| B | SM | 60 cm | 549 | 114 | 17 (17 sel.) | 0.870 | 0.593 +/- 0.021 | 0.514 +/- 0.093 | -0.079 |
| B | SM | 90 cm | 535 | 112 | 18 (18 sel.) | 0.850 | 0.534 +/- 0.036 | 0.507 +/- 0.029 | -0.028 |
| B | DT | 30 cm | 213 | 104 | 9 (9 sel.) | 0.820 | 0.531 +/- 0.110 | 0.605 +/- 0.057 | +0.074 |
| B | DT | 60 cm | 225 | 103 | 6 (6 sel.) | 0.742 | 0.392 +/- 0.088 | 0.424 +/- 0.031 | +0.033 |
| B | DT | 90 cm | 186 | 95 | 7 (7 sel.) | 0.736 | 0.252 +/- 0.020 | 0.282 +/- 0.166 | +0.030 |

## G3  Comparison to full-model baseline

| Model | Depth | Subset A CV R2 | Subset B CV R2 | Full CV R2 | delta_A | delta_B |
|-------|-------|----------------|----------------|------------|---------|--------|
| SM | 30 cm | 0.570 | 0.604 | 0.51 | +0.060 | +0.094 |
| SM | 60 cm | 0.544 | 0.593 | 0.55 | -0.006 | +0.043 |
| SM | 90 cm | 0.498 | 0.534 | 0.52 | -0.022 | +0.014 |
| DT | 30 cm | 0.418 | 0.531 | 0.45 | -0.032 | +0.081 |
| DT | 60 cm | 0.356 | 0.392 | 0.45 | -0.094 | -0.058 |
| DT | 90 cm | 0.283 | 0.252 | 0.40 | -0.117 | -0.148 |

## G4  DTGW Boruta confirmation

| Subset | Model | Depth | Status | Rank | Share % | Pearson r(feat,SHAP) |
|--------|-------|-------|--------|------|---------|---------------------|
| A | SM | 30 cm | Not selected | -- | -- | -- |
| A | SM | 60 cm | Not selected | -- | -- | -- |
| A | SM | 90 cm | Not selected | -- | -- | -- |
| A | DT | 30 cm | Not selected | -- | -- | -- |
| A | DT | 60 cm | Not selected | -- | -- | -- |
| A | DT | 90 cm | Not selected | -- | -- | -- |
| B | SM | 30 cm | Not selected | -- | -- | -- |
| B | SM | 60 cm | Not selected | -- | -- | -- |
| B | SM | 90 cm | Not selected | -- | -- | -- |
| B | DT | 30 cm | Not selected | -- | -- | -- |
| B | DT | 60 cm | Not selected | -- | -- | -- |
| B | DT | 90 cm | Not selected | -- | -- | -- |

## G5  DTGW effect size (subset A, where selected)

| Model | Depth | IQR SHAP | Effect (target units) | Direction |
|-------|-------|----------|-----------------------|----------|
*DTGW was not selected by Boruta in any subset A model; effect size cannot be quantified from SHAP. See G6 for the univariate signal.*

## G6  Univariate signal: Spearman rho (DTGW vs log-target, subset A)

| Model | Depth | n | Spearman rho | p-value |
|-------|-------|---|--------------|--------|
| SM | 30 cm | 1,571 | -0.140 | 2.50e-08 |
| SM | 60 cm | 1,615 | -0.154 | 5.55e-10 |
| SM | 90 cm | 1,531 | -0.129 | 4.10e-07 |
| DT | 30 cm | 627 | +0.089 | 2.55e-02 |
| DT | 60 cm | 705 | +0.006 | 8.78e-01 |
| DT | 90 cm | 643 | -0.030 | 4.46e-01 |

Target is log1p-transformed for both SM (delta_pct_dyn) and DT (dry_h) to match training scale. Negative rho indicates deeper groundwater -> smaller or shorter response. The manuscript previously cited rho = -0.149; values above are post-CHANGE-2 exclusion (DT) and post-y>0 filter (both).

## G7  Random-subset sanity control (n_trees = |A|, depth 60 cm)

| Model | Subset | n events | CV R2 +/- sigma | GKF R2 +/- sigma |
|-------|--------|----------|-----------------|------------------|
| SM | A (DTGW trees) | 1,615 | 0.544 +/- 0.031 | 0.505 +/- 0.022 |
| SM | Random 279 trees | 2,628 | 0.539 +/- 0.021 | 0.497 +/- 0.026 |
| DT | A (DTGW trees) | 590 | 0.356 +/- 0.018 | 0.371 +/- 0.056 |
| DT | Random 279 trees | 992 | 0.342 +/- 0.031 | 0.348 +/- 0.027 |

If subset A CV R2 >> random CV R2, it suggests that spatial or soil co-variates specific to the DTGW-covered cities drive performance rather than the DTGW signal itself. If they are comparable, the DTGW-tree subsetting does not inflate the score.

## G8  Data-quality notes

**Bremen in subset A:** Yes (expected shallowest groundwater in network).

**DTGW coverage in subset A by city:**

- Aachen: 2 trees
- Bamberg: 18 trees
- Berlin: 15 trees
- Bremen: 19 trees
- Dresden: 27 trees
- Erfurt: 3 trees
- Hagen: 30 trees
- Ingolstadt: 2 trees
- Leipzig: 3 trees
- Lüdenscheid: 2 trees
- Pappenheim: 2 trees
- Pillnitz: 1 trees
- Pirmasens: 3 trees
- Potsdam: 134 trees
- Salzburg: 3 trees
- Vienna: 15 trees

**Reference dates:** DTGW values were interpolated from groundwater isoline maps with reference dates spanning two decades. The `assessDate` field in tree_data.json records sensor deployment, not the groundwater survey date; no per-city GW reference date is stored in the current tree metadata. Potsdam values derive from the public utility network (KWB), with surveys from the 2010s-2020s. Bremen combines data from the Botanical Garden project and one company site.

**DTGW modelling:** isoline-based spatial interpolation. All values in subset A are non-null by construction; subset B (Potsdam+Berlin) contains trees both with and without DTGW -- those without are filled with the column median during training.
