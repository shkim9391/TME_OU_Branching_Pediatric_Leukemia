# TME OU-Branching Pediatric Leukemia

Code and reproducible analysis for tumor microenvironment-modulated Ornstein-Uhlenbeck and OU-Branching modeling of pediatric leukemia evolution using single-cell Pediatric Cancer Atlas data.

This repository supports a workflow that extracts patient-level tumor microenvironment features from ScPCA single-cell RNA-seq datasets, defines ecological contexts from immune/TME composition, calibrates hierarchical OU models, compares baseline stochastic models, and generates manuscript figures and supplementary figures.

## Overview

Pediatric leukemias often show clinically meaningful heterogeneity that is not fully explained by recurrent coding mutations alone. This project models leukemia evolution as a microenvironment-modulated stochastic process, where patient-level tumor microenvironment composition informs latent evolutionary regimes.

The final workflow uses revised ecological contexts E1-E4 derived from single-cell TME composition:

| Context | Interpretation | Cohort size |
|---|---:|---:|
| E1 | Typical / low-deviation context | 88 |
| E2 | T / unknown-enriched context | 2 |
| E3 | Unknown-enriched context | 7 |
| E4 | B / myeloid-enriched context | 3 |

For longitudinal OU modeling, only contexts with longitudinal support contribute directly to transition-based inference:

| Context | Longitudinally supported participants | Transitions |
|---|---:|---:|
| E1 | 32 | 422 |
| E2 | 0 | 0 |
| E3 | 3 | 12 |
| E4 | 0 | 0 |

## Data Sources

The workflow uses public single-cell pediatric leukemia datasets from the Single-cell Pediatric Cancer Atlas:

- `SCPCP000022`: diverse pediatric leukemia single-cell RNA-seq cohort.
- `SCPCP000008`: pediatric acute lymphoblastic leukemia single-cell RNA-seq cohort.

Raw `.h5ad` files and metadata should be downloaded from the ScPCA Portal and placed under `data/raw/`. Large raw single-cell files are not redistributed in this repository.

## Workflow Summary

### 1. Single-cell TME feature extraction

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scpc000022.py` | SCPCP000022 `.h5ad` and metadata | `scpcp22_sample_TME_features.csv` |
| 2 | `scpc000022_clean.py` | SCPCP000022 TME features | `scpcp22_sample_TME_features_broad.csv` |
| 3 | `scpc00008_clean.py` | SCPCP000008 `.h5ad` and metadata | `scpcp8_sample_TME_features_raw.csv`, `scpcp8_sample_TME_features_broad.csv` |
| 4 | `scpc00008_stream.py` | SCPCP000008 `.h5ad` and metadata | `scpcp8_sample_TME_features_broad.csv` |

`scpc00008_stream.py` is the preferred memory-safe implementation for SCPCP000008.

### 2. Cohort merging and patient-level design matrices

| Step | Script | Main output |
|---:|---|---|
| 5 | `merge_scpcp_TME.py` | `scpcp_combined_sample_TME_features_broad.csv` |
| 6 | `scpcp_combined_sample_TME_feature_broad.py` | `scpcp_combined_participant_TME_features_broad.csv` |
| 7 | `scpcp_combined_sample_TME_features_modelready.py` | model-ready combined TME table |
| 8-13 | `make_E_*` scripts | sample-level and patient-level design matrices |
| 14 | `pm_model_ou_branching_model.py` | full PyMC model attempt; high memory demand |
| 15 | `pm_model_ou_branching_minimal.py` | minimal PyMC model; working implementation |

Key design matrix outputs include:

- `E_sample_simple_z.csv`
- `E_sample_simple_z.npy`
- `Ep_baseline_z.csv`
- `Ep_baseline_z.npy`
- `Ep_baseline_rowmeta.csv`
- `Ep_all_z.csv`
- `Ep_all_z.npy`

### 3. Patient mapping and covariate construction

| Step | Script | Main output |
|---:|---|---|
| 16 | `ou_branching_TME_block.py` | `patient_master_table.csv`, `patient_id_mapping.csv` |
| 17 | `merge_TME_with_patients.py` | merged TME and longitudinal patient table |
| 18 | `build_covariates.py` | `covariate_matrix.csv` |
| 19 | `generate_patient_immune_ecotypes.py` | `patient_immune_ecotypes.csv` |

The patient ID mapping file was manually completed after automated matching.

### 4. Exploratory interpretation

Exploratory analysis includes PCA, local UMAP, diagnosis/subdiagnosis coloring, outlier detection, immune/ecological context interpretation, and radar plots for representative outlier patients.

Representative outputs include:

- `PCA of Covariates (2D).png`
- `Local UMAP of Covariates (2D).png`
- `PCA Colored by Diagnosis.png`
- `Local UMAP Colored by Diagnosis.png`
- `PCA Colored by Subdiagnosis.png`
- `Local UMAP Colored by Subdiagnosis.png`
- `Top 10 PCA Outliers.png`
- `Immune ecotype composition (k = 4).png`
- `Mean TME Composition per Immune Ecotype.png`
- `Summary Four Immune Ecotypes.png`
- `UMAP Colored by Immune Ecotype.png`

### 5. Final ecological-context model

The final manuscript model uses the revised ecological-context assignment from Figure 3.

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 20 | `fig3_full_revised.py` | `covariate_matrix.csv`, `patient_master_table.csv` | `patient_ecological_context_assignments.csv`, `ecological_context_master_table.csv`, `ecological_context_color_key.json` |
| 21 | `ou_revised_context_calibration.py` | ecological contexts, `kmt2a_longitudinal_clean.xlsx` | `ou_revised_ecological_context_idata_ppc.nc` |
| 22 | `check_longitudinal_support.py` | ecological contexts, longitudinal data | `longitudinal_support_revised_context.csv` |
| 23 | `fig4_revised.py` | posterior model output and context files | Figure 4 and parameter summaries |

Final calibration diagnostics:

- Total divergences: 0
- R-hat: approximately 1.000-1.001
- Effective sample sizes: generally high

The earlier `ou_ecotype_ou_branching_calibration.py` script is retained as a legacy immune-ecotype-based model. The revised ecological-context model is the final workflow used for Figure 3, Figure 4, Figure 5, and Supplementary Figures S4-S5.

## Main Figures

| Figure | Script | Main outputs |
|---|---|---|
| Figure 1 | `fig1_revised_v2.py` | `Figure1_revised_v2.png`, `Figure1_revised_v2_600dpi.tiff` |
| Figure 2 | `fig2_full_revised.py` | `Figure2_composite_revised.png`, `Figure2_composite_revised.tiff` |
| Figure 3 | `fig3_full_revised.py` | `Figure3_candidate_ecological_contexts.png`, `.tiff`, context tables |
| Figure 4 | `fig4_revised.py` | `Figure4_OU_parameter_summaries_by_context.png`, `.tiff`, parameter summary CSV |
| Figure 5 | `fig5_ablations_baselines_revised.py` | `Figure5_model_comparison_revised.png`, `.tiff`, `.pdf`, model comparison CSV |

## Baseline and Ablation Models

Figure 5 compares the final context-aware OU model against baseline or ablation models.

| Script | Output |
|---|---|
| `fig5_model_file_check.py` | `Figure5_model_file_inventory.csv` |
| `fit_revised_baseline_models.py` | OU-only, shuffled-context, random-walk, and static-context model traces |
| `fig5_ablations_baselines_revised.py` | `model_comparison_revised_context.csv`, Figure 5 |

Model audit summary:

- `n_obs = 434`
- same observed response vector as context-aware OU model: `True`
- log likelihood available: `True`
- posterior predictive samples available: `True`

## Supplementary Figures

| Supplement | Script | Main outputs |
|---|---|---|
| SuppFigS1 | `generate_suppfigS1_cohort_structure_context_distribution.py` | cohort/context distribution figure and summary table |
| SuppFigS2 | `generate_suppfigS2_context_discovery_tme_profiles.py` | context discovery profiles and prototype tables |
| SuppFigS3 | `generate_suppfigS3_mcmc_diagnostics.py` | MCMC diagnostics figure and summary CSV |
| SuppFigS4 | `generate_suppfigS4_posterior_mu_theta_by_context.py` | posterior mu/theta summaries by context |
| SuppFigS5 | `generate_suppfigS5_joint_mu_theta_by_context.py` | joint posterior mu-theta summaries |
| SuppFigS6 | `generate_suppfigS6_posterior_predictive_calibration_trajectories.py` | posterior predictive calibration trajectories |

## Suggested Execution Order

```bash
# 1. Extract and clean ScPCA TME features
python scripts/preprocessing/scpc000022.py
python scripts/preprocessing/scpc000022_clean.py
python scripts/preprocessing/scpc00008_stream.py

# 2. Merge cohorts and generate design matrices
python scripts/preprocessing/merge_scpcp_TME.py
python scripts/preprocessing/scpcp_combined_sample_TME_feature_broad.py
python scripts/design_matrices/make_E_sample_simple.py
python scripts/design_matrices/make_E_patient_simple.py

# 3. Build patient-level covariates
python scripts/modeling/ou_branching_TME_block.py
python scripts/modeling/merge_TME_with_patients.py
python scripts/modeling/build_covariates.py

# 4. Generate ecological contexts
python scripts/figure_generation/fig3_full_revised.py
python scripts/figure_generation/fig3_safetry_check.py

# 5. Calibrate final ecological-context OU model
python scripts/modeling/ou_revised_context_calibration.py

# 6. Check longitudinal support and generate final figures
python scripts/figure_generation/check_longitudinal_support.py
python scripts/figure_generation/fig1_revised_v2.py
python scripts/figure_generation/fig2_full_revised.py
python scripts/figure_generation/fig4_revised.py

# 7. Fit baselines and generate Figure 5
python scripts/baselines/fig5_model_file_check.py
python scripts/baselines/fit_revised_baseline_models.py
python scripts/figure_generation/fig5_ablations_baselines_revised.py

# 8. Generate supplementary figures
python scripts/supplementary/generate_suppfigS1_cohort_structure_context_distribution.py
python scripts/supplementary/generate_suppfigS2_context_discovery_tme_profiles.py
python scripts/supplementary/generate_suppfigS3_mcmc_diagnostics.py
python scripts/supplementary/generate_suppfigS4_posterior_mu_theta_by_context.py
python scripts/supplementary/generate_suppfigS5_joint_mu_theta_by_context.py
python scripts/supplementary/generate_suppfigS6_posterior_predictive_calibration_trajectories.py

Software Requirements

The workflow was developed in Python and uses common scientific-computing and Bayesian-modeling libraries, including:

* numpy
* pandas
* scipy
* scikit-learn
* matplotlib
* seaborn
* anndata
* scanpy
* umap-learn
* pymc
* arviz
* xarray
* netcdf4
* openpyxl

Reproducibility Notes

* Raw ScPCA files are not included because of file size and data-distribution constraints.
* Processed tables and model outputs are organized to reproduce the manuscript figures.
* The final model is ou_revised_context_calibration.py, which uses revised ecological contexts E1-E4.
* ou_ecotype_ou_branching_calibration.py is retained for comparison with the earlier immune-ecotype formulation.
* Figure 5 baseline models should be regenerated before final model comparison if any upstream context assignments or longitudinal data tables change.

Citation

If you use this repository, please cite the associated manuscript once available.

Author

Seung-Hwan Kim, PhD
Associate Professor and Program Director of Biology
Fisher College
Visiting Scholar, Department of Pediatric Oncology, Dana-Farber Cancer Institute

DOI
10.5281/zenodo.19619392

[![DOI](https://zenodo.org/badge/1170246489.svg)](https://doi.org/10.5281/zenodo.19619391)
