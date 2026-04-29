# TME_OU_Branching_Pediatric_Leukemia

Code and reproducible analysis for TME-modulated OU / OU–Branching modeling of pediatric leukemia evolution using ScPCA single-cell tumor microenvironment (TME) composition and immune ecotypes. The repository includes preprocessing, sample-level TME feature construction, immune ecotype discovery, Bayesian model fitting, posterior summaries, posterior predictive checks, ablation analyses, and figure generation.

**Associated preprint**  
Seung-Hwan Kim. *Single-cell immune ecotypes shape microenvironment-modulated evolutionary dynamics in pediatric leukemia*. Research Square, 03 March 2026, Version 1.  
DOI: `10.21203/rs.3.rs-9012769/v1`

## Overview

This repository implements a reproducible pipeline for studying how pediatric leukemia evolutionary dynamics are modulated by the tumor microenvironment. The central modeling idea is that patient-level evolutionary trajectories can be described by Ornstein–Uhlenbeck (OU) dynamics, optionally extended with branching structure, while allowing the TME to shift evolutionary optima and stabilizing-selection strength through immune ecotypes and TME covariates.

At a high level, the workflow:

1. preprocesses ScPCA single-cell datasets,
2. aggregates single-cell annotations into sample-level TME composition features,
3. builds patient/timepoint-level covariates,
4. identifies immune ecotypes from standardized TME composition,
5. fits ecotype-modulated OU / OU–Branching models,
6. summarizes posterior estimates for drift and stabilizing-selection parameters,
7. runs posterior predictive checks and ablation baselines, and
8. generates the main manuscript figures.

This repository is intended to support reproducibility of the manuscript analyses and figures, not to serve as a general-purpose software package.

## Biological and modeling focus

The study uses pediatric leukemia single-cell cohorts from the ScPCA resource to test whether immune ecotypes and TME composition are associated with distinct evolutionary regimes. In the model:

- **OU drift / optimum terms** capture directional pull toward latent evolutionary states,
- **OU stabilizing-selection parameters** quantify how strongly trajectories revert toward those states,
- **branching structure** allows divergence of trajectories across disease progression or state transitions,
- **immune ecotypes** provide discrete microenvironmental regimes, and
- **continuous TME covariates** provide patient-level modulation of model parameters.

The resulting workflow emphasizes interpretable, uncertainty-aware inference rather than black-box prediction.

## Repository structure

```text
repo_root/
  data/
    Supplementary_Data_3_ecotype_posterior_sum...
    Supplementary_Data_4_patient_posterior_summ...
    covariate_matrix.csv
    kmt2a_longitudinal_clean.xlsx
    patient_master_table.csv

  scripts/
    SFig1.py
    SFig2.py
    build_covariates.py
    fig1_full.py
    fig2_full.py
    fig3_full.py
    fig4_full.py
    fig5_ablations_baselines.py
    fit_fig5_models.py
    generate_patient_immune_ecotypes.py
    make_SuppFig3_ppc_y_obs.py
    make_SuppFig4_ou_trajectories_by_ecotype_lo...
    make_fig6_layout_row1row2row3.py
    make_fig6_main_and_SI_composites.py
    merge_scpcp_TME.py
    ou_ecotype_ou_branching_calibration.py
    plot_mu_theta_ecotype_violin.py
    plot_mu_theta_ecotype_with_patients.py
    plot_mu_theta_scatter_with_centroids.py
    scpc000022.py
    scpc000022_clean.py
    scpc00008_clean.py
    scpc00008_stream.py
    scpcp_combined_sample_TME_feature_broad.py
    summary_mu_theta_ecotype.py

 Figures/
    Figure1_clean.png
    Figure2_composite.png
	Figure3_immune_ecotypes.png
	Figure4_OU_dynamics_by_ecotype_fixedE.png
	Fig5_ablations_baselines.png
	Fig6_composite.png
	Fig7_k_sensitivity.png

  README.md

## Directory descriptions

data/

Contains processed and manuscript-facing input tables used in the ecotype-modulated OU / OU-Branching analysis, including:
	•	cleaned longitudinal clinical data
	•	patient-level master tables
	•	covariate matrices
	•	posterior summary tables for ecotype-level and patient-level results

scripts/

Contains the main preprocessing, feature-construction, model-fitting, posterior summarization, and figure-generation scripts, including:
	•	single-cell preprocessing scripts for the ScPCA cohorts
	•	TME merging and covariate-construction scripts
	•	immune ecotype generation
	•	ecotype-modulated OU / OU-Branching calibration
	•	posterior summary and visualization scripts
	•	main figure and supplementary figure scripts

 Figures/
    Figure1_clean.png
    Figure2_composite.png
	Figure3_immune_ecotypes.png
	Figure4_OU_dynamics_by_ecotype_fixedE.png
	Fig5_ablations_baselines.png
	Fig6_composite.png
	Fig7_k_sensitivity.png

README.md

## Data inputs

This repository expects the following major input sources:

1. ScPCA single-cell datasets

Two pediatric leukemia single-cell resources are used as the basis for TME feature extraction and ecotype construction:
	•	SCPCP000022
	•	SCPCP000008

The expected files include:
	•	.h5ad AnnData objects containing expression matrices and cell-level metadata
	•	single_cell_metadata.tsv files containing cell annotations and related metadata

2. Clinical longitudinal data
	•	data/clinical/kmt2a_longitudinal_clean.xlsx

This file provides the cleaned longitudinal structure needed to link patient-level evolution, disease phase, and model covariates.

Expected outputs

The pipeline produces several classes of outputs:

TME feature outputs

Written to results/tme_features/, these typically include:
	•	per-sample cell-type fractions,
	•	broad TME composition summaries,
	•	patient/timepoint aggregated microenvironment covariates.

Combined cohort tables

Written to results/combined/, these may include:
	•	harmonized tables combining SCPCP000008 and SCPCP000022,
	•	merged patient-level TME features,
	•	cross-cohort summary tables used for downstream design-matrix construction.

Matrices and covariates

Written to results/matrices/, these typically include:
	•	standardized covariate matrices,
	•	ecotype input matrices,
	•	patient-by-feature model tables.

Ecotype outputs

Written to results/ecotypes/, these may include:
	•	immune ecotype assignments,
	•	ecotype centroids,
	•	cluster summaries,
	•	patient/ecotype mapping tables.

Model outputs

Written to results/models/, these may include:
	•	posterior samples or serialized fit objects,
	•	posterior summaries for drift and stabilizing-selection parameters,
	•	patient-level and ecotype-level parameter estimates,
	•	ablation and baseline comparison outputs.

Posterior predictive checks

Written to results/ppc/, these may include:
	•	posterior predictive summaries,
	•	observed-versus-simulated comparisons,
	•	uncertainty calibration outputs,
	•	model-fit diagnostics.

Analysis workflow

The typical workflow proceeds in the following stages.

Stage 1. Preprocess cohort SCPCP000022

python scripts/scpc000022.py
python scripts/scpc000022_clean.py

These scripts prepare and clean the SCPCP000022 cohort data for downstream use. Typical operations may include:
	•	loading the merged .h5ad,
	•	harmonizing observation metadata,
	•	filtering invalid or incomplete entries,
	•	standardizing sample or patient identifiers,
	•	exporting cleaned tables for aggregation.

Stage 2. Preprocess cohort SCPCP000008

python scripts/scpc00008_clean.py
python scripts/scpc00008_stream.py

These scripts process the SCPCP000008 cohort, likely working across multiple per-sample .h5ad files. Typical operations may include:
	•	reading per-sample filtered objects,
	•	aligning metadata fields across samples,
	•	extracting cell annotations,
	•	generating sample-level summaries.

Stage 3. Merge cohorts and build TME feature tables

python scripts/merge_scpcp_TME.py
python scripts/scpcp_combined_sample_TME_feature_broad.py
python scripts/build_covariates.py

This stage combines cohort-level outputs and constructs model-ready TME covariates.

Typical outputs include:
	•	sample-level TME composition matrices,
	•	broad cell-type fraction tables,
	•	standardized patient-level covariates,
	•	matrices for ecotype discovery and downstream modeling.

Stage 4. Identify patient immune ecotypes

python scripts/generate_patient_immune_ecotypes.py

This script clusters or assigns samples/patients into immune ecotypes using the derived TME composition features. In the manuscript context, these ecotypes represent discrete microenvironmental regimes used to modulate model parameters.

Typical outputs may include:
	•	ecotype labels per patient or sample,
	•	ecotype centroid tables,
	•	cluster-membership summaries,
	•	files used in ecotype visualization panels.

Stage 5. Fit ecotype-modulated OU / OU–Branching models

python scripts/ou_ecotype_ou_branching_calibration.py

This is the core model-fitting step. It calibrates ecotype-modulated OU / OU–Branching dynamics using the patient-level longitudinal structure and TME-derived covariates.

Depending on implementation, this script may:
	•	build design matrices,
	•	specify priors,
	•	fit Bayesian models,
	•	summarize posterior samples,
	•	export posterior objects and diagnostics.

Stage 6. Summarize posterior parameters

python scripts/summary_mu_theta_ecotype.py

This step generates summary tables for key model parameters, especially the ecotype-associated posterior distributions of:
	•	mu-related quantities (drift / optimum terms),
	•	theta-related quantities (stabilizing-selection terms).

These summaries underlie later plots and interpretation.

Stage 7. Generate ecotype parameter visualizations

python scripts/plot_mu_thetha_ecotype_violin.py
python scripts/plot_mu_theta_ecotype_with_patients.py
python scripts/plot_mu_theta_scatter_with_centroids.py

These scripts visualize the posterior parameter structure across ecotypes and patients.

Typical figure types include:
	•	violin plots of posterior distributions by ecotype,
	•	patient-overlaid ecotype parameter plots,
	•	scatter plots with ecotype centroids.

Figure generation

Main figures

python scripts/fig2_full.py
python scripts/fig3_full.py
python scripts/fig4_full.py

These scripts generate the main manuscript figures after preprocessing, ecotype generation, and model fitting have completed.

The exact panel contents depend on the current manuscript version, but broadly:
	•	Figure 2: TME/ecotype construction or cohort-level composition summaries
	•	Figure 3: ecotype-associated parameter summaries
	•	Figure 4: integrated model-based biological interpretation and/or patient-level visualization

Figure 5

python scripts/fit_fig5_models.py
python scripts/fig5_ablations_baselines.py

These scripts support the model-comparison and ablation figure.

Typical analyses here may include:
	•	OU-only baseline,
	•	ecotype-shuffled control,
	•	reduced covariate models,
	•	alternative calibration setups,
	•	predictive or uncertainty-comparison summaries.

Figures 6 and 7

python scripts/make_fig6_panels.py
python scripts/make_fig6_composite.py
python scripts/make_fig6_fig7_layout_row1row2row3.py

These scripts assemble later-stage manuscript figures and composite layouts. Depending on the version of the manuscript, they may include:
	•	posterior predictive checks,
	•	centroid or ecotype summary panels,
	•	cross-patient overlays,
	•	multi-row final figure composition.

Script inventory

Below is a brief description of each script currently listed in the repository.

Preprocessing and cohort harmonization
	•	scripts/scpc000022.py
Loads and preprocesses the SCPCP000022 merged single-cell dataset.
	•	scripts/scpc000022_clean.py
Cleans metadata and prepares SCPCP000022 for harmonized downstream analysis.
	•	scripts/scpc00008_clean.py
Cleans SCPCP000008 sample-level inputs and harmonizes annotation structure.
	•	scripts/scpc00008_stream.py
Processes SCPCP000008 files in a sample-wise or stream-like workflow.
	•	scripts/merge_scpcp_TME.py
Merges cohort-level TME information across SCPCP000008 and SCPCP000022.
	•	scripts/scpcp_combined_sample_TME_feature_broad.py
Builds broad sample-level TME composition features for combined analysis.
	•	scripts/build_covariates.py
Creates model-ready covariates, standardizations, and matrices.

Ecotype analysis
	•	scripts/generate_patient_immune_ecotypes.py
Infers or assigns patient-level immune ecotypes from TME composition features.

Model fitting and summaries
	•	scripts/ou_ecotype_ou_branching_calibration.py
Fits the ecotype-modulated OU / OU–Branching model.
	•	scripts/summary_mu_theta_ecotype.py
Summarizes posterior parameter estimates by ecotype.

Parameter visualization
	•	scripts/plot_mu_thetha_ecotype_violin.py
Generates violin plots for ecotype-stratified posterior parameter distributions.
	•	scripts/plot_mu_theta_ecotype_with_patients.py
Overlays patient-level estimates on ecotype-level parameter summaries.
	•	scripts/plot_mu_theta_scatter_with_centroids.py
Generates scatter plots of ecotype parameter space with centroids.

Main figure scripts
	•	scripts/fig2_full.py
Generates the full Figure 2.
	•	scripts/fig3_full.py
Generates the full Figure 3.
	•	scripts/fig4_full.py
Generates the full Figure 4.

Figure 5 scripts
	•	scripts/fit_fig5_models.py
Fits models used in the Figure 5 comparison / ablation analyses.
	•	scripts/fig5_ablations_baselines.py
Generates Figure 5 from the fitted comparison outputs.

Figures 6–7 layout scripts
	•	scripts/make_fig6_panels.py
Creates panel-level components for Figure 6.
	•	scripts/make_fig6_composite.py
Assembles Figure 6 from the panel-level outputs.
	•	scripts/make_fig6_fig7_layout_row1row2row3.py
Builds the final multi-row layout used for Figure 6 and/or Figure 7.

Typical usage

A minimal full run from the repository root may look like:

python scripts/scpc000022.py
python scripts/scpc000022_clean.py
python scripts/scpc00008_clean.py
python scripts/scpc00008_stream.py

python scripts/merge_scpcp_TME.py
python scripts/scpcp_combined_sample_TME_feature_broad.py
python scripts/build_covariates.py
python scripts/generate_patient_immune_ecotypes.py

python scripts/ou_ecotype_ou_branching_calibration.py
python scripts/summary_mu_theta_ecotype.py

python scripts/plot_mu_thetha_ecotype_violin.py
python scripts/plot_mu_theta_ecotype_with_patients.py
python scripts/plot_mu_theta_scatter_with_centroids.py

python scripts/fig2_full.py
python scripts/fig3_full.py
python scripts/fig4_full.py

python scripts/fit_fig5_models.py
python scripts/fig5_ablations_baselines.py

python scripts/make_fig6_panels.py
python scripts/make_fig6_composite.py
python scripts/make_fig6_fig7_layout_row1row2row3.py

Python environment

The analysis was developed in Python 3 and uses the standard scientific Python stack. A typical environment will require many of the following packages:

pip install numpy pandas scipy matplotlib scikit-learn openpyxl
pip install scanpy anndata
pip install pymc arviz

Depending on the exact script implementations, additional packages may be required.

A conda-based setup is also reasonable, for example:

conda create -n tme-ou python=3.11
conda activate tme-ou
pip install numpy pandas scipy matplotlib scikit-learn openpyxl scanpy anndata pymc arviz

Typical data loading examples

Read the cleaned clinical longitudinal table

import pandas as pd

clinical = pd.read_excel("data/clinical/kmt2a_longitudinal_clean.xlsx")
print(clinical.head())

Load a single-cell AnnData object

import scanpy as sc

adata = sc.read_h5ad(
    "data/SCPCP000022_SINGLE-CELL_ANN-DATA_MERGED_2025-12-08/SCPCP000022_merged_rna.h5ad"
)
print(adata)

Read ecotype or TME summary tables

import pandas as pd

tme = pd.read_csv("results/tme_features/sample_tme_features.csv")
ecotypes = pd.read_csv("results/ecotypes/patient_immune_ecotypes.csv")

Reproducibility notes
	•	The repository is organized around a manuscript-specific workflow.
	•	Intermediate outputs are written to results/ so that later stages can be regenerated without rerunning the entire pipeline.
	•	Figure scripts assume that required upstream outputs already exist.
	•	Some scripts may be cohort-specific or manuscript-version-specific.
	•	Exact outputs may vary slightly depending on software versions and any stochastic model-fitting settings.

A simple starting point could be:

numpy
pandas
scipy
matplotlib
scikit-learn
openpyxl
scanpy
anndata
pymc
arviz

Citation

If you use this repository, please cite:

Kim SH. Single-cell immune ecotypes shape microenvironment-modulated evolutionary dynamics in pediatric leukemia. Research Square. 2026. doi:10.21203/rs.3.rs-9012769/v1

Contact

For questions about the analysis, figure generation, or manuscript-specific outputs, please contact the repository author.

Author: Seung-Hwan Kim

DOI
10.5281/zenodo.19619392

[![DOI](https://zenodo.org/badge/1170246489.svg)](https://doi.org/10.5281/zenodo.19619391)
