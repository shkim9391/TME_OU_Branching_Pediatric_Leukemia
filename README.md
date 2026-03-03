# TME_OU_Branching_Pediatric_Leukemia
Code and reproducible analysis for TME-modulated OU/OU–Branching modeling of pediatric leukemia evolution using ScPCA single-cell TME composition and immune ecotypes. Includes preprocessing, model fitting, posterior predictive checks, and figure generation. 
Seung-Hwan Kim. Single-cell immune ecotypes shape microenvironment-modulated evolutionary dynamics in pediatric leukemia, 03 March 2026, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-9012769/v1]

# Recommended repo layout
repo_root/
  scripts/                       # all .py scripts below
  models/                        # model code (e.g., ou_ecotype_model.py)
  data/
    SCPCP000022_SINGLE-CELL_ANN-DATA_MERGED_2025-12-08/
      SCPCP000022_merged_rna.h5ad
      single_cell_metadata.tsv
    SCPCP000008_SINGLE-CELL_ANN-DATA_2025-12-08/
      *_filtered_rna.h5ad
      single_cell_metadata.tsv
    clinical/
      kmt2a_longitudinal_clean.xlsx
  results/
    tme_features/
    combined/
    matrices/
    ecotypes/
    models/
    ppc/
  figures/

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

# Figure 5
python scripts/fit_fig5_models.py
python scripts/fig5_ablations_baselines.py

# Figure 6
python scripts/make_fig6_panels.py
python scripts/make_fig6_composite.py
python scripts/make_fig6_fig7_layout_row1row2row3.py
