import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Load combined sample-level TME
# -----------------------------
base = Path("/TME_OU_Branching/derived_features")
df = pd.read_csv(base / "scpcp_combined_sample_TME_features_broad.csv", index_col=0)

# -----------------------------
# Construct useful features
# -----------------------------
# Tumor-like fraction
df["frac_unknown"] = 1.0 - df["frac_known"]
eps = 1e-6

# logit transform
def logit(x):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))

df["logit_frac_unknown"] = logit(df["frac_unknown"])

# Immune fractions among known
immune_cols = ["frac_T","frac_B","frac_myeloid","frac_NK","frac_stromal"]
den = df["frac_known"].clip(lower=eps)

for c in immune_cols:
    df[c + "_given_known"] = df[c] / den

# Log ratios relative to stromal
for c in ["frac_T", "frac_B", "frac_myeloid", "frac_NK"]:
    df[c + "_logratio"] = np.log(df[c + "_given_known"].clip(lower=eps)) \
                        - np.log(df["frac_stromal_given_known"].clip(lower=eps))

# -----------------------------
# Build final design matrix E
# -----------------------------
E_cols = [
    "logit_frac_unknown",
    "frac_T_logratio",
    "frac_B_logratio",
    "frac_myeloid_logratio",
    "frac_NK_logratio"
]

E = df[E_cols].copy()

# Z-score for model stability
E = (E - E.mean()) / E.std()

# Save to CSV and NumPy
E.to_csv(base / "E_design_matrix_sample_level.csv")

np.save(base / "E_design_matrix_sample_level.npy", E.to_numpy())

# Also save the mapping from row index → sample metadata
meta_cols = ["participant_id", "diagnosis", "subdiagnosis",
             "tissue_location", "disease_timing", "cohort"]
df[meta_cols].to_csv(base / "E_sample_metadata.csv")

print("Wrote:")
print(" ", base / "E_design_matrix_sample_level.csv")
print(" ", base / "E_sample_metadata.csv")
print("Shape of E:", E.shape)
