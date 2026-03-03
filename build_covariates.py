#build_covariates.py

"""

import pandas as pd
from pathlib import Path

ROOT = Path("/")

# Load cleaned master table
df = pd.read_csv(ROOT / "patient_master_table.csv")

# ---------------------------------------------------------
# 1. Remove duplicate ID column if still present
# ---------------------------------------------------------
if "participant_id.1" in df.columns:
    df = df.drop(columns=["participant_id.1"])

# ---------------------------------------------------------
# 2. Define which columns are TME continuous covariates
# ---------------------------------------------------------
tme_cols = [
    "frac_T_given_known_z",
    "frac_B_given_known_z",
    "frac_myeloid_given_known_z",
    "frac_NK_given_known_z",
    "frac_stromal_given_known_z",
    "frac_unknown_z",
]

# ---------------------------------------------------------
# 3. Define categorical variables to encode
# ---------------------------------------------------------
cat_cols = [
    "diagnosis",
    "subdiagnosis",
    "tissue_location",
    "disease_timing",
    "cohort",
]

# Keep only columns that actually exist in the table
cat_cols = [c for c in cat_cols if c in df.columns]
tme_cols = [c for c in tme_cols if c in df.columns]

# ---------------------------------------------------------
# 4. Build one-hot encoded categorical columns
# ---------------------------------------------------------
df_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=True)

# ---------------------------------------------------------
# 5. Build final covariate matrix
# ---------------------------------------------------------
covariates = pd.concat(
    [
        df[["Patient_ID"]],  # ID column
        df[tme_cols],        # TME continuous values
        df_cat,              # categorical encodings
    ],
    axis=1
)

# ---------------------------------------------------------
# 6. Save covariate matrix
# ---------------------------------------------------------
out = ROOT / "covariate_matrix.csv"
covariates.to_csv(out, index=False)

print("Covariate matrix written to:")
print("  ", out)

print("\nColumns included:")
print(list(covariates.columns))
