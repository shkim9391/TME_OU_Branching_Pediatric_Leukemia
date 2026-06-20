import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
ROOT = Path("/TME_OU_Branching")
FEAT_DIR = ROOT / "derived_features"
IN_FILE = FEAT_DIR / "scpcp_combined_sample_TME_features_broad.csv"

# ------------------------------------------------------------
# KNOBS (edit these if you want)
# ------------------------------------------------------------
FILTER_INITIAL_BM = False         # True = keep only Initial diagnosis + Bone marrow
MIN_FRAC_KNOWN = 0.05             # drop samples with too few "known" labeled cells (set 0.0 to keep all)
INCLUDE_CATEGORICAL = True        # one-hot encode metadata
CAT_COLS = ["diagnosis", "tissue_location", "disease_timing", "cohort"]
INCLUDE_SUBDIAGNOSIS = False      # can explode #features (many subtypes); keep False initially

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
df = pd.read_csv(IN_FILE, index_col=0)

# Optional filters
if FILTER_INITIAL_BM:
    df = df[(df["disease_timing"] == "Initial diagnosis") &
            (df["tissue_location"] == "Bone marrow")].copy()

if MIN_FRAC_KNOWN > 0:
    df = df[df["frac_known"] >= MIN_FRAC_KNOWN].copy()

# ------------------------------------------------------------
# DERIVED FEATURES (robust for tumor-heavy samples)
# ------------------------------------------------------------
eps = 1e-6
df["frac_unknown"] = 1.0 - df["frac_known"]  # tumor/unannotated proxy

# immune/stromal composition among "known" cells
for c in ["frac_T", "frac_B", "frac_myeloid", "frac_NK", "frac_stromal"]:
    df[f"{c}_given_known"] = df[c] / df["frac_known"].clip(lower=eps)

# Numeric covariates you’ll feed into the OU–Branching parameters
num_cols = [
    "frac_unknown",
    "frac_T_given_known",
    "frac_B_given_known",
    "frac_myeloid_given_known",
    "frac_NK_given_known",
    "frac_stromal_given_known",
]

# ------------------------------------------------------------
# Z-SCORE NUMERIC COVARIATES (helps MCMC/VI a lot)
# ------------------------------------------------------------
X_num = df[num_cols].astype(float)
scaler = StandardScaler()
X_num_z = pd.DataFrame(
    scaler.fit_transform(X_num),
    index=df.index,
    columns=[f"z_{c}" for c in num_cols],
)

# ------------------------------------------------------------
# OPTIONAL: ONE-HOT CATEGORICAL COVARIATES
# ------------------------------------------------------------
X_cat = pd.DataFrame(index=df.index)
if INCLUDE_CATEGORICAL:
    use_cat = [c for c in CAT_COLS if c in df.columns]
    if INCLUDE_SUBDIAGNOSIS and "subdiagnosis" in df.columns:
        use_cat = use_cat + ["subdiagnosis"]

    if use_cat:
        X_cat = pd.get_dummies(
            df[use_cat].fillna("NA").astype(str),
            drop_first=True,   # avoids collinearity with intercept
        )

# ------------------------------------------------------------
# ASSEMBLE E
# ------------------------------------------------------------
E_df = pd.concat([X_num_z, X_cat], axis=1)
E_df.insert(0, "intercept", 1.0)

# ------------------------------------------------------------
# SAVE OUTPUTS
# ------------------------------------------------------------
out_csv = FEAT_DIR / "E_sample_level.csv"
out_npy = FEAT_DIR / "E_sample_level.npy"
out_cols = FEAT_DIR / "E_sample_level_columns.txt"
out_rowmap = FEAT_DIR / "E_sample_level_rowmap.tsv"
out_scale = FEAT_DIR / "E_numeric_scaler_params.csv"

E_df.to_csv(out_csv)
np.save(out_npy, E_df.values)

with open(out_cols, "w") as f:
    f.write("\n".join(E_df.columns) + "\n")

rowmap_cols = [c for c in ["participant_id", "diagnosis", "subdiagnosis",
                           "tissue_location", "disease_timing", "cohort"] if c in df.columns]
df[rowmap_cols].to_csv(out_rowmap, sep="\t")

pd.DataFrame({"feature": num_cols, "mean": scaler.mean_, "std": scaler.scale_}).to_csv(out_scale, index=False)

print("Loaded:", IN_FILE)
print("Samples in E:", E_df.shape[0])
print("Features in E:", E_df.shape[1])
print("Saved:")
print(" ", out_csv)
print(" ", out_npy)
print(" ", out_rowmap)
print(" ", out_cols)
print(" ", out_scale)
