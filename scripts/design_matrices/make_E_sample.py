import pandas as pd
import numpy as np
from pathlib import Path

#---------------------------------------------------
# 1. Load combined sample-level TME features
#---------------------------------------------------
base = Path("/Desktop/TME_OU_Branching")
feat_dir = base / "derived_features"

df = pd.read_csv(
    feat_dir / "scpcp_combined_sample_TME_features_broad.csv",
    index_col=0
)

print("Loaded combined sample TME table:", df.shape)

# Columns we know are there:
# participant_id, diagnosis, subdiagnosis, tissue_location,
# disease_timing, frac_T, frac_B, frac_myeloid, frac_NK,
# frac_stromal, frac_known, cohort

#---------------------------------------------------
# 2. Engineer extra TME features
#---------------------------------------------------

# tumor / unannotated fraction
df["frac_unknown"] = 1.0 - df["frac_known"]

# avoid divide-by-zero
den = df["frac_known"].clip(lower=1e-6)

for col in ["frac_T", "frac_B", "frac_myeloid", "frac_NK", "frac_stromal"]:
    new_col = col + "_given_known"
    df[new_col] = df[col] / den

#---------------------------------------------------
# 3. Select columns for E (features) and X_meta (mapping)
#---------------------------------------------------

# choose TME features for the design matrix E
E_cols = [
    "frac_unknown",
    "frac_T_given_known",
    "frac_B_given_known",
    "frac_myeloid_given_known",
    "frac_NK_given_known",
    "frac_stromal_given_known",
]

E = df[E_cols].to_numpy(dtype=float)   # n_samples x K

print("Design matrix E shape:", E.shape)
print("Feature columns:", E_cols)

# optional: z-score features (comment out if you want raw scale)
E_mean = E.mean(axis=0, keepdims=True)
E_std  = E.std(axis=0, keepdims=True) + 1e-8
E_z = (E - E_mean) / E_std

#---------------------------------------------------
# 4. Build a metadata table mapping each row of E
#---------------------------------------------------
meta_cols = [
    "participant_id",
    "diagnosis",
    "subdiagnosis",
    "tissue_location",
    "disease_timing",
    "cohort",
]

meta = df[meta_cols].copy()
meta.index.name = "sample_index"  # row index of E

#---------------------------------------------------
# 5. Save E and meta
#---------------------------------------------------
outdir = feat_dir / "model_ready"
outdir.mkdir(exist_ok=True)

# raw and z-scored E as .npy
np.save(outdir / "E_sample_raw.npy", E)
np.save(outdir / "E_sample_z.npy", E_z)

# also save a CSV with E_z and metadata for sanity checks
E_df = meta.join(pd.DataFrame(E_z, index=meta.index, columns=E_cols))
E_df.to_csv(outdir / "E_sample_z_with_meta.csv")

print("\nSaved:")
print("  ", outdir / "E_sample_raw.npy")
print("  ", outdir / "E_sample_z.npy")
print("  ", outdir / "E_sample_z_with_meta.csv")
