import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
base = Path("/TME_OU_Branching")
feat_dir = base / "derived_features"
in_path = feat_dir / "scpcp_combined_sample_TME_features_broad.csv"

df = pd.read_csv(in_path, index_col=0)
print("Loaded:", in_path, "shape:", df.shape)

# -----------------------------
# 1. Basic sanity & derived fractions
# -----------------------------
frac_cols = ["frac_T", "frac_B", "frac_myeloid", "frac_NK", "frac_stromal", "frac_known"]
df[frac_cols] = df[frac_cols].astype(float).clip(0.0, 1.0)

eps = 1e-6
df["frac_unknown"] = (1.0 - df["frac_known"]).clip(0.0, 1.0)

den = df["frac_known"].clip(lower=eps)
df["frac_T_given_known"]       = df["frac_T"]       / den
df["frac_myeloid_given_known"] = df["frac_myeloid"] / den
df["frac_NK_given_known"]      = df["frac_NK"]      / den
df["frac_B_given_known"]       = df["frac_B"]       / den
df["frac_stromal_given_known"] = df["frac_stromal"] / den

# -----------------------------
# 2. Choose E columns
# -----------------------------
E_cols = [
    "frac_unknown",
    "frac_T_given_known",
    "frac_myeloid_given_known",
    "frac_NK_given_known",
    "frac_B_given_known",
    "frac_stromal_given_known",
]

E_raw = df[E_cols].to_numpy(dtype=float)

# z-score each column
mu = E_raw.mean(axis=0)
sd = E_raw.std(axis=0)
sd = np.where(sd == 0, 1.0, sd)  # safety
E_z = (E_raw - mu) / sd

E_colnames_z = [c + "_z" for c in E_cols]

# -----------------------------
# 3. Save outputs
# -----------------------------
# row-level metadata (sample_id is df.index)
meta_cols = ["participant_id", "diagnosis", "subdiagnosis",
             "tissue_location", "disease_timing", "cohort"]
rowmeta = df[meta_cols].copy()
rowmeta.index.name = "sample_id"

# CSV with metadata + z-scored features
E_z_df = rowmeta.join(
    pd.DataFrame(E_z, index=df.index, columns=E_colnames_z)
)

out_csv = feat_dir / "E_sample_simple_z.csv"
out_npy = feat_dir / "E_sample_simple_z.npy"
out_feats = feat_dir / "E_sample_simple_features.txt"
out_rowmeta = feat_dir / "E_sample_simple_rowmeta.csv"
out_stats = feat_dir / "E_sample_simple_stats.csv"

E_z_df.to_csv(out_csv)
np.save(out_npy, E_z)

with open(out_feats, "w") as f:
    for c in E_colnames_z:
        f.write(c + "\n")

rowmeta.to_csv(out_rowmeta)

pd.DataFrame(
    {"feature": E_cols, "mean_raw": mu, "sd_raw": sd}
).to_csv(out_stats, index=False)

print("Wrote:")
print(" ", out_csv)
print(" ", out_npy)
print(" ", out_feats)
print(" ", out_rowmeta)
print(" ", out_stats)
print("E_z shape:", E_z.shape)
