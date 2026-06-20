import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
base = Path("/TME_OU_Branching")
feat_dir = base / "derived_features"

sample_E_path = feat_dir / "E_sample_simple_z.csv"

df = pd.read_csv(sample_E_path, index_col=0)   # index = sample_id
print("Loaded sample-level E:", sample_E_path, "shape:", df.shape)

# figure out which columns are features (end with "_z")
feature_cols = [c for c in df.columns if c.endswith("_z")]
meta_cols = ["participant_id", "diagnosis", "subdiagnosis",
             "tissue_location", "disease_timing", "cohort"]

print("Feature columns:", feature_cols)

# -----------------------------
# Helper: aggregate by participant
# -----------------------------
def make_Ep(subdf, label):
    """
    subdf: subset of df (rows = samples) to use
    label: string prefix for filenames (e.g., 'Ep_baseline')
    """
    # numeric part
    X = subdf[feature_cols]

    # group by participant and average features
    Xp = X.groupby(subdf["participant_id"]).mean()

    # for metadata, take first non-null per participant
    meta = (
        subdf
        .groupby("participant_id")[meta_cols]
        .first()
    )

    # align order
    meta = meta.loc[Xp.index]

    # save
    out_csv = feat_dir / f"{label}_z.csv"
    out_npy = feat_dir / f"{label}_z.npy"
    out_meta = feat_dir / f"{label}_rowmeta.csv"

    Ep = Xp.to_numpy(dtype=float)
    np.save(out_npy, Ep)
    Xp.to_csv(out_csv)
    meta.to_csv(out_meta)

    print(f"\n{label}:")
    print("  patients:", Ep.shape[0])
    print("  features:", Ep.shape[1])
    print("  wrote:", out_csv)
    print("        ", out_npy)
    print("        ", out_meta)

# -----------------------------
# 1) Baseline: Initial Dx + Bone marrow
# -----------------------------
baseline_mask = (
    (df["disease_timing"] == "Initial diagnosis") &
    (df["tissue_location"] == "Bone marrow")
)

df_baseline = df[baseline_mask].copy()
print("Baseline samples:", df_baseline.shape[0])

make_Ep(df_baseline, label="Ep_baseline")

# -----------------------------
# 2) All samples averaged per patient
# -----------------------------
make_Ep(df, label="Ep_all")
