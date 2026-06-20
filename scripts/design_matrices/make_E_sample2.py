import numpy as np
import pandas as pd
from pathlib import Path

EPS = 1e-6

def logit(p, eps=EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

base = Path("/TME_OU_Branching")
feat_dir = base / "derived_features"
infile = feat_dir / "scpcp_combined_sample_TME_features_broad.csv"

df = pd.read_csv(infile, index_col=0)

# ----------------------------
# 1) Basic checks + derived cols
# ----------------------------
needed = ["participant_id","diagnosis","subdiagnosis","tissue_location","disease_timing",
          "frac_T","frac_B","frac_myeloid","frac_NK","frac_stromal","frac_known","cohort"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in input: {missing}")

# tumor/unannotated proxy
df["frac_unknown"] = 1.0 - df["frac_known"]

# immune fractions among "known" (avoid divide-by-zero)
den = df["frac_known"].clip(lower=EPS)
for c in ["frac_T","frac_B","frac_myeloid","frac_NK","frac_stromal"]:
    df[c + "_given_known"] = df[c] / den

# optional: total immune
df["frac_immune_total"] = df["frac_T"] + df["frac_B"] + df["frac_myeloid"] + df["frac_NK"]
df["frac_immune_given_known"] = df["frac_immune_total"] / den

# ----------------------------
# 2) Choose your E feature set (TME-only)
# ----------------------------
E_cols_tme = [
    "frac_unknown",
    "frac_T_given_known",
    "frac_myeloid_given_known",
    "frac_NK_given_known",
    "frac_B_given_known",
    "frac_stromal_given_known",
]

X = df[E_cols_tme].astype(float).copy()

# logit-transform proportions (recommended for OU covariates)
X_logit = X.apply(lambda s: logit(s.values), axis=0)
X_logit = pd.DataFrame(X_logit, index=X.index, columns=[f"logit_{c}" for c in X.columns])

# z-score
Xz = (X_logit - X_logit.mean(axis=0)) / X_logit.std(axis=0).replace(0, 1)

# ----------------------------
# 3) Optional: add nuisance covariates (one-hot)
# ----------------------------
cat_cols = ["diagnosis","tissue_location","disease_timing","cohort"]
C = pd.get_dummies(df[cat_cols], drop_first=True)

E_full = pd.concat([Xz, C], axis=1)

# ----------------------------
# 4) Write outputs
# ----------------------------
out_tme_csv  = feat_dir / "E_sample_TMEonly.csv"
out_full_csv = feat_dir / "E_sample_TMEplusNuisance.csv"
meta_csv     = feat_dir / "E_sample_row_metadata.csv"

Xz.to_csv(out_tme_csv)
E_full.to_csv(out_full_csv)

df[["participant_id","diagnosis","subdiagnosis","tissue_location","disease_timing","cohort"]].to_csv(meta_csv)

# also write npz for fast loading
np.savez(
    feat_dir / "E_sample_TMEonly.npz",
    E=Xz.to_numpy(dtype=float),
    sample_ids=Xz.index.to_numpy(),
    feature_names=Xz.columns.to_numpy()
)
np.savez(
    feat_dir / "E_sample_TMEplusNuisance.npz",
    E=E_full.to_numpy(dtype=float),
    sample_ids=E_full.index.to_numpy(),
    feature_names=E_full.columns.to_numpy()
)

print("Wrote:")
print(" ", out_tme_csv)
print(" ", out_full_csv)
print(" ", meta_csv)
print("NPZ:")
print(" ", feat_dir / "E_sample_TMEonly.npz")
print(" ", feat_dir / "E_sample_TMEplusNuisance.npz")
print("Shapes:")
print("  TME-only:", Xz.shape)
print("  Full:", E_full.shape)
