"""
scpc000022_clean.py
Collapse SCPCP000022 sample-level cell-type fractions into broad TME axes
+ add TNK and immune-only compositions.
"""

import pandas as pd
from pathlib import Path

base = Path("/")
inp = base / "derived_features/scpcp22_sample_TME_features.csv"
df = pd.read_csv(inp, index_col=0)

# ---------------------------
# columns
# ---------------------------
meta_cols = ["participant_id", "diagnosis", "subdiagnosis", "tissue_location", "disease_timing"]
ctype_cols = [c for c in df.columns if c not in meta_cols]

def pick_cols(keywords):
    kws = [k.lower() for k in keywords]
    out = []
    for c in ctype_cols:
        cl = c.lower()
        if any(k in cl for k in kws):
            out.append(c)
    return out

# ---------------------------
# broad categories (robust matching)
# ---------------------------
t_cells = pick_cols(["t cell", "t-cell", "cd4", "cd8", "treg"])
b_cells = pick_cols(["b cell", "b-cell", "plasma", "plasmablast"])
myeloid = pick_cols(["monocyte", "macrophage", "myeloid", "dendritic", "granulocyte", "neutrophil"])
nk      = pick_cols(["natural killer"])  # IMPORTANT: avoid "nk" (matches 'Unknown')
stromal = pick_cols(["fibroblast", "fibro", "platelet", "endothelial", "endo", "stromal"])

# ---------------------------
# broad fractions
# ---------------------------
df_broad = df[meta_cols].copy()
df_broad["frac_Tcell"]   = df[t_cells].sum(axis=1)   if t_cells else 0.0
df_broad["frac_B"]       = df[b_cells].sum(axis=1)   if b_cells else 0.0
df_broad["frac_myeloid"] = df[myeloid].sum(axis=1)   if myeloid else 0.0
df_broad["frac_NK"]      = df[nk].sum(axis=1)        if nk else 0.0
df_broad["frac_stromal"] = df[stromal].sum(axis=1)   if stromal else 0.0

# T+NK combined (for SCPCP000008 comparability)
df_broad["frac_TNK"] = df_broad["frac_Tcell"] + df_broad["frac_NK"]

# known fraction (exclude Unknown if present)
if "Unknown" in df.columns:
    df_broad["frac_known"] = df[ctype_cols].sum(axis=1) - df["Unknown"]
else:
    df_broad["frac_known"] = df[ctype_cols].sum(axis=1)

# ---------------------------
# immune-only (composition within immune/stromal compartment)
# denom = T + B + myeloid + NK + stromal
# ---------------------------
df_broad["frac_immune_total"] = (
    df_broad["frac_Tcell"] + df_broad["frac_B"] + df_broad["frac_myeloid"] +
    df_broad["frac_NK"] + df_broad["frac_stromal"]
)

den = df_broad["frac_immune_total"].replace(0, pd.NA)

df_broad["frac_Tcell_immune"]   = (df_broad["frac_Tcell"]   / den).fillna(0.0)
df_broad["frac_B_immune"]       = (df_broad["frac_B"]       / den).fillna(0.0)
df_broad["frac_myeloid_immune"] = (df_broad["frac_myeloid"] / den).fillna(0.0)
df_broad["frac_NK_immune"]      = (df_broad["frac_NK"]      / den).fillna(0.0)
df_broad["frac_stromal_immune"] = (df_broad["frac_stromal"] / den).fillna(0.0)
df_broad["frac_TNK_immune"]     = (df_broad["frac_TNK"]     / den).fillna(0.0)

# ---------------------------
# output
# ---------------------------
out = base / "derived_features/scpcp22_sample_TME_features_broad.csv"
df_broad.to_csv(out)

print("Wrote:", out, "shape:", df_broad.shape)
print("Unique diagnoses:", df_broad["diagnosis"].unique())

print("\nMatched columns:")
print("  T:", t_cells)
print("  B:", b_cells)
print("  Myeloid:", myeloid)
print("  NK:", nk)
print("  Stromal:", stromal)
