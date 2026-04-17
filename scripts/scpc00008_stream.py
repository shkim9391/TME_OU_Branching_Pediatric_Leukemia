#scpc00008_stream.py

Build broad TME features for SCPCP000008 (ALL_104) in exactly
the same way as for SCPCP000022.
"""

import scanpy as sc
import pandas as pd
from pathlib import Path

#---------------------------------------------------------
# Paths
#---------------------------------------------------------
base = Path("/")
proj_dir = base / "SCPCP000008_SINGLE-CELL_ANN-DATA_2025-12-08"
meta_path = proj_dir / "single_cell_metadata.tsv"

#---------------------------------------------------------
# 1. Load all filtered AnnData files and concatenate
#---------------------------------------------------------
h5ad_files = {}
for folder in sorted(proj_dir.glob("SCPCS*")):
    f = next(folder.glob("*_filtered_rna.h5ad"), None)
    if f is None:
        continue
    h5ad_files[folder.name] = f

print(f"Found {len(h5ad_files)} filtered AnnData files.")

adatas = []
for scpca_sample_id, f in h5ad_files.items():
    print("Reading:", scpca_sample_id, "from", f.name)
    ad = sc.read_h5ad(f)

    # ensure each cell knows which sample it came from
    ad.obs["scpca_sample_id"] = scpca_sample_id
    ad.obs["sample_id"] = scpca_sample_id

    adatas.append(ad)

# IMPORTANT: pass a *list* to concat, not a dict; no keys, no label
adata = sc.concat(adatas, join="outer")
print(adata)

# we also keep a simple 'sample_id' column matching scpca_sample_id
adata.obs["scpca_sample_id"] = adata.obs["scpca_sample_id"].astype("category")
if "sample_id" not in adata.obs.columns:
    adata.obs["sample_id"] = adata.obs["scpca_sample_id"]
adata.obs["sample_id"] = adata.obs["sample_id"].astype("category")

#---------------------------------------------------------
# 2. Cell-type counts and fractions per sample
#---------------------------------------------------------
ctype_col = "consensus_celltype_annotation"
assert ctype_col in adata.obs.columns, f"{ctype_col} not in adata.obs!"

adata.obs[ctype_col] = adata.obs[ctype_col].astype("category")

print("\nCell-type categories:")
print(adata.obs[ctype_col].value_counts())

print("\nNumber of samples:", adata.obs["sample_id"].nunique())

# counts by sample × cell type
counts = (
    adata.obs
    .groupby(["sample_id", ctype_col])
    .size()
    .unstack(fill_value=0)
)

fractions = counts.div(counts.sum(axis=1), axis=0)

#---------------------------------------------------------
# 3. Attach sample-level metadata
#---------------------------------------------------------
meta = pd.read_csv(meta_path, sep="\t")

meta_sample = (
    meta[["scpca_sample_id", "participant_id", "diagnosis",
          "subdiagnosis", "tissue_location", "disease_timing"]]
    .drop_duplicates("scpca_sample_id")
    .set_index("scpca_sample_id")
)

fractions.index.name = "scpca_sample_id"
design = meta_sample.join(fractions, how="inner")

outdir = base / "derived_features"
outdir.mkdir(exist_ok=True)

raw_out = outdir / "scpcp8_sample_TME_features_raw.csv"
design.to_csv(raw_out)
print(f"\nWrote raw sample-level TME feature table to:\n  {raw_out}")
print("Shape:", design.shape)

#---------------------------------------------------------
# 4. Collapse to broad TME axes (same as SCPCP000022)
#---------------------------------------------------------
meta_cols = ["participant_id", "diagnosis", "subdiagnosis",
             "tissue_location", "disease_timing"]
ctype_cols = [c for c in design.columns if c not in meta_cols]

def pick_cols(substrings):
    return [c for c in ctype_cols if any(s in c for s in substrings)]

t_cells = pick_cols(["T cell"])
b_cells = pick_cols(["B cell", "plasma"])
myeloid = pick_cols(["monocyte", "macrophage", "myeloid",
                     "dendritic", "granulocyte"])
nk      = pick_cols(["natural killer", "NK"])
stromal = [c for c in ctype_cols if c in ["fibroblast", "platelet"]]

df_broad = design[meta_cols].copy()
df_broad["frac_T"]       = design[t_cells].sum(axis=1)       if t_cells else 0.0
df_broad["frac_B"]       = design[b_cells].sum(axis=1)       if b_cells else 0.0
df_broad["frac_myeloid"] = design[myeloid].sum(axis=1)       if myeloid else 0.0
df_broad["frac_NK"]      = design[nk].sum(axis=1)            if nk else 0.0
df_broad["frac_stromal"] = design[stromal].sum(axis=1)       if stromal else 0.0
if "Unknown" in design.columns:
    df_broad["frac_known"] = design[ctype_cols].sum(axis=1) - design["Unknown"]
else:
    df_broad["frac_known"] = design[ctype_cols].sum(axis=1)

broad_out = outdir / "scpcp8_sample_TME_features_broad.csv"
df_broad.to_csv(broad_out)
print(f"\nWrote broad TME feature table to:\n  {broad_out}")
print("Shape:", df_broad.shape)
