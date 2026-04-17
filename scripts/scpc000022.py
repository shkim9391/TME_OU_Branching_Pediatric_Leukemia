#scpc000022.py
Build sample-level TME features from SCPCP000022 merged AnnData.
"""

import scanpy as sc
import pandas as pd
from pathlib import Path

base = Path("/")
scpcp22_dir = base / "SCPCP000022_SINGLE-CELL_ANN-DATA_MERGED_2025-12-08"

#--------------------------------------------------------------------
# 1. Load merged AnnData
#--------------------------------------------------------------------
adata = sc.read_h5ad(scpcp22_dir / "SCPCP000022_merged_rna.h5ad")
print(adata)

# pick a nice cell-type column
ctype_col = "consensus_celltype_annotation"
assert ctype_col in adata.obs.columns

adata.obs[ctype_col] = adata.obs[ctype_col].astype("category")
adata.obs["sample_id"] = adata.obs["sample_id"].astype("category")

print("\nCell-type categories:")
print(adata.obs[ctype_col].value_counts())
print("\nNumber of samples:", adata.obs["sample_id"].nunique())

#--------------------------------------------------------------------
# 2. Cell-type counts and fractions per sample
#--------------------------------------------------------------------
counts = (
    adata.obs
    .groupby(["sample_id", ctype_col])
    .size()
    .unstack(fill_value=0)
)

fractions = counts.div(counts.sum(axis=1), axis=0)

#--------------------------------------------------------------------
# 3. Attach sample-level metadata from single_cell_metadata.tsv
#--------------------------------------------------------------------
meta = pd.read_csv(scpcp22_dir / "single_cell_metadata.tsv", sep="\t")

# one row per sample
meta_sample = (
    meta[["scpca_sample_id", "participant_id", "diagnosis",
          "subdiagnosis", "tissue_location", "disease_timing"]]
    .drop_duplicates("scpca_sample_id")
    .set_index("scpca_sample_id")
)

# align index names with fractions table
fractions.index.name = "scpca_sample_id"

design = meta_sample.join(fractions, how="inner")

outdir = base / "derived_features"
outdir.mkdir(exist_ok=True)

out_path = outdir / "scpcp22_sample_TME_features.csv"
design.to_csv(out_path)

print(f"\nWrote sample-level TME feature table to:\n  {out_path}")
print("Shape:", design.shape)
