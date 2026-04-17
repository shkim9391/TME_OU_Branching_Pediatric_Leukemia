#scpc00008_clean.py

Memory‑safe TME feature extraction for SCPCP000008 (ALL_104).
This version NEVER concatenates AnnData objects,
so it will not crash your kernel or exceed macOS RAM.
"""

import scanpy as sc
import pandas as pd
from pathlib import Path
import gc

# ------------------------------------------------------
# PATHS
# ------------------------------------------------------
base = Path("/")
proj = base / "SCPCP000008_SINGLE-CELL_ANN-DATA_2025-12-08"
meta_path = proj / "single_cell_metadata.tsv"

# read sample metadata once
meta = pd.read_csv(meta_path, sep="\t")

# collect sample IDs from metadata
sample_ids = meta["scpca_sample_id"].unique()

print(f"Found {len(sample_ids)} samples in metadata.")

# ------------------------------------------------------
# STEP 1 — STREAMING CELL‑TYPE COUNTS
# ------------------------------------------------------
all_counts = {}      # dict: sample_id -> Series of cell‑type counts

for sid in sample_ids:
    folder = proj / sid

    # find the filtered h5ad file
    h5ads = list(folder.glob("*_filtered_rna.h5ad"))
    if not h5ads:
        print(f"[WARN] No filtered h5ad for {sid}, skipping.")
        continue

    h5ad_path = h5ads[0]
    print(f"Reading {sid} from {h5ad_path.name} ...")

    # load in backed mode (does NOT load X matrices into RAM)
    ad = sc.read_h5ad(h5ad_path, backed="r")

    # choose the cell‑type label
    candidates = [
        "consensus_celltype_annotation",
        "scimilarity_celltype_annotation",
        "cellassign_celltype_annotation",
        "singler_celltype_annotation",
        "submitter_celltype_annotation",
    ]

    for c in candidates:
        if c in ad.obs.columns:
            ctype_col = c
            break
    else:
        print(f"[WARN] No cell‑type column found for {sid}, skipping.")
        continue

    # count cell types
    counts = ad.obs[ctype_col].value_counts()

    all_counts[sid] = counts

    # free memory
    del ad
    gc.collect()

# ------------------------------------------------------
# STEP 2 — ALIGN COUNTS INTO DATAFRAME
# ------------------------------------------------------
counts_df = pd.DataFrame(all_counts).T.fillna(0)
fractions_df = counts_df.div(counts_df.sum(axis=1), axis=0)

# ------------------------------------------------------
# STEP 3 — JOIN WITH SAMPLE METADATA
# ------------------------------------------------------
meta_sample = (
    meta[
        ["scpca_sample_id", "participant_id", "diagnosis",
         "subdiagnosis", "tissue_location", "disease_timing"]
    ]
    .drop_duplicates("scpca_sample_id")
    .set_index("scpca_sample_id")
)

design = meta_sample.join(fractions_df, how="inner")

outdir = base / "derived_features"
outdir.mkdir(exist_ok=True)

raw_out = outdir / "scpcp8_sample_TME_features_raw.csv"
design.to_csv(raw_out)
print(f"\nRaw sample‑level TME features written to:\n  {raw_out}")
print("Shape:", design.shape)

# ------------------------------------------------------
# STEP 4 — BUILD BROAD TME AXES (same as SCPCP000022)
# ------------------------------------------------------
meta_cols = ["participant_id", "diagnosis", "subdiagnosis",
             "tissue_location", "disease_timing"]
ctype_cols = [c for c in design.columns if c not in meta_cols]

def pick(substrings):
    return [c for c in ctype_cols if any(s in c for s in substrings)]

t_cells = pick(["T cell"])
b_cells = pick(["B cell", "plasma"])
myeloid = pick(["monocyte", "macrophage", "myeloid", "dendritic", "granulocyte"])
nk      = pick(["natural killer", "NK"])
stromal = [c for c in ctype_cols if c in ["fibroblast", "platelet"]]

df_broad = design[meta_cols].copy()
df_broad["frac_T"]       = design[t_cells].sum(axis=1) if t_cells else 0.0
df_broad["frac_B"]       = design[b_cells].sum(axis=1) if b_cells else 0.0
df_broad["frac_myeloid"] = design[myeloid].sum(axis=1) if myeloid else 0.0
df_broad["frac_NK"]      = design[nk].sum(axis=1) if nk else 0.0
df_broad["frac_stromal"] = design[stromal].sum(axis=1) if stromal else 0.0
df_broad["frac_known"]   = design[ctype_cols].sum(axis=1) - design.get("Unknown", 0)

broad_out = outdir / "scpcp8_sample_TME_features_broad.csv"
df_broad.to_csv(broad_out)

print(f"\nBroad TME feature table written to:\n  {broad_out}")
print("Shape:", df_broad.shape)

