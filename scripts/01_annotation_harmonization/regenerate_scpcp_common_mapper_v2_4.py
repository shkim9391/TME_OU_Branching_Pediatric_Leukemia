from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc


ANNOTATION_PRIORITY = [
    "consensus_celltype_annotation",
    "scimilarity_celltype_annotation",
    "cellassign_celltype_annotation",
    "singler_celltype_annotation",
    "submitter_celltype_annotation",
]

META_COLS = [
    "scpca_sample_id",
    "participant_id",
    "diagnosis",
    "subdiagnosis",
    "tissue_location",
    "disease_timing",
]

BROAD_CLASSES = ["TNK", "B", "Myeloid", "Progenitor", "Erythroid", "Stromal", "Blast_Malignant", "Unknown", "Excluded", "Other"]


def norm_label(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def broad_cell_type(label) -> str:
    """Map both projects to one conservative shared higher-level ontology."""
    if pd.isna(label):
        return "Unknown"
    s = str(label).strip().lower()

    exact = {
        "b": "B", "plasmablast": "B",
        "monocyte": "Myeloid", "dc": "Myeloid",
        "t_nk": "TNK", "t-nk": "TNK",
        "hsc/mpp and pro.": "Progenitor", "hsc/mpp and pro": "Progenitor",
        "erythroid": "Erythroid",
        "blast": "Blast_Malignant",
        "submitter-excluded": "Excluded", "submitter excluded": "Excluded",
        "unknown": "Unknown",
    }
    if s in exact:
        return exact[s]

    if any(k in s for k in ["submitter-excluded","submitter excluded","excluded"]):
        return "Excluded"
    if any(k in s for k in ["unknown","unassigned","unannotated","undetermined"]):
        return "Unknown"
    if any(k in s for k in ["blast","leukemic","leukaemic","malignant"]):
        return "Blast_Malignant"
    if any(k in s for k in [
        "hsc","hematopoietic precursor","haematopoietic precursor",
        "hematopoietic stem","haematopoietic stem","progenitor",
        "common lymphoid progenitor","oligopotent progenitor",
        "multipotent progenitor","mpp"
    ]):
        return "Progenitor"
    if any(k in s for k in ["erythroid","erythrocyte","erythroblast"]):
        return "Erythroid"
    if any(k in s for k in [
        "t cell","t-cell","cd4","cd8","treg","regulatory t",
        "natural killer","nk cell","nk-cell","t_nk","t-nk"
    ]):
        return "TNK"
    if any(k in s for k in [
        "b cell","b-cell","b lineage","b-lineage","lymphocyte of b lineage",
        "plasma","plasmablast","antibody secreting","antibody-secreting"
    ]):
        return "B"
    if any(k in s for k in [
        "monocyte","macrophage","myeloid","dendritic","granulocyte",
        "neutrophil","mononuclear phagocyte","phagocyte"
    ]):
        return "Myeloid"
    if any(k in s for k in ["fibroblast","fibro","platelet","endothelial","stromal"]):
        return "Stromal"
    return "Other"

def choose_annotation_column(obs: pd.DataFrame) -> Optional[str]:
    for c in ANNOTATION_PRIORITY:
        if c in obs.columns:
            return c
    return None


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path, sep="\t")
    missing = [c for c in META_COLS if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing metadata columns in {path}: {missing}")
    return (
        meta[META_COLS]
        .drop_duplicates("scpca_sample_id")
        .copy()
    )


def counts_to_outputs(
    project: str,
    per_sample_counts: Dict[str, pd.Series],
    sample_annotation_col: Dict[str, str],
    metadata: pd.DataFrame,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert original-label counts to shared broad fractions + audits."""

    audit_rows: List[dict] = []
    broad_rows: List[dict] = []

    meta_idx = metadata.set_index("scpca_sample_id")

    for sid, counts in per_sample_counts.items():
        if sid not in meta_idx.index:
            continue

        counts = counts.astype(float)
        total = float(counts.sum())
        if total <= 0:
            continue

        broad_counts = {k: 0.0 for k in BROAD_CLASSES}
        for original_label, n in counts.items():
            broad = broad_cell_type(original_label)
            broad_counts[broad] += float(n)
            audit_rows.append({
                "project": project,
                "scpca_sample_id": sid,
                "annotation_column": sample_annotation_col.get(sid, ""),
                "original_annotation": str(original_label),
                "broad_class": broad,
                "cell_count": int(n),
                "cell_fraction": float(n) / total,
            })

        row = {
            "project": project,
            "scpca_sample_id": sid,
            "annotation_column": sample_annotation_col.get(sid, ""),
            "n_cells": int(total),
        }
        m = meta_idx.loc[sid]
        for c in META_COLS:
            if c != "scpca_sample_id":
                row[c] = m[c]

        for broad in BROAD_CLASSES:
            row[f"frac_{broad}"] = broad_counts[broad] / total

        # Annotation/QC coverage. Keep biological Unknown, Excluded, and Other distinct.
        row["frac_known_broad"] = (
            1.0 - row["frac_Unknown"] - row["frac_Excluded"] - row["frac_Other"]
        )
        row["frac_mapped_any"] = 1.0 - row["frac_Other"]

        # Primary conservative nonmalignant immune/stromal composition.
        primary_classes = ["TNK", "B", "Myeloid", "Stromal"]
        row["frac_immune_stromal_total"] = sum(
            row[f"frac_{broad}"] for broad in primary_classes
        )
        den = row["frac_immune_stromal_total"]
        for broad in primary_classes:
            row[f"frac_{broad}_given_immune_stromal"] = (
                row[f"frac_{broad}"] / den if den > 0 else np.nan
            )

        # Broader marrow-ecology composition, excluding malignant/annotation-QC classes.
        ecology_classes = ["TNK", "B", "Myeloid", "Progenitor", "Erythroid", "Stromal"]
        row["frac_ecology_total"] = sum(
            row[f"frac_{broad}"] for broad in ecology_classes
        )
        eden = row["frac_ecology_total"]
        for broad in ecology_classes:
            row[f"frac_{broad}_given_ecology"] = (
                row[f"frac_{broad}"] / eden if eden > 0 else np.nan
            )

        broad_rows.append(row)

    broad_df = pd.DataFrame(broad_rows)
    audit_df = pd.DataFrame(audit_rows)

    if len(audit_df):
        audit_summary = (
            audit_df.groupby(
                ["project", "annotation_column", "original_annotation", "broad_class"],
                as_index=False,
            )["cell_count"].sum()
        )
        audit_summary["fraction_within_project"] = (
            audit_summary["cell_count"]
            / audit_summary.groupby("project")["cell_count"].transform("sum")
        )
    else:
        audit_summary = audit_df.copy()

    outdir.mkdir(parents=True, exist_ok=True)
    broad_df.to_csv(outdir / f"{project}_sample_TME_features_common_mapper.csv", index=False)
    audit_df.to_csv(outdir / f"{project}_annotation_audit_by_sample.csv", index=False)
    audit_summary.to_csv(outdir / f"{project}_annotation_audit_summary.csv", index=False)

    return broad_df, audit_summary


def process_scpcp000008(root: Path, outdir: Path) -> pd.DataFrame:
    project = "SCPCP000008"
    proj = root / "SCPCP000008_SINGLE-CELL_ANN-DATA_2025-12-08"
    meta = load_metadata(proj / "single_cell_metadata.tsv")

    per_sample_counts: Dict[str, pd.Series] = {}
    sample_annotation_col: Dict[str, str] = {}

    for sid in meta["scpca_sample_id"].astype(str).unique():
        folder = proj / sid
        h5ad = next(folder.glob("*_filtered_rna.h5ad"), None)
        if h5ad is None:
            print(f"[WARN] {project}: missing filtered h5ad for {sid}")
            continue

        ad = sc.read_h5ad(h5ad, backed="r")
        ctype_col = choose_annotation_column(ad.obs)
        if ctype_col is None:
            print(f"[WARN] {project}: no supported annotation column for {sid}")
            del ad
            gc.collect()
            continue

        per_sample_counts[sid] = ad.obs[ctype_col].value_counts(dropna=False)
        sample_annotation_col[sid] = ctype_col
        del ad
        gc.collect()

    broad_df, _ = counts_to_outputs(
        project, per_sample_counts, sample_annotation_col, meta, outdir
    )
    return broad_df


def process_scpcp000022(root: Path, outdir: Path) -> pd.DataFrame:
    project = "SCPCP000022"
    proj = root / "SCPCP000022_SINGLE-CELL_ANN-DATA_MERGED_2025-12-08"
    meta = load_metadata(proj / "single_cell_metadata.tsv")

    h5ad = proj / "SCPCP000022_merged_rna.h5ad"
    ad = sc.read_h5ad(h5ad, backed="r")
    ctype_col = choose_annotation_column(ad.obs)
    if ctype_col is None:
        raise ValueError(f"{project}: no supported annotation column in merged AnnData")

    # Prefer explicit sample_id; fall back to scpca_sample_id.
    if "sample_id" in ad.obs.columns:
        sample_col = "sample_id"
    elif "scpca_sample_id" in ad.obs.columns:
        sample_col = "scpca_sample_id"
    else:
        raise ValueError(f"{project}: missing sample identifier column in ad.obs")

    obs = ad.obs[[sample_col, ctype_col]].copy()
    obs[sample_col] = obs[sample_col].astype(str)

    per_sample_counts: Dict[str, pd.Series] = {}
    sample_annotation_col: Dict[str, str] = {}
    for sid, g in obs.groupby(sample_col, observed=True):
        per_sample_counts[str(sid)] = g[ctype_col].value_counts(dropna=False)
        sample_annotation_col[str(sid)] = ctype_col

    del ad
    gc.collect()

    broad_df, _ = counts_to_outputs(
        project, per_sample_counts, sample_annotation_col, meta, outdir
    )
    return broad_df


def build_combined(sc8: pd.DataFrame, sc22: pd.DataFrame, outdir: Path) -> None:
    shared = sorted(set(sc8.columns) | set(sc22.columns))
    a = sc8.reindex(columns=shared)
    b = sc22.reindex(columns=shared)
    combined = pd.concat([a, b], ignore_index=True)
    combined.to_csv(outdir / "scpcp_combined_sample_TME_features_common_mapper.csv", index=False)

    # Participant aggregation: average sample-level fractions; retain first metadata value.
    frac_cols = [c for c in combined.columns if c.startswith("frac_")]
    meta_first_cols = [
        c for c in ["project", "diagnosis", "subdiagnosis", "tissue_location", "disease_timing"]
        if c in combined.columns
    ]

    agg_spec = {c: "mean" for c in frac_cols}
    agg_spec.update({c: "first" for c in meta_first_cols})
    participant = (
        combined.groupby("participant_id", as_index=False)
        .agg(agg_spec)
        .rename(columns={"participant_id": "Patient_ID"})
    )
    participant.to_csv(
        outdir / "scpcp_combined_participant_TME_features_common_mapper.csv",
        index=False,
    )

    # Project-level QC summary.
    qc = combined.groupby("project").agg(
        n_samples=("scpca_sample_id", "nunique"),
        n_participants=("participant_id", "nunique"),
        n_cells=("n_cells", "sum"),
        mean_frac_TNK=("frac_TNK", "mean"),
        mean_frac_B=("frac_B", "mean"),
        mean_frac_Myeloid=("frac_Myeloid", "mean"),
        mean_frac_Progenitor=("frac_Progenitor", "mean"),
        mean_frac_Erythroid=("frac_Erythroid", "mean"),
        mean_frac_Stromal=("frac_Stromal", "mean"),
        mean_frac_Blast_Malignant=("frac_Blast_Malignant", "mean"),
        mean_frac_Unknown=("frac_Unknown", "mean"),
        mean_frac_Excluded=("frac_Excluded", "mean"),
        mean_frac_Other=("frac_Other", "mean"),
        mean_frac_known_broad=("frac_known_broad", "mean"),
        mean_frac_immune_stromal_total=("frac_immune_stromal_total", "mean"),
        mean_frac_ecology_total=("frac_ecology_total", "mean"),
    ).reset_index()
    qc.to_csv(outdir / "common_mapper_project_qc_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/seung-hwan.kim/Desktop/TME_OU_Branching"),
        help="TME_OU_Branching project root",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: ROOT/revision_common_mapper)",
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else root / "revision_common_mapper"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Processing SCPCP000008...")
    sc8 = process_scpcp000008(root, outdir)
    print(f"[INFO] SCPCP000008 retained {len(sc8)} samples")

    print("[INFO] Processing SCPCP000022...")
    sc22 = process_scpcp000022(root, outdir)
    print(f"[INFO] SCPCP000022 retained {len(sc22)} samples")

    build_combined(sc8, sc22, outdir)

    print("\n[OK] Common-mapper v2 regeneration complete.")
    print(f"Outputs: {outdir}")
    print("\nPrimary files:")
    for name in [
        "SCPCP000008_sample_TME_features_common_mapper.csv",
        "SCPCP000022_sample_TME_features_common_mapper.csv",
        "scpcp_combined_sample_TME_features_common_mapper.csv",
        "scpcp_combined_participant_TME_features_common_mapper.csv",
        "common_mapper_project_qc_summary.csv",
        "SCPCP000008_annotation_audit_summary.csv",
        "SCPCP000022_annotation_audit_summary.csv",
    ]:
        print("  -", outdir / name)


if __name__ == "__main__":
    main()
