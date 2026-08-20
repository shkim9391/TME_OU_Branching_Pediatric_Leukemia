from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


EPSILONS = [0.01, 0.025, 0.05]
PRIMARY = ["TNK", "B", "Myeloid"]


def clean_diag(x):
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    mp = {
        "B-cell acute lymphoblastic leukemia": "B-ALL",
        "T-cell acute lymphoblastic leukemia": "T-ALL",
        "Early T-cell precursor T-cell acute lymphoblastic leukemia": "ETP-ALL",
        "Acute myeloid leukemia": "AML",
        "Mixed phenotype acute leukemia": "MPAL",
    }
    return mp.get(s, s if s else "Unknown")


def normalize_ids(s):
    return s.astype(str).str.strip()


def load_ecology(root: Path):
    primary_path = (
        root
        / "revision_wave1_id_ball"
        / "primary_composition_with_legacy_ids.csv"
    )
    assign_path = (
        root
        / "revision_wave1_context_stability"
        / "cluster_assignments_raw_k2.csv"
    )

    if not primary_path.exists():
        raise FileNotFoundError(primary_path)
    if not assign_path.exists():
        raise FileNotFoundError(assign_path)

    eco = pd.read_csv(primary_path)
    eco["participant_id"] = normalize_ids(eco["participant_id"])
    eco["Patient_ID"] = normalize_ids(eco["Patient_ID"])
    eco["diagnosis"] = eco["diagnosis"].map(clean_diag)

    ass = pd.read_csv(assign_path)
    if "participant_id" not in ass.columns:
        if "Patient_ID" in ass.columns:
            ass = ass.rename(columns={"Patient_ID": "participant_id"})
        else:
            raise ValueError("Raw k=2 assignment file lacks participant identifier.")
    ass["participant_id"] = normalize_ids(ass["participant_id"])

    context_map = {
        "C1": "TNK-dominant",
        "C2": "B/Myeloid-shifted",
    }
    ass["ecological_context_k2"] = ass["cluster"].map(context_map)

    eco = eco.merge(
        ass[["participant_id", "cluster", "ecological_context_k2"]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    for outcol, src in [
        ("TNK", "comp_TNK"),
        ("B", "comp_B"),
        ("Myeloid", "comp_Myeloid"),
    ]:
        if src not in eco.columns:
            raise ValueError(f"Missing {src}")
        eco[outcol] = pd.to_numeric(eco[src], errors="coerce")

    eco["has_legacy_id"] = eco["Patient_ID"].notna() & eco["Patient_ID"].ne("nan")
    eco["has_k2_context"] = eco["ecological_context_k2"].notna()

    return eco


def build_n_series(series):
    df = series.copy()
    df["Patient_ID"] = normalize_ids(df["Patient_ID"])
    df["series"] = df["series"].astype(str).str.strip().str.lower()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.loc[df["series"].eq("n")].copy()
    df = df.dropna(subset=["Patient_ID", "t", "value"])
    df = df.sort_values(["Patient_ID", "t"]).reset_index(drop=True)

    # n must be ordinal state 0/1/2.
    bad = ~df["value"].isin([0, 1, 2])
    if bad.any():
        raise ValueError(
            "Unexpected n values outside {0,1,2}:\n"
            + df.loc[bad, ["Patient_ID", "t", "value"]].to_string(index=False)
        )
    df["value"] = df["value"].astype(int)

    # Audit and deterministically resolve same-time duplicates only if exact.
    dup_mask = df.duplicated(["Patient_ID", "t"], keep=False)
    duplicates = df.loc[dup_mask].copy()

    resolved = []
    resolution_rows = []

    for (pid, tt), g in df.groupby(["Patient_ID", "t"], sort=False):
        vals = sorted(pd.unique(g["value"]).tolist())

        if len(vals) == 1:
            row = g.iloc[0].copy()
            resolved.append(row)
            if len(g) > 1:
                resolution_rows.append({
                    "Patient_ID": pid,
                    "t": tt,
                    "original_values": "|".join(map(str, sorted(g["value"].tolist()))),
                    "resolved_value": int(vals[0]),
                    "resolution_rule": "collapsed_exact_duplicates",
                    "n_rows_original": int(len(g)),
                })
        else:
            # Do not invent an ordinal precedence rule.
            raise ValueError(
                f"Conflicting n states at identical time for {pid}, t={tt}: {vals}. "
                "Manual review is required."
            )

    obs = pd.DataFrame(resolved).sort_values(
        ["Patient_ID", "t"]
    ).reset_index(drop=True)
    resolution = pd.DataFrame(resolution_rows)

    obs["n_prev"] = obs.groupby("Patient_ID")["value"].shift(1)
    obs["dt"] = obs.groupby("Patient_ID")["t"].diff()

    trans = obs.dropna(subset=["n_prev", "dt"]).copy()
    trans = trans.loc[trans["dt"] > 0].copy()
    trans["n_prev"] = trans["n_prev"].astype(int)
    trans = trans.rename(columns={"t": "time", "value": "n"})
    trans["series"] = "n"

    return obs, trans, duplicates, resolution


def n_participant_summary(obs, trans):
    rows = []
    for pid, g in obs.groupby("Patient_ID"):
        gt = trans.loc[trans["Patient_ID"].eq(pid)]
        times = np.sort(g["t"].unique())
        dts = gt["dt"].to_numpy(float)

        row = {
            "Patient_ID": pid,
            "n_observations": int(len(g)),
            "n_unique_times": int(len(times)),
            "n_transitions": int(len(gt)),
            "first_time": float(times.min()),
            "last_time": float(times.max()),
            "followup_years": float(times.max() - times.min())
                if len(times) >= 2 else 0.0,
            "median_dt": float(np.median(dts)) if len(dts) else np.nan,
            "min_dt": float(np.min(dts)) if len(dts) else np.nan,
            "max_dt": float(np.max(dts)) if len(dts) else np.nan,
        }
        for state in [0, 1, 2]:
            row[f"n_state_{state}_count"] = int(np.sum(g["value"] == state))
        rows.append(row)

    return pd.DataFrame(rows)


def logit_clip(x, eps):
    x = np.asarray(x, dtype=float)
    xc = np.clip(x, eps, 1.0 - eps)
    return np.log(xc / (1.0 - xc)), xc


def eps_tag(eps):
    return str(eps).replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/TME_OU_Branching"
    )
    ap.add_argument(
        "--longitudinal",
        default="kmt2a_longitudinal_clean.xlsx"
    )
    ap.add_argument(
        "--x-input",
        default="revision_wave2_inputs/wave2_x_transition_table.csv"
    )
    args = ap.parse_args()

    root = Path(args.root)
    long_path = root / args.longitudinal
    x_path = root / args.x_input

    if not long_path.exists():
        raise FileNotFoundError(long_path)
    if not x_path.exists():
        raise FileNotFoundError(x_path)

    outroot = root / "revision_wave2_secondary_inputs"
    n_dir = outroot / "n"
    tx_dir = outroot / "transformed_x"
    n_dir.mkdir(parents=True, exist_ok=True)
    tx_dir.mkdir(parents=True, exist_ok=True)

    eco = load_ecology(root)

    # ============================================================
    # A. ORDINAL n ENDPOINT
    # ============================================================
    series = pd.read_excel(long_path, sheet_name="Series")
    required = {"Patient_ID", "series", "t", "value"}
    missing = required - set(series.columns)
    if missing:
        raise ValueError(f"Series sheet missing columns: {sorted(missing)}")

    n_obs, n_trans, n_dups, n_resolution = build_n_series(series)

    n_ids = set(n_obs["Patient_ID"])
    nt_ids = set(n_trans["Patient_ID"])

    eco["has_n_observation"] = eco["Patient_ID"].isin(n_ids)
    eco["has_n_transition"] = eco["Patient_ID"].isin(nt_ids)

    n_model_eco = eco.loc[
        eco["has_legacy_id"]
        & eco["has_k2_context"]
        & eco["has_n_transition"]
    ].copy()

    n_final = n_trans.merge(
        n_model_eco[
            [
                "Patient_ID", "participant_id", "project", "diagnosis",
                "ecological_context_k2", "cluster",
                "TNK", "B", "Myeloid"
            ]
        ],
        on="Patient_ID",
        how="inner",
        validate="many_to_one",
    )

    n_cols = [
        "Patient_ID", "participant_id", "project", "diagnosis",
        "ecological_context_k2", "cluster",
        "TNK", "B", "Myeloid",
        "series", "time", "n_prev", "n", "dt"
    ]
    n_final = n_final[n_cols].sort_values(
        ["Patient_ID", "time"]
    ).reset_index(drop=True)
    n_final.to_csv(n_dir / "wave2_n_transition_table.csv", index=False)

    nps = n_participant_summary(n_obs, n_trans)
    nps = nps.merge(
        n_model_eco[
            [
                "Patient_ID", "participant_id", "project", "diagnosis",
                "ecological_context_k2", "TNK", "B", "Myeloid"
            ]
        ],
        on="Patient_ID",
        how="inner",
        validate="one_to_one",
    )
    nps.to_csv(n_dir / "wave2_n_participant_summary.csv", index=False)

    # State counts.
    state_counts = (
        n_final["n"].value_counts()
        .reindex([0,1,2], fill_value=0)
        .rename_axis("n_state")
        .reset_index(name="count")
    )
    state_counts["proportion"] = state_counts["count"] / state_counts["count"].sum()
    state_counts.to_csv(n_dir / "wave2_n_state_counts.csv", index=False)

    # Transition matrix.
    mat = pd.crosstab(
        n_final["n_prev"],
        n_final["n"]
    ).reindex(index=[0,1,2], columns=[0,1,2], fill_value=0)
    mat.index.name = "n_prev"
    mat.columns.name = "n_next"
    mat.to_csv(n_dir / "wave2_n_transition_matrix_counts.csv")

    rowprop = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)
    rowprop.to_csv(n_dir / "wave2_n_transition_matrix_row_proportions.csv")

    # Context support.
    context_support = (
        n_final[["Patient_ID","ecological_context_k2"]]
        .drop_duplicates()
        .groupby("ecological_context_k2")
        .size()
        .rename("n_participants")
        .reset_index()
    )
    tcounts = (
        n_final.groupby("ecological_context_k2")
        .size()
        .rename("n_transitions")
        .reset_index()
    )
    context_support = context_support.merge(tcounts, on="ecological_context_k2")
    context_support.to_csv(n_dir / "wave2_n_context_support.csv", index=False)

    # Exclusion audit.
    exclusions = []
    for _, r in eco.loc[~eco["has_legacy_id"]].iterrows():
        exclusions.append({
            "participant_id": r["participant_id"],
            "Patient_ID": np.nan,
            "reason": "No legacy P1-P100 mapping",
        })
    for _, r in eco.loc[
        eco["has_legacy_id"] & ~eco["has_n_observation"]
    ].iterrows():
        exclusions.append({
            "participant_id": r["participant_id"],
            "Patient_ID": r["Patient_ID"],
            "reason": "No n-series observation",
        })
    for _, r in eco.loc[
        eco["has_legacy_id"] & eco["has_n_observation"] & ~eco["has_n_transition"]
    ].iterrows():
        exclusions.append({
            "participant_id": r["participant_id"],
            "Patient_ID": r["Patient_ID"],
            "reason": "Fewer than two valid n time points / no positive-dt transition",
        })
    pd.DataFrame(exclusions).drop_duplicates().to_csv(
        n_dir / "wave2_n_exclusions.csv", index=False
    )

    n_dups.to_csv(n_dir / "wave2_n_duplicate_time_audit.csv", index=False)
    n_resolution.to_csv(
        n_dir / "wave2_n_duplicate_resolution_audit.csv", index=False
    )

    # ============================================================
    # B. TRANSFORMED-x SENSITIVITY
    # ============================================================
    x = pd.read_csv(x_path)

    # Ensure canonical x table integrity.
    if not x["series"].astype(str).str.lower().eq("x").all():
        raise ValueError("Canonical x input contains non-x rows.")

    mapping_rows = []
    transformed_summary_rows = []

    unique_raw_vals = sorted(
        set(x["x"].astype(float).tolist())
        | set(x["x_prev"].astype(float).tolist())
    )

    for eps in EPSILONS:
        z_x, clipped_x = logit_clip(x["x"].to_numpy(float), eps)
        z_prev, clipped_prev = logit_clip(x["x_prev"].to_numpy(float), eps)

        xt = x.copy()
        xt["x_raw"] = xt["x"]
        xt["x_prev_raw"] = xt["x_prev"]
        xt["x_clipped"] = clipped_x
        xt["x_prev_clipped"] = clipped_prev
        xt["x_logit"] = z_x
        xt["x_prev_logit"] = z_prev
        xt["epsilon"] = eps

        tag = eps_tag(eps)
        xt.to_csv(
            tx_dir / f"wave2_x_logit_eps_{tag}_transition_table.csv",
            index=False
        )

        transformed_summary_rows.append({
            "epsilon": eps,
            "n_participants": int(xt["Patient_ID"].nunique()),
            "n_transitions": int(len(xt)),
            "n_x_values_clipped_low": int(np.sum(x["x"].to_numpy(float) < eps)),
            "n_x_values_clipped_high": int(np.sum(x["x"].to_numpy(float) > 1-eps)),
            "n_xprev_values_clipped_low": int(np.sum(x["x_prev"].to_numpy(float) < eps)),
            "n_xprev_values_clipped_high": int(np.sum(x["x_prev"].to_numpy(float) > 1-eps)),
            "x_logit_min": float(np.min(z_x)),
            "x_logit_max": float(np.max(z_x)),
            "x_logit_mean": float(np.mean(z_x)),
            "x_logit_sd": float(np.std(z_x, ddof=1)),
            "xprev_logit_min": float(np.min(z_prev)),
            "xprev_logit_max": float(np.max(z_prev)),
        })

        for raw in unique_raw_vals:
            z, clipped = logit_clip(np.array([raw]), eps)
            mapping_rows.append({
                "epsilon": eps,
                "raw_value": raw,
                "clipped_value": float(clipped[0]),
                "logit_value": float(z[0]),
            })

    pd.DataFrame(transformed_summary_rows).to_csv(
        tx_dir / "wave2_x_logit_summary.csv", index=False
    )
    pd.DataFrame(mapping_rows).to_csv(
        tx_dir / "wave2_x_logit_value_mapping.csv", index=False
    )

    # ============================================================
    # RUN SUMMARY
    # ============================================================
    summary = {
        "n_endpoint": {
            "n_model_participants": int(n_final["Patient_ID"].nunique()),
            "n_model_transitions": int(len(n_final)),
            "state_counts": {
                str(int(r["n_state"])): int(r["count"])
                for _, r in state_counts.iterrows()
            },
            "n_duplicate_patient_time_rows": int(len(n_dups)),
            "n_same_time_groups_resolved": int(len(n_resolution)),
        },
        "transformed_x": {
            "epsilons": EPSILONS,
            "n_participants_each": int(x["Patient_ID"].nunique()),
            "n_transitions_each": int(len(x)),
            "transform": "logit(clip(x, epsilon, 1-epsilon))",
            "both_x_and_x_prev_transformed": True,
        },
    }

    with open(outroot / "wave2_secondary_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] Secondary endpoint input preparation complete.")
    print("Outputs:", outroot)

    print("\nOrdinal n support:")
    print(json.dumps(summary["n_endpoint"], indent=2))

    print("\nn state counts:")
    print(state_counts.to_string(index=False))

    print("\nn transition matrix (counts):")
    print(mat.to_string())

    print("\nn context support:")
    print(context_support.to_string(index=False))

    print("\nTransformed-x summary:")
    print(pd.DataFrame(transformed_summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
