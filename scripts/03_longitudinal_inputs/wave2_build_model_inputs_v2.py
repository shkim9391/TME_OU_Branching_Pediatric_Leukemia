import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


PRIMARY_COMPONENTS = ["TNK", "B", "Myeloid"]


def normalize_id_series(s):
    return s.astype(str).str.strip()


def clean_diagnosis(x):
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    mapping = {
        "B-cell acute lymphoblastic leukemia": "B-ALL",
        "T-cell acute lymphoblastic leukemia": "T-ALL",
        "Early T-cell precursor T-cell acute lymphoblastic leukemia": "ETP-ALL",
        "Acute myeloid leukemia": "AML",
        "Mixed phenotype acute leukemia": "MPAL",
    }
    return mapping.get(s, s if s else "Unknown")


def build_x_transitions(series_df):
    df = series_df.copy()
    df["Patient_ID"] = normalize_id_series(df["Patient_ID"])
    df["series"] = df["series"].astype(str).str.strip().str.lower()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Primary endpoint: x only.
    df = df.loc[df["series"].eq("x")].copy()
    df = df.dropna(subset=["Patient_ID", "t", "value"])
    df = df.sort_values(["Patient_ID", "t"]).reset_index(drop=True)

    # Audit duplicate time points before transition creation.
    dup_mask = df.duplicated(["Patient_ID", "t"], keep=False)
    duplicates = df.loc[dup_mask].copy()

    # Resolve same-time x collisions deterministically.
    #
    # The historical x coding uses:
    #   diagnosis/relapse = 1.0
    #   generic longitudinal default = 0.1
    #
    # In the current workbook, conflicting duplicate times occur at t=0 and
    # consist of 1.0 plus 0.1. For baseline collisions of exactly this form,
    # retain 1.0 (diagnosis state). Exact duplicates are collapsed.
    # Any other conflicting same-time values remain an error and require
    # manual review.
    resolution_rows = []
    resolved_groups = []

    for (pid, tt), g in df.groupby(["Patient_ID", "t"], sort=False):
        vals = sorted(pd.unique(g["value"].dropna()).tolist())

        if len(vals) <= 1:
            row = g.iloc[0].copy()
            resolved_groups.append(row)
            if len(g) > 1:
                resolution_rows.append({
                    "Patient_ID": pid,
                    "t": tt,
                    "original_values": "|".join(map(str, sorted(g["value"].tolist()))),
                    "resolved_value": float(row["value"]),
                    "resolution_rule": "collapsed_exact_duplicates",
                    "n_rows_original": int(len(g)),
                })
            continue

        # Explicitly supported baseline collision: diagnosis=1.0 vs generic=0.1
        if np.isclose(tt, 0.0) and set(np.round(vals, 10)) == {0.1, 1.0}:
            row = g.loc[np.isclose(g["value"], 1.0)].iloc[0].copy()
            resolved_groups.append(row)
            resolution_rows.append({
                "Patient_ID": pid,
                "t": tt,
                "original_values": "|".join(map(str, sorted(g["value"].tolist()))),
                "resolved_value": 1.0,
                "resolution_rule": "baseline_diagnosis_over_generic_longitudinal",
                "n_rows_original": int(len(g)),
            })
            continue

        raise ValueError(
            f"Unsupported conflicting x values at Patient_ID={pid}, t={tt}: {vals}. "
            "Manual review is required."
        )

    df = pd.DataFrame(resolved_groups).sort_values(["Patient_ID", "t"]).reset_index(drop=True)
    duplicate_resolution = pd.DataFrame(resolution_rows)

    df["x_prev"] = df.groupby("Patient_ID")["value"].shift(1)
    df["dt"] = df.groupby("Patient_ID")["t"].diff()

    transitions = df.dropna(subset=["x_prev", "dt"]).copy()
    transitions = transitions.loc[transitions["dt"] > 0].copy()
    transitions = transitions.rename(columns={"t": "time", "value": "x"})
    transitions["series"] = "x"

    return df, transitions, duplicates, duplicate_resolution


def participant_sampling_summary(x_obs, x_trans):
    rows = []

    for pid, g in x_obs.groupby("Patient_ID"):
        gt = x_trans.loc[x_trans["Patient_ID"].eq(pid)]
        times = np.sort(g["t"].unique())
        follow = float(times.max() - times.min()) if len(times) >= 2 else 0.0

        dts = gt["dt"].to_numpy(float)
        if len(dts) >= 2 and np.mean(dts) > 0:
            interval_cv = float(np.std(dts, ddof=1) / np.mean(dts))
        else:
            interval_cv = np.nan

        rows.append({
            "Patient_ID": pid,
            "n_x_observations": int(len(g)),
            "n_unique_times": int(len(times)),
            "n_x_transitions": int(len(gt)),
            "first_time": float(times.min()),
            "last_time": float(times.max()),
            "followup_years": follow,
            "median_dt": float(np.median(dts)) if len(dts) else np.nan,
            "mean_dt": float(np.mean(dts)) if len(dts) else np.nan,
            "min_dt": float(np.min(dts)) if len(dts) else np.nan,
            "max_dt": float(np.max(dts)) if len(dts) else np.nan,
            "interval_cv": interval_cv,
            "x_min": float(g["value"].min()),
            "x_max": float(g["value"].max()),
            "x_mean": float(g["value"].mean()),
        })

    return pd.DataFrame(rows)


def overall_sampling_summary(x_obs, x_trans, supported_participants):
    per = participant_sampling_summary(x_obs, x_trans)
    per = per.loc[per["Patient_ID"].isin(supported_participants)].copy()

    vals = {
        "n_supported_participants": int(per["Patient_ID"].nunique()),
        "n_x_observations": int(
            x_obs.loc[x_obs["Patient_ID"].isin(supported_participants)].shape[0]
        ),
        "n_x_transitions": int(
            x_trans.loc[x_trans["Patient_ID"].isin(supported_participants)].shape[0]
        ),
        "median_unique_times_per_participant": float(per["n_unique_times"].median()),
        "min_unique_times_per_participant": int(per["n_unique_times"].min()),
        "max_unique_times_per_participant": int(per["n_unique_times"].max()),
        "median_followup_years": float(per["followup_years"].median()),
        "min_followup_years": float(per["followup_years"].min()),
        "max_followup_years": float(per["followup_years"].max()),
        "median_interval_cv": float(per["interval_cv"].dropna().median())
            if per["interval_cv"].notna().any() else np.nan,
        "median_dt": float(
            x_trans.loc[
                x_trans["Patient_ID"].isin(supported_participants), "dt"
            ].median()
        ),
        "q25_dt": float(
            x_trans.loc[
                x_trans["Patient_ID"].isin(supported_participants), "dt"
            ].quantile(0.25)
        ),
        "q75_dt": float(
            x_trans.loc[
                x_trans["Patient_ID"].isin(supported_participants), "dt"
            ].quantile(0.75)
        ),
        "q90_dt": float(
            x_trans.loc[
                x_trans["Patient_ID"].isin(supported_participants), "dt"
            ].quantile(0.90)
        ),
        "q95_dt": float(
            x_trans.loc[
                x_trans["Patient_ID"].isin(supported_participants), "dt"
            ].quantile(0.95)
        ),
        "max_dt": float(
            x_trans.loc[
                x_trans["Patient_ID"].isin(supported_participants), "dt"
            ].max()
        ),
    }
    return pd.DataFrame([vals]), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/TME_OU_Branching"
    )
    ap.add_argument(
        "--primary-composition",
        default="revision_wave1_id_ball/primary_composition_with_legacy_ids.csv"
    )
    ap.add_argument(
        "--raw-k2-assignments",
        default="revision_wave1_context_stability/cluster_assignments_raw_k2.csv"
    )
    ap.add_argument(
        "--longitudinal",
        default="kmt2a_longitudinal_clean.xlsx"
    )
    ap.add_argument(
        "--mapping",
        default="patient_id_mapping.csv"
    )
    args = ap.parse_args()

    root = Path(args.root)
    outdir = root / "revision_wave2_inputs"
    outdir.mkdir(parents=True, exist_ok=True)

    primary_path = root / args.primary_composition
    assign_path = root / args.raw_k2_assignments
    long_path = root / args.longitudinal
    map_path = root / args.mapping

    for p in [primary_path, assign_path, long_path, map_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    # --------------------------------------------------
    # 1. Load ecological composition with legacy IDs
    # --------------------------------------------------
    eco = pd.read_csv(primary_path)

    # Normalize columns.
    if "participant_id" not in eco.columns and "Patient_ID" in eco.columns:
        raise ValueError(
            "primary_composition_with_legacy_ids.csv should contain both "
            "participant_id and Patient_ID."
        )

    eco["participant_id"] = normalize_id_series(eco["participant_id"])
    eco["Patient_ID"] = normalize_id_series(eco["Patient_ID"])
    eco["diagnosis"] = eco["diagnosis"].map(clean_diagnosis)

    # Keep only rows with a resolved legacy ID.
    eco["has_legacy_id"] = eco["Patient_ID"].notna() & eco["Patient_ID"].ne("nan")

    # --------------------------------------------------
    # 2. Load raw k=2 assignments and attach legacy IDs
    # --------------------------------------------------
    ass = pd.read_csv(assign_path)
    if "participant_id" not in ass.columns:
        if "Patient_ID" in ass.columns:
            ass = ass.rename(columns={"Patient_ID": "participant_id"})
        else:
            raise ValueError("k=2 assignment file lacks participant identifier.")

    ass["participant_id"] = normalize_id_series(ass["participant_id"])
    if "cluster" not in ass.columns:
        raise ValueError("k=2 assignment file lacks 'cluster' column.")

    context_map = {
        "C1": "TNK-dominant",
        "C2": "B/Myeloid-shifted",
    }
    ass["ecological_context_k2"] = ass["cluster"].map(context_map)
    if ass["ecological_context_k2"].isna().any():
        bad = ass.loc[ass["ecological_context_k2"].isna(), "cluster"].unique()
        raise ValueError(f"Unexpected k=2 cluster labels: {bad}")

    eco = eco.merge(
        ass[["participant_id", "cluster", "ecological_context_k2"]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------
    # 3. Define corrected continuous ecological covariates
    # --------------------------------------------------
    source_map = {
        "TNK": "comp_TNK",
        "B": "comp_B",
        "Myeloid": "comp_Myeloid",
    }
    for outcol, srccol in source_map.items():
        if srccol not in eco.columns:
            raise ValueError(f"Missing corrected composition column: {srccol}")
        eco[outcol] = pd.to_numeric(eco[srccol], errors="coerce")

    # Ensure simplex sums correctly.
    eco["composition_sum"] = eco[PRIMARY_COMPONENTS].sum(axis=1)
    bad_sum = np.abs(eco["composition_sum"] - 1.0) > 1e-6
    if bad_sum.any():
        raise ValueError(
            "Corrected TNK/B/Myeloid composition does not sum to 1 for some participants."
        )

    # --------------------------------------------------
    # 4. Load longitudinal x data and create transitions
    # --------------------------------------------------
    series = pd.read_excel(long_path, sheet_name="Series")
    required = {"Patient_ID", "series", "t", "value"}
    missing = required - set(series.columns)
    if missing:
        raise ValueError(f"Longitudinal Series sheet missing columns: {sorted(missing)}")

    x_obs, x_trans, duplicates, duplicate_resolution = build_x_transitions(series)

    # --------------------------------------------------
    # 5. Crosswalk audit
    # --------------------------------------------------
    x_ids = set(x_obs["Patient_ID"])
    trans_ids = set(x_trans["Patient_ID"])
    eco_ids = set(eco.loc[eco["has_legacy_id"], "Patient_ID"])

    eco["has_x_observation"] = eco["Patient_ID"].isin(x_ids)
    eco["has_x_transition"] = eco["Patient_ID"].isin(trans_ids)
    eco["has_k2_context"] = eco["ecological_context_k2"].notna()

    crosswalk_cols = [
        "participant_id", "Patient_ID", "project", "diagnosis", "subdiagnosis",
        "TNK", "B", "Myeloid", "cluster", "ecological_context_k2",
        "has_legacy_id", "has_x_observation", "has_x_transition", "has_k2_context"
    ]
    crosswalk_cols = [c for c in crosswalk_cols if c in eco.columns]
    eco[crosswalk_cols].to_csv(outdir / "wave2_crosswalk_audit.csv", index=False)

    # --------------------------------------------------
    # 6. Build modeling transition table
    # --------------------------------------------------
    model_eco = eco.loc[
        eco["has_legacy_id"]
        & eco["has_k2_context"]
        & eco["has_x_transition"]
    ].copy()

    trans = x_trans.merge(
        model_eco[
            [
                "Patient_ID", "participant_id", "diagnosis",
                "ecological_context_k2", "cluster",
                "TNK", "B", "Myeloid", "project"
            ]
        ],
        on="Patient_ID",
        how="inner",
        validate="many_to_one",
    )

    final_cols = [
        "Patient_ID",
        "participant_id",
        "project",
        "diagnosis",
        "ecological_context_k2",
        "cluster",
        "TNK",
        "B",
        "Myeloid",
        "series",
        "time",
        "x_prev",
        "x",
        "dt",
    ]
    trans = trans[final_cols].sort_values(
        ["Patient_ID", "time"]
    ).reset_index(drop=True)

    trans.to_csv(outdir / "wave2_x_transition_table.csv", index=False)

    # --------------------------------------------------
    # 7. Participant-level sampling summaries
    # --------------------------------------------------
    supported_ids = set(trans["Patient_ID"])
    overall_summary, per_summary = overall_sampling_summary(
        x_obs, x_trans, supported_ids
    )

    # Attach ecology/context to participant sampling table.
    per_summary = per_summary.merge(
        model_eco[
            [
                "Patient_ID", "participant_id", "project", "diagnosis",
                "ecological_context_k2", "TNK", "B", "Myeloid"
            ]
        ],
        on="Patient_ID",
        how="inner",
        validate="one_to_one",
    )
    per_summary.to_csv(outdir / "wave2_x_participant_summary.csv", index=False)
    overall_summary.to_csv(outdir / "wave2_x_sampling_summary.csv", index=False)

    # --------------------------------------------------
    # 8. Exclusion audit
    # --------------------------------------------------
    exclusions = []

    # Ecological participants without legacy mapping
    for _, r in eco.loc[~eco["has_legacy_id"]].iterrows():
        exclusions.append({
            "participant_id": r.get("participant_id"),
            "Patient_ID": np.nan,
            "reason": "No legacy P1-P100 mapping",
        })

    # Ecological participants mapped but without x observations/transitions
    for _, r in eco.loc[
        eco["has_legacy_id"] & ~eco["has_x_observation"]
    ].iterrows():
        exclusions.append({
            "participant_id": r.get("participant_id"),
            "Patient_ID": r.get("Patient_ID"),
            "reason": "No x-series observation",
        })

    for _, r in eco.loc[
        eco["has_legacy_id"] & eco["has_x_observation"] & ~eco["has_x_transition"]
    ].iterrows():
        exclusions.append({
            "participant_id": r.get("participant_id"),
            "Patient_ID": r.get("Patient_ID"),
            "reason": "Fewer than two valid x time points / no positive-dt transition",
        })

    for _, r in eco.loc[
        eco["has_legacy_id"] & eco["has_x_transition"] & ~eco["has_k2_context"]
    ].iterrows():
        exclusions.append({
            "participant_id": r.get("participant_id"),
            "Patient_ID": r.get("Patient_ID"),
            "reason": "No raw k=2 ecological-context assignment",
        })

    exclusions_df = pd.DataFrame(exclusions).drop_duplicates()
    exclusions_df.to_csv(outdir / "wave2_x_exclusions.csv", index=False)

    # --------------------------------------------------
    # 9. Context summary among modeled participants
    # --------------------------------------------------
    context_summary = (
        trans[[
            "Patient_ID", "ecological_context_k2", "TNK", "B", "Myeloid"
        ]]
        .drop_duplicates("Patient_ID")
        .groupby("ecological_context_k2")
        .agg(
            n_participants=("Patient_ID", "nunique"),
            mean_TNK=("TNK", "mean"),
            mean_B=("B", "mean"),
            mean_Myeloid=("Myeloid", "mean"),
        )
        .reset_index()
    )

    trans_counts = (
        trans.groupby("ecological_context_k2")
        .size()
        .rename("n_transitions")
        .reset_index()
    )
    context_summary = context_summary.merge(
        trans_counts, on="ecological_context_k2", how="left"
    )
    context_summary.to_csv(outdir / "wave2_context_summary.csv", index=False)

    # --------------------------------------------------
    # 10. Duplicate-time audit, if any
    # --------------------------------------------------
    duplicates.to_csv(outdir / "wave2_x_duplicate_time_audit.csv", index=False)
    duplicate_resolution.to_csv(
        outdir / "wave2_x_duplicate_resolution_audit.csv", index=False
    )

    # --------------------------------------------------
    # 11. Run summary
    # --------------------------------------------------
    summary = {
        "n_primary_ecological_participants": int(len(eco)),
        "n_with_legacy_id": int(eco["has_legacy_id"].sum()),
        "n_with_x_observation": int(eco["has_x_observation"].sum()),
        "n_with_x_transition": int(eco["has_x_transition"].sum()),
        "n_with_k2_context": int(eco["has_k2_context"].sum()),
        "n_model_participants": int(trans["Patient_ID"].nunique()),
        "n_model_x_transitions": int(len(trans)),
        "n_duplicate_patient_time_rows_in_x": int(len(duplicates)),
        "n_same_time_groups_resolved": int(len(duplicate_resolution)),
        "endpoint": "x",
        "endpoint_description": (
            "Pre-specified semiquantitative leukemia-state proxy from the "
            "longitudinal Series table; modeled separately from n."
        ),
        "continuous_ecological_covariates": PRIMARY_COMPONENTS,
        "coarse_contexts": {
            "C1": "TNK-dominant",
            "C2": "B/Myeloid-shifted",
        },
    }

    with open(outdir / "wave2_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] Wave 2 model-input build complete.")
    print("Outputs:", outdir)
    print("\nRun summary:")
    print(json.dumps(summary, indent=2))

    print("\nContext support among modeled x participants:")
    print(context_summary.to_string(index=False))

    print("\nOverall x sampling summary:")
    print(overall_summary.to_string(index=False))


if __name__ == "__main__":
    main()
