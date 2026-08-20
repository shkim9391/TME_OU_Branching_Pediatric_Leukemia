from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az


MODEL_SPECS = {
    "M_0": [],
    "M_B": ["B_z"],
    "M_M": ["Myeloid_z"],
    "M_E": ["B_z", "Myeloid_z"],
    "M_C": ["Context_z"],
    "M_EC": ["B_z", "Myeloid_z", "Context_z"],
    "M_S": ["ShuffledContext_z"],
}


def ou_transition_mean_var(y_prev, mu_i, theta_i, sigma_proc, dt):
    """
    Exact OU transition:
      Y_t | Y_{t-dt} ~ Normal(mean, variance)

    mean = mu + (y_prev - mu) exp(-theta dt)
    var  = sigma^2/(2 theta) * [1 - exp(-2 theta dt)]
    """
    decay = pt.exp(-theta_i * dt)
    mean = mu_i + (y_prev - mu_i) * decay
    var = (
        (sigma_proc ** 2)
        / (2.0 * theta_i)
        * (1.0 - pt.exp(-2.0 * theta_i * dt))
    )
    return mean, var


def safe_zscore(values: pd.Series, name: str):
    x = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Predictor {name} has zero/non-finite standard deviation.")
    z = (x - mean) / sd
    return z, mean, sd


def audit_participant_constancy(df: pd.DataFrame, columns):
    bad = []
    for col in columns:
        nunique = df.groupby("Patient_ID")[col].nunique(dropna=False)
        offenders = nunique[nunique > 1]
        if len(offenders):
            bad.append((col, offenders.index.tolist()))
    if bad:
        msg = "\n".join(
            f"{col}: {ids[:20]}" for col, ids in bad
        )
        raise ValueError(
            "Participant-level covariates are not constant across transitions:\n" + msg
        )


def build_design_table(transitions: pd.DataFrame, shuffle_seed: int):
    needed = [
        "Patient_ID", "participant_id", "project", "diagnosis",
        "ecological_context_k2", "cluster", "TNK", "B", "Myeloid",
    ]
    missing = [c for c in needed if c not in transitions.columns]
    if missing:
        raise ValueError(f"Missing required transition columns: {missing}")

    audit_participant_constancy(transitions, needed[1:])

    design = (
        transitions[needed]
        .drop_duplicates("Patient_ID")
        .sort_values("Patient_ID")
        .reset_index(drop=True)
    )

    # Continuous composition.
    design["B_z"], b_mean, b_sd = safe_zscore(design["B"], "B")
    design["Myeloid_z"], m_mean, m_sd = safe_zscore(
        design["Myeloid"], "Myeloid"
    )

    # Context coding.
    context_numeric = design["cluster"].map({"C1": 0.0, "C2": 1.0})
    if context_numeric.isna().any():
        bad = design.loc[context_numeric.isna(), "cluster"].unique()
        raise ValueError(f"Unexpected context labels: {bad}")

    design["Context_numeric"] = context_numeric.astype(float)
    design["Context_z"], c_mean, c_sd = safe_zscore(
        design["Context_numeric"], "Context_numeric"
    )

    # Participant-level shuffled control, preserving exact counts.
    rng = np.random.default_rng(shuffle_seed)
    shuffled = rng.permutation(design["Context_numeric"].to_numpy())
    design["ShuffledContext_numeric"] = shuffled.astype(float)

    # Use original context centering/scaling for shuffled context because the
    # permutation preserves the exact value distribution.
    design["ShuffledContext_z"] = (
        design["ShuffledContext_numeric"] - c_mean
    ) / c_sd

    scaler = pd.DataFrame([
        {
            "predictor": "B",
            "mean": b_mean,
            "sd_population": b_sd,
            "transformed_column": "B_z",
        },
        {
            "predictor": "Myeloid",
            "mean": m_mean,
            "sd_population": m_sd,
            "transformed_column": "Myeloid_z",
        },
        {
            "predictor": "Context_numeric",
            "mean": c_mean,
            "sd_population": c_sd,
            "transformed_column": "Context_z",
        },
        {
            "predictor": "ShuffledContext_numeric",
            "mean": c_mean,
            "sd_population": c_sd,
            "transformed_column": "ShuffledContext_z",
        },
    ])

    return design, scaler


def build_model(
    model_name: str,
    active_predictors,
    transitions: pd.DataFrame,
    design: pd.DataFrame,
):
    patient_ids = design["Patient_ID"].tolist()
    patient_map = {pid: i for i, pid in enumerate(patient_ids)}

    pat_idx = transitions["Patient_ID"].map(patient_map)
    if pat_idx.isna().any():
        raise ValueError(f"{model_name}: transition patient absent from design table.")
    pat_idx = pat_idx.to_numpy(dtype="int64")

    y = transitions["x"].to_numpy(dtype="float64")
    y_prev = transitions["x_prev"].to_numpy(dtype="float64")
    dt = transitions["dt"].to_numpy(dtype="float64")

    if not np.all(np.isfinite(y)):
        raise ValueError(f"{model_name}: non-finite x values.")
    if not np.all(np.isfinite(y_prev)):
        raise ValueError(f"{model_name}: non-finite x_prev values.")
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError(f"{model_name}: invalid dt values.")

    coords = {
        "patient": patient_ids,
        "obs": np.arange(len(transitions)),
    }

    X = None
    if active_predictors:
        X = design[active_predictors].to_numpy(dtype="float64")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"{model_name}: non-finite predictors.")
        coords["predictor"] = active_predictors

    with pm.Model(coords=coords) as model:
        # -------------------------------------------------------------
        # Shared priors across every ablation
        # -------------------------------------------------------------
        mu_0 = pm.Normal("mu_0", mu=0.0, sigma=1.0)
        theta_0 = pm.Normal("theta_0", mu=0.0, sigma=1.0)
        sigma_proc = pm.HalfNormal("sigma_proc", sigma=1.0)

        if active_predictors:
            beta_mu = pm.Normal(
                "beta_mu",
                mu=0.0,
                sigma=0.35,
                dims="predictor",
            )
            beta_theta = pm.Normal(
                "beta_theta",
                mu=0.0,
                sigma=0.35,
                dims="predictor",
            )

            mu_linear = mu_0 + pt.dot(X, beta_mu)
            log_theta_linear = theta_0 + pt.dot(X, beta_theta)
        else:
            # Explicit patient vectors keep posterior output structure
            # consistent with covariate models.
            mu_linear = pt.repeat(mu_0, len(patient_ids))
            log_theta_linear = pt.repeat(theta_0, len(patient_ids))

        mu_pat = pm.Deterministic(
            "mu_pat",
            mu_linear,
            dims="patient",
        )

        log_theta_pat = pm.Deterministic(
            "log_theta_pat",
            log_theta_linear,
            dims="patient",
        )

        theta_pat = pm.Deterministic(
            "theta_pat",
            pt.exp(log_theta_pat) + 1e-6,
            dims="patient",
        )

        mu_obs = mu_pat[pat_idx]
        theta_obs = theta_pat[pat_idx]

        mean_tr, var_tr = ou_transition_mean_var(
            y_prev=y_prev,
            mu_i=mu_obs,
            theta_i=theta_obs,
            sigma_proc=sigma_proc,
            dt=dt,
        )
        var_tr = pt.clip(var_tr, 1e-12, np.inf)

        pm.Normal(
            "y_obs",
            mu=mean_tr,
            sigma=pt.sqrt(var_tr),
            observed=y,
            dims="obs",
        )

    return model


def diagnostic_summary(model_name: str, idata: az.InferenceData):
    divergences = int(idata.sample_stats["diverging"].sum().values)

    var_names = ["mu_0", "theta_0", "sigma_proc"]
    if "beta_mu" in idata.posterior:
        var_names += ["beta_mu", "beta_theta"]

    summ = az.summary(
        idata,
        var_names=var_names,
        round_to=None,
    )

    max_rhat = (
        float(np.nanmax(summ["r_hat"].to_numpy()))
        if "r_hat" in summ.columns else np.nan
    )
    min_ess_bulk = (
        float(np.nanmin(summ["ess_bulk"].to_numpy()))
        if "ess_bulk" in summ.columns else np.nan
    )
    min_ess_tail = (
        float(np.nanmin(summ["ess_tail"].to_numpy()))
        if "ess_tail" in summ.columns else np.nan
    )

    return {
        "model": model_name,
        "divergences": divergences,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
    }, summ


def loo_and_pareto(model_name: str, idata: az.InferenceData):
    loo = az.loo(idata, pointwise=True)

    pareto = np.asarray(loo.pareto_k.values, dtype=float)
    pareto_rows = pd.DataFrame({
        "model": model_name,
        "obs_index": np.arange(len(pareto)),
        "pareto_k": pareto,
    })

    summary = {
        "model": model_name,
        "elpd_loo": float(loo.elpd_loo),
        "se_elpd_loo": float(loo.se),
        "p_loo": float(loo.p_loo),
        "n_pareto_gt_0_5": int(np.sum(pareto > 0.5)),
        "n_pareto_gt_0_7": int(np.sum(pareto > 0.7)),
        "n_pareto_gt_1_0": int(np.sum(pareto > 1.0)),
        "max_pareto_k": float(np.max(pareto)),
    }
    return summary, pareto_rows


def save_idata_atomic(idata, final_path: Path):
    tmp = final_path.with_name(final_path.stem + "_tmp.nc")
    if tmp.exists():
        tmp.unlink()
    az.to_netcdf(idata, tmp, engine="netcdf4")
    if final_path.exists():
        final_path.unlink()
    tmp.replace(final_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/TME_OU_Branching",
    )
    ap.add_argument(
        "--input",
        default="revision_wave2_inputs/wave2_x_transition_table.csv",
    )
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=3000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--shuffle-seed", type=int, default=20260820)
    ap.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_SPECS.keys()),
        choices=list(MODEL_SPECS.keys()),
        help="Subset of models to fit. Default: all.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    input_path = root / args.input
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    outroot = root / "revision_wave2_models"
    design_dir = outroot / "design"
    idata_dir = outroot / "idata"
    diag_dir = outroot / "diagnostics"
    for d in [design_dir, idata_dir, diag_dir]:
        d.mkdir(parents=True, exist_ok=True)

    transitions = pd.read_csv(input_path)

    required = [
        "Patient_ID", "participant_id", "project", "diagnosis",
        "ecological_context_k2", "cluster",
        "TNK", "B", "Myeloid",
        "series", "time", "x_prev", "x", "dt",
    ]
    missing = [c for c in required if c not in transitions.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    if not transitions["series"].astype(str).str.lower().eq("x").all():
        raise ValueError("Input table contains non-x series.")

    transitions["Patient_ID"] = transitions["Patient_ID"].astype(str).str.strip()

    # Build one participant design table shared by every model.
    design, scaler = build_design_table(
        transitions,
        shuffle_seed=args.shuffle_seed,
    )
    design.to_csv(
        design_dir / "wave2_participant_design_matrix.csv",
        index=False,
    )
    scaler.to_csv(
        design_dir / "wave2_predictor_scaler_parameters.csv",
        index=False,
    )

    design[
        [
            "Patient_ID", "participant_id",
            "Context_numeric", "ShuffledContext_numeric",
        ]
    ].to_csv(
        design_dir / "wave2_shuffled_context_mapping.csv",
        index=False,
    )

    model_specs_df = pd.DataFrame([
        {
            "model": name,
            "active_predictors": "|".join(preds) if preds else "None",
            "n_predictors": len(preds),
        }
        for name, preds in MODEL_SPECS.items()
    ])
    model_specs_df.to_csv(
        design_dir / "wave2_model_specifications.csv",
        index=False,
    )

    run_config = {
        "input": str(input_path),
        "n_participants": int(design["Patient_ID"].nunique()),
        "n_transitions": int(len(transitions)),
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "seed": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "models_requested": args.models,
        "endpoint": "x",
    }
    with open(outroot / "wave2_full_ablation_run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    diagnostics = []
    loo_summaries = []
    pareto_tables = []

    for model_offset, model_name in enumerate(args.models):
        predictors = MODEL_SPECS[model_name]
        print("\n" + "=" * 78)
        print(f"[INFO] Fitting {model_name}: predictors = {predictors or 'None'}")
        print("=" * 78)

        model = build_model(
            model_name=model_name,
            active_predictors=predictors,
            transitions=transitions,
            design=design,
        )

        with model:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                target_accept=args.target_accept,
                random_seed=args.seed + model_offset,
                idata_kwargs={"log_likelihood": True},
                return_inferencedata=True,
            )

            idata = pm.sample_posterior_predictive(
                idata,
                var_names=["y_obs"],
                random_seed=args.seed + 1000 + model_offset,
                extend_inferencedata=True,
            )

        out_nc = idata_dir / f"{model_name}_ou_x_idata_ppc.nc"
        print(f"[INFO] Saving {model_name} idata: {out_nc}")
        save_idata_atomic(idata, out_nc)

        diag_row, posterior_summary = diagnostic_summary(model_name, idata)
        diagnostics.append(diag_row)
        posterior_summary.to_csv(
            diag_dir / f"{model_name}_posterior_summary.csv"
        )

        try:
            loo_row, pareto_df = loo_and_pareto(model_name, idata)
            loo_summaries.append(loo_row)
            pareto_tables.append(pareto_df)

            print(
                f"[CHECK] {model_name}: divergences={diag_row['divergences']}, "
                f"max R-hat={diag_row['max_rhat']:.4f}, "
                f"ELPD-LOO={loo_row['elpd_loo']:.3f}, "
                f"max Pareto-k={loo_row['max_pareto_k']:.3f}"
            )
        except Exception as e:
            print(f"[WARN] LOO failed for {model_name}: {e}")

        # Free memory before the next full model.
        del idata
        del model
        gc.collect()

    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(
        diag_dir / "wave2_ablation_diagnostics.csv",
        index=False,
    )

    if loo_summaries:
        loo_df = pd.DataFrame(loo_summaries)
        loo_df.to_csv(
            diag_dir / "wave2_ablation_loo_summary.csv",
            index=False,
        )

    if pareto_tables:
        pareto_all = pd.concat(pareto_tables, ignore_index=True)

        # Attach transition identities so every influential observation is traceable.
        obs_meta = transitions[
            [
                "Patient_ID", "participant_id", "ecological_context_k2",
                "time", "x_prev", "x", "dt"
            ]
        ].reset_index(drop=True)
        obs_meta["obs_index"] = np.arange(len(obs_meta))

        pareto_all = pareto_all.merge(
            obs_meta,
            on="obs_index",
            how="left",
            validate="many_to_one",
        )
        pareto_all.to_csv(
            diag_dir / "wave2_ablation_pareto_k.csv",
            index=False,
        )

    print("\n[OK] Full-data Wave 2 ablation fitting complete.")
    print("Outputs:", outroot)
    print("\nDiagnostics:")
    print(diagnostics_df.to_string(index=False))

    if loo_summaries:
        print("\nTransition-level PSIS-LOO summary:")
        print(pd.DataFrame(loo_summaries).to_string(index=False))
        print(
            "\n[NOTE] These are transition-level PSIS-LOO diagnostics only. "
            "They do NOT replace the planned true leave-one-participant-out "
            "validation."
        )


if __name__ == "__main__":
    main()
