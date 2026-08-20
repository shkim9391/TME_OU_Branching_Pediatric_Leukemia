from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.special import logsumexp


MODEL_SPECS = {
    "X_0": [],
    "X_B": ["B_z"],
    "X_E": ["B_z", "Myeloid_z"],
    "X_C": ["Context_z"],
}


def ou_mean_var_np(y_prev, mu, theta, sigma, dt):
    decay = np.exp(-theta * dt)
    mean = mu + (y_prev - mu) * decay
    var = (sigma**2) / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * dt))
    return mean, np.maximum(var, 1e-12)


def ou_mean_var_pt(y_prev, mu, theta, sigma, dt):
    decay = pt.exp(-theta * dt)
    mean = mu + (y_prev - mu) * decay
    var = (sigma**2) / (2.0 * theta) * (1.0 - pt.exp(-2.0 * theta * dt))
    return mean, pt.clip(var, 1e-12, np.inf)


def normal_logpdf(y, mu, var):
    return -0.5 * (np.log(2.0 * np.pi * var) + ((y - mu) ** 2) / var)


def participant_design(trans):
    cols = [
        "Patient_ID",
        "participant_id",
        "project",
        "diagnosis",
        "cluster",
        "ecological_context_k2",
        "B",
        "Myeloid",
        "TNK",
    ]
    d = (
        trans[cols]
        .drop_duplicates("Patient_ID")
        .sort_values("Patient_ID")
        .reset_index(drop=True)
    )

    d["Context_numeric"] = d["cluster"].map({"C1": 0.0, "C2": 1.0})
    if d["Context_numeric"].isna().any():
        bad = d.loc[d["Context_numeric"].isna(), "cluster"].unique()
        raise ValueError(f"Unexpected context labels: {bad}")

    return d


def safe_scale(train_vals, test_vals, name):
    train_vals = np.asarray(train_vals, dtype=float)
    test_vals = np.asarray(test_vals, dtype=float)

    mean = float(np.mean(train_vals))
    sd = float(np.std(train_vals, ddof=0))

    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"{name}: zero/non-finite training SD.")

    return (
        (train_vals - mean) / sd,
        (test_vals - mean) / sd,
        mean,
        sd,
    )


def prepare_fold_design(full_design, heldout_id):
    train = full_design.loc[full_design["Patient_ID"] != heldout_id].copy()
    test = full_design.loc[full_design["Patient_ID"] == heldout_id].copy()

    if len(test) != 1:
        raise ValueError(
            f"Expected one participant design row for {heldout_id}; got {len(test)}."
        )

    train["B_z"], test_B, b_mean, b_sd = safe_scale(
        train["B"], test["B"], "B"
    )
    test["B_z"] = test_B

    train["Myeloid_z"], test_M, m_mean, m_sd = safe_scale(
        train["Myeloid"], test["Myeloid"], "Myeloid"
    )
    test["Myeloid_z"] = test_M

    train["Context_z"], test_C, c_mean, c_sd = safe_scale(
        train["Context_numeric"], test["Context_numeric"], "Context"
    )
    test["Context_z"] = test_C

    scales = {
        "B_mean": b_mean,
        "B_sd": b_sd,
        "Myeloid_mean": m_mean,
        "Myeloid_sd": m_sd,
        "Context_mean": c_mean,
        "Context_sd": c_sd,
    }

    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        scales,
    )


def build_training_model(predictors, train_trans, train_design):
    pids = train_design["Patient_ID"].tolist()
    pmap = {pid: i for i, pid in enumerate(pids)}

    pat_idx = train_trans["Patient_ID"].map(pmap)
    if pat_idx.isna().any():
        raise ValueError("Training transition patient absent from training design.")
    pat_idx = pat_idx.to_numpy(dtype="int64")

    y = train_trans["x_logit"].to_numpy(dtype=float)
    y_prev = train_trans["x_prev_logit"].to_numpy(dtype=float)
    dt = train_trans["dt"].to_numpy(dtype=float)

    coords = {
        "patient": pids,
        "obs": np.arange(len(train_trans)),
    }

    X = None
    if predictors:
        X = train_design[predictors].to_numpy(dtype=float)
        coords["predictor"] = predictors

    with pm.Model(coords=coords) as model:
        # Same transformed-x priors used in the full-data sensitivity fits.
        mu_0 = pm.Normal("mu_0", mu=0.0, sigma=2.0)
        theta_0 = pm.Normal("theta_0", mu=0.0, sigma=1.0)
        sigma_proc = pm.HalfNormal("sigma_proc", sigma=2.0)

        if predictors:
            beta_mu = pm.Normal(
                "beta_mu",
                mu=0.0,
                sigma=0.5,
                dims="predictor",
            )
            beta_theta = pm.Normal(
                "beta_theta",
                mu=0.0,
                sigma=0.35,
                dims="predictor",
            )

            mu_pat = pm.Deterministic(
                "mu_pat",
                mu_0 + pt.dot(X, beta_mu),
                dims="patient",
            )
            log_theta_pat = pm.Deterministic(
                "log_theta_pat",
                theta_0 + pt.dot(X, beta_theta),
                dims="patient",
            )
        else:
            mu_pat = pm.Deterministic(
                "mu_pat",
                pt.repeat(mu_0, len(pids)),
                dims="patient",
            )
            log_theta_pat = pm.Deterministic(
                "log_theta_pat",
                pt.repeat(theta_0, len(pids)),
                dims="patient",
            )

        theta_pat = pm.Deterministic(
            "theta_pat",
            pt.exp(log_theta_pat) + 1e-6,
            dims="patient",
        )

        mean, var = ou_mean_var_pt(
            y_prev,
            mu_pat[pat_idx],
            theta_pat[pat_idx],
            sigma_proc,
            dt,
        )

        pm.Normal(
            "y_obs",
            mu=mean,
            sigma=pt.sqrt(var),
            observed=y,
            dims="obs",
        )

    return model


def heldout_lpd(idata, predictors, test_design, test_trans):
    mu0 = idata.posterior["mu_0"].values.reshape(-1)
    theta0 = idata.posterior["theta_0"].values.reshape(-1)
    sigma = idata.posterior["sigma_proc"].values.reshape(-1)

    ndraw = len(mu0)

    if predictors:
        beta_mu = idata.posterior["beta_mu"].values.reshape(-1, len(predictors))
        beta_theta = idata.posterior["beta_theta"].values.reshape(-1, len(predictors))

        x = test_design[predictors].iloc[0].to_numpy(dtype=float)

        mu = mu0 + beta_mu @ x
        theta = np.exp(theta0 + beta_theta @ x) + 1e-6
    else:
        mu = mu0
        theta = np.exp(theta0) + 1e-6

    pointwise = []

    for _, row in test_trans.iterrows():
        mean, var = ou_mean_var_np(
            float(row["x_prev_logit"]),
            mu,
            theta,
            sigma,
            float(row["dt"]),
        )

        ll = normal_logpdf(
            float(row["x_logit"]),
            mean,
            var,
        )

        lpd = float(logsumexp(ll) - np.log(ndraw))
        pointwise.append(lpd)

    pointwise = np.asarray(pointwise, dtype=float)
    return pointwise, float(np.sum(pointwise))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="/TME_OU_Branching",
    )
    ap.add_argument(
        "--input",
        default=(
            "revision_wave2_secondary_inputs/transformed_x/"
            "wave2_x_logit_eps_0p05_transition_table.csv"
        ),
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["X_0", "X_B", "X_E", "X_C"],
        choices=list(MODEL_SPECS.keys()),
    )
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--target-accept", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument(
        "--only-patients",
        nargs="*",
        default=None,
        help="Optional subset of Patient_IDs for debugging/problematic-fold reruns.",
    )

    args = ap.parse_args()

    root = Path(args.root)
    input_path = root / args.input

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    outroot = root / "revision_wave2_lopo_transformed_x_eps005"
    fold_dir = outroot / "fold_results"
    fold_dir.mkdir(parents=True, exist_ok=True)

    trans = pd.read_csv(input_path)

    required = [
        "Patient_ID",
        "participant_id",
        "project",
        "diagnosis",
        "cluster",
        "ecological_context_k2",
        "B",
        "Myeloid",
        "TNK",
        "x_prev_logit",
        "x_logit",
        "dt",
    ]

    missing = [c for c in required if c not in trans.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    if "epsilon" in trans.columns:
        unique_eps = sorted(pd.unique(trans["epsilon"]))
        if len(unique_eps) != 1 or not np.isclose(unique_eps[0], 0.05):
            raise ValueError(
                f"Expected only epsilon=0.05; found {unique_eps}"
            )

    design = participant_design(trans)

    patient_ids = design["Patient_ID"].tolist()
    if args.only_patients:
        requested = set(args.only_patients)
        patient_ids = [p for p in patient_ids if p in requested]

    config = {
        "input": str(input_path),
        "epsilon": 0.05,
        "models": args.models,
        "n_folds": len(patient_ids),
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "seed": args.seed,
        "participant_level_scaling": "training fold only",
        "endpoint": "logit(clipped x), epsilon=0.05",
    }

    outroot.mkdir(parents=True, exist_ok=True)
    with open(outroot / "wave2_lopo_eps005_run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    results = []
    diagnostics = []

    for model_index, model_name in enumerate(args.models):
        predictors = MODEL_SPECS[model_name]

        print("\n" + "#" * 80)
        print(
            f"[MODEL] {model_name} "
            f"predictors={predictors if predictors else 'None'}"
        )
        print("#" * 80)

        for fold_index, pid in enumerate(patient_ids):
            print(
                f"[FOLD {fold_index + 1}/{len(patient_ids)}] "
                f"Hold out {pid}"
            )

            train_trans = trans.loc[
                trans["Patient_ID"] != pid
            ].copy()

            test_trans = trans.loc[
                trans["Patient_ID"] == pid
            ].copy()

            train_design, test_design, scales = prepare_fold_design(
                design,
                pid,
            )

            model = build_training_model(
                predictors,
                train_trans,
                train_design,
            )

            fold_seed = (
                args.seed
                + model_index * 10000
                + fold_index
            )

            with model:
                idata = pm.sample(
                    draws=args.draws,
                    tune=args.tune,
                    chains=args.chains,
                    cores=args.cores,
                    target_accept=args.target_accept,
                    random_seed=fold_seed,
                    progressbar=False,
                    return_inferencedata=True,
                )

            # Fold diagnostics.
            divergence_count = int(
                idata.sample_stats["diverging"].sum().values
            )

            var_names = [
                "mu_0",
                "theta_0",
                "sigma_proc",
            ]

            if predictors:
                var_names += [
                    "beta_mu",
                    "beta_theta",
                ]

            summary = az.summary(
                idata,
                var_names=var_names,
                round_to=None,
            )

            max_rhat = (
                float(np.nanmax(summary["r_hat"]))
                if "r_hat" in summary.columns
                else np.nan
            )

            min_ess_bulk = (
                float(np.nanmin(summary["ess_bulk"]))
                if "ess_bulk" in summary.columns
                else np.nan
            )

            pointwise_lpd, participant_elpd = heldout_lpd(
                idata,
                predictors,
                test_design,
                test_trans,
            )

            row = {
                "model": model_name,
                "Patient_ID": pid,
                "participant_id": test_design.iloc[0]["participant_id"],
                "project": test_design.iloc[0]["project"],
                "diagnosis": test_design.iloc[0]["diagnosis"],
                "ecological_context_k2": test_design.iloc[0][
                    "ecological_context_k2"
                ],
                "n_test_transitions": int(len(test_trans)),
                "heldout_elpd": participant_elpd,
                "mean_lpd_per_transition": float(np.mean(pointwise_lpd)),
                "divergences": divergence_count,
                "max_rhat": max_rhat,
                "min_ess_bulk": min_ess_bulk,
                **scales,
            }

            results.append(row)

            diagnostics.append({
                "model": model_name,
                "Patient_ID": pid,
                "divergences": divergence_count,
                "max_rhat": max_rhat,
                "min_ess_bulk": min_ess_bulk,
            })

            pd.DataFrame([row]).to_csv(
                fold_dir / f"{model_name}_{pid}_fold_summary.csv",
                index=False,
            )

    results_df = pd.DataFrame(results)
    diagnostics_df = pd.DataFrame(diagnostics)

    results_df.to_csv(
        outroot / "wave2_lopo_eps005_participant_results.csv",
        index=False,
    )

    diagnostics_df.to_csv(
        outroot / "wave2_lopo_eps005_diagnostics.csv",
        index=False,
    )

    # Model-level summary at participant validation unit.
    model_summary = (
        results_df.groupby("model")
        .agg(
            n_participants=("Patient_ID", "nunique"),
            total_elpd=("heldout_elpd", "sum"),
            mean_participant_elpd=("heldout_elpd", "mean"),
            se_participant_elpd=(
                "heldout_elpd",
                lambda x: float(
                    np.std(x, ddof=1) / np.sqrt(len(x))
                ),
            ),
            mean_lpd_per_transition=(
                "mean_lpd_per_transition",
                "mean",
            ),
            total_test_transitions=(
                "n_test_transitions",
                "sum",
            ),
            n_folds_with_divergence=(
                "divergences",
                lambda x: int(
                    np.sum(np.asarray(x) > 0)
                ),
            ),
            max_fold_rhat=("max_rhat", "max"),
            min_fold_ess_bulk=("min_ess_bulk", "min"),
        )
        .reset_index()
        .sort_values(
            "total_elpd",
            ascending=False,
        )
    )

    model_summary.to_csv(
        outroot / "wave2_lopo_eps005_model_summary.csv",
        index=False,
    )

    # Paired participant-level differences.
    wide = results_df.pivot(
        index="Patient_ID",
        columns="model",
        values="heldout_elpd",
    )

    pair_rows = []
    models = list(wide.columns)

    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            d = (
                wide[model_a]
                - wide[model_b]
            ).dropna()

            pair_rows.append({
                "model_A": model_a,
                "model_B": model_b,
                "mean_delta_elpd_A_minus_B": float(d.mean()),
                "total_delta_elpd_A_minus_B": float(d.sum()),
                "se_delta": (
                    float(d.std(ddof=1) / np.sqrt(len(d)))
                    if len(d) > 1
                    else np.nan
                ),
                "n_participants": int(len(d)),
                "n_A_better": int(np.sum(d > 0)),
                "n_B_better": int(np.sum(d < 0)),
                "n_ties": int(np.sum(d == 0)),
            })

    pairwise = pd.DataFrame(pair_rows)

    pairwise.to_csv(
        outroot / "wave2_lopo_eps005_pairwise_differences.csv",
        index=False,
    )

    print("\n[OK] True LOPO transformed-x epsilon=0.05 complete.")

    print("\nModel summary:")
    print(
        model_summary.to_string(index=False)
    )

    print("\nPaired participant-level differences:")
    print(
        pairwise.to_string(index=False)
    )


if __name__ == "__main__":
    main()
