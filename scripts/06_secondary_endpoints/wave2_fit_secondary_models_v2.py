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


N_MODEL_SPECS = {
    "N_0": ["n_prev_z", "dt_z"],
    "N_B": ["n_prev_z", "dt_z", "B_z"],
    "N_E": ["n_prev_z", "dt_z", "B_z", "Myeloid_z"],
    "N_C": ["n_prev_z", "dt_z", "Context_z"],
    "N_EC": ["n_prev_z", "dt_z", "B_z", "Myeloid_z", "Context_z"],
}

X_MODEL_SPECS = {
    "X_0": [],
    "X_B": ["B_z"],
    "X_E": ["B_z", "Myeloid_z"],
    "X_C": ["Context_z"],
}

EPSILONS = [0.01, 0.025, 0.05]


def zscore_series(s, name):
    x = pd.to_numeric(s, errors="raise").to_numpy(float)
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"{name}: zero/non-finite SD.")
    return (x - m) / sd, m, sd


def save_idata_atomic(idata, final_path: Path):
    tmp = final_path.with_name(final_path.stem + "_tmp.nc")
    if tmp.exists():
        tmp.unlink()
    az.to_netcdf(idata, tmp, engine="netcdf4")
    if final_path.exists():
        final_path.unlink()
    tmp.replace(final_path)


def posterior_diag(model_name, idata, var_names):
    summ = az.summary(idata, var_names=var_names, round_to=None)
    return {
        "model": model_name,
        "divergences": int(idata.sample_stats["diverging"].sum().values),
        "max_rhat": float(np.nanmax(summ["r_hat"])) if "r_hat" in summ.columns else np.nan,
        "min_ess_bulk": float(np.nanmin(summ["ess_bulk"])) if "ess_bulk" in summ.columns else np.nan,
        "min_ess_tail": float(np.nanmin(summ["ess_tail"])) if "ess_tail" in summ.columns else np.nan,
    }, summ


def loo_diag(model_name, idata):
    loo = az.loo(idata, pointwise=True)
    pk = np.asarray(loo.pareto_k.values, float)
    summary = {
        "model": model_name,
        "elpd_loo": float(loo.elpd_loo),
        "se_elpd_loo": float(loo.se),
        "p_loo": float(loo.p_loo),
        "n_pareto_gt_0_5": int(np.sum(pk > 0.5)),
        "n_pareto_gt_0_7": int(np.sum(pk > 0.7)),
        "n_pareto_gt_1_0": int(np.sum(pk > 1.0)),
        "max_pareto_k": float(np.max(pk)),
    }
    pointwise = pd.DataFrame({
        "model": model_name,
        "obs_index": np.arange(len(pk)),
        "pareto_k": pk,
    })
    return summary, pointwise


def coef_summary_from_idata(model_name, idata, parameter_names):
    rows = []
    for pname in parameter_names:
        if pname not in idata.posterior:
            continue
        da = idata.posterior[pname]
        if "predictor" in da.dims:
            for pred in da.coords["predictor"].values.tolist():
                vals = da.sel(predictor=pred).values.reshape(-1)
                hdi = az.hdi(vals, hdi_prob=0.95)
                rows.append({
                    "model": model_name,
                    "parameter": pname,
                    "predictor": str(pred),
                    "mean": float(vals.mean()),
                    "sd": float(vals.std(ddof=1)),
                    "median": float(np.median(vals)),
                    "hdi_2.5%": float(hdi[0]),
                    "hdi_97.5%": float(hdi[1]),
                    "p_gt_0": float(np.mean(vals > 0)),
                    "p_lt_0": float(np.mean(vals < 0)),
                })
        else:
            vals = da.values.reshape(-1)
            hdi = az.hdi(vals, hdi_prob=0.95)
            rows.append({
                "model": model_name,
                "parameter": pname,
                "predictor": pname,
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)),
                "median": float(np.median(vals)),
                "hdi_2.5%": float(hdi[0]),
                "hdi_97.5%": float(hdi[1]),
                "p_gt_0": float(np.mean(vals > 0)),
                "p_lt_0": float(np.mean(vals < 0)),
            })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# ORDINAL n
# -------------------------------------------------------------------------

def build_n_design(n_df):
    # participant-level ecology/context
    part = (
        n_df[
            ["Patient_ID", "participant_id", "cluster", "B", "Myeloid"]
        ]
        .drop_duplicates("Patient_ID")
        .sort_values("Patient_ID")
        .reset_index(drop=True)
    )

    part["Context_numeric"] = part["cluster"].map({"C1": 0.0, "C2": 1.0})
    if part["Context_numeric"].isna().any():
        raise ValueError("Unexpected context coding in n endpoint.")

    part["B_z"], bmean, bsd = zscore_series(part["B"], "B")
    part["Myeloid_z"], mmean, msd = zscore_series(part["Myeloid"], "Myeloid")
    part["Context_z"], cmean, csd = zscore_series(part["Context_numeric"], "Context")

    # transition-level baseline predictors
    df = n_df.copy()
    df["n_prev_z"], npmean, npsd = zscore_series(df["n_prev"], "n_prev")
    df["dt_z"], dtmean, dtsd = zscore_series(df["dt"], "dt")

    df = df.merge(
        part[["Patient_ID", "B_z", "Myeloid_z", "Context_z"]],
        on="Patient_ID",
        how="left",
        validate="many_to_one",
    )

    scaler = pd.DataFrame([
        {"predictor": "B", "mean": bmean, "sd_population": bsd},
        {"predictor": "Myeloid", "mean": mmean, "sd_population": msd},
        {"predictor": "Context_numeric", "mean": cmean, "sd_population": csd},
        {"predictor": "n_prev", "mean": npmean, "sd_population": npsd},
        {"predictor": "dt", "mean": dtmean, "sd_population": dtsd},
    ])
    return df, part, scaler


def build_n_model(df, predictors):
    X = df[predictors].to_numpy(float)
    y = df["n"].to_numpy("int64")

    coords = {
        "obs": np.arange(len(df)),
        "predictor": predictors,
        "cutpoint": ["cut1", "cut2"],
    }

    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", 0.0, 0.5, dims="predictor")

        # Explicitly ordered cutpoints without a transformed RV.
        # This avoids PyMC log-likelihood conversion issues with non-default
        # initial values on ordered transforms.
        cut1 = pm.Normal("cut1", mu=-0.75, sigma=1.0)
        cut_delta = pm.HalfNormal("cut_delta", sigma=1.0)
        cut2 = pm.Deterministic("cut2", cut1 + cut_delta)

        cutpoints = pm.Deterministic(
            "cutpoints",
            pt.stack([cut1, cut2]),
            dims="cutpoint",
        )

        eta = pt.dot(X, beta)

        pm.OrderedLogistic(
            "n_obs",
            eta=eta,
            cutpoints=cutpoints,
            observed=y,
            dims="obs",
        )

    return model


# -------------------------------------------------------------------------
# TRANSFORMED x OU
# -------------------------------------------------------------------------

def ou_mean_var(y_prev, mu, theta, sigma_proc, dt):
    decay = pt.exp(-theta * dt)
    mean = mu + (y_prev - mu) * decay
    var = (sigma_proc**2)/(2.0*theta) * (1.0 - pt.exp(-2.0*theta*dt))
    return mean, pt.clip(var, 1e-12, np.inf)


def build_x_design(df):
    part = (
        df[
            ["Patient_ID", "participant_id", "cluster", "B", "Myeloid"]
        ]
        .drop_duplicates("Patient_ID")
        .sort_values("Patient_ID")
        .reset_index(drop=True)
    )

    part["Context_numeric"] = part["cluster"].map({"C1": 0.0, "C2": 1.0})
    if part["Context_numeric"].isna().any():
        raise ValueError("Unexpected context coding in transformed x.")

    part["B_z"], bmean, bsd = zscore_series(part["B"], "B")
    part["Myeloid_z"], mmean, msd = zscore_series(part["Myeloid"], "Myeloid")
    part["Context_z"], cmean, csd = zscore_series(part["Context_numeric"], "Context")

    scaler = pd.DataFrame([
        {"predictor": "B", "mean": bmean, "sd_population": bsd},
        {"predictor": "Myeloid", "mean": mmean, "sd_population": msd},
        {"predictor": "Context_numeric", "mean": cmean, "sd_population": csd},
    ])
    return part, scaler


def build_x_model(df, part, predictors):
    pids = part["Patient_ID"].tolist()
    pmap = {p:i for i,p in enumerate(pids)}
    pat_idx = df["Patient_ID"].map(pmap).to_numpy("int64")

    yp = df["x_prev_logit"].to_numpy(float)
    y = df["x_logit"].to_numpy(float)
    dt = df["dt"].to_numpy(float)

    coords = {
        "patient": pids,
        "obs": np.arange(len(df)),
    }

    X = None
    if predictors:
        X = part[predictors].to_numpy(float)
        coords["predictor"] = predictors

    with pm.Model(coords=coords) as model:
        mu_0 = pm.Normal("mu_0", 0.0, 2.0)
        theta_0 = pm.Normal("theta_0", 0.0, 1.0)
        sigma_proc = pm.HalfNormal("sigma_proc", 2.0)

        if predictors:
            beta_mu = pm.Normal("beta_mu", 0.0, 0.5, dims="predictor")
            beta_theta = pm.Normal("beta_theta", 0.0, 0.35, dims="predictor")
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

        mean, var = ou_mean_var(
            yp,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/TME_OU_Branching"
    )
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=3000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    root = Path(args.root)

    n_input = (
        root
        / "revision_wave2_secondary_inputs"
        / "n"
        / "wave2_n_transition_table.csv"
    )
    tx_dir = (
        root
        / "revision_wave2_secondary_inputs"
        / "transformed_x"
    )

    if not n_input.exists():
        raise FileNotFoundError(n_input)

    outroot = root / "revision_wave2_secondary_models"
    n_idata_dir = outroot / "n_idata"
    x_idata_dir = outroot / "x_idata"
    diag_dir = outroot / "diagnostics"
    design_dir = outroot / "design"
    for d in [n_idata_dir, x_idata_dir, diag_dir, design_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "seed": args.seed,
        "n_models": list(N_MODEL_SPECS.keys()),
        "x_models": list(X_MODEL_SPECS.keys()),
        "epsilons": EPSILONS,
    }
    with open(outroot / "wave2_secondary_model_run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    diagnostics = []
    loo_rows = []
    pareto_tables = []
    coef_tables = []

    # =============================================================
    # A. ORDINAL n ABALATIONS
    # =============================================================
    n_df = pd.read_csv(n_input)
    n_df2, n_part, n_scaler = build_n_design(n_df)
    n_df2.to_csv(design_dir / "wave2_n_model_design_transition.csv", index=False)
    n_part.to_csv(design_dir / "wave2_n_participant_design.csv", index=False)
    n_scaler.to_csv(design_dir / "wave2_n_scaler_parameters.csv", index=False)

    for offset, (model_name, predictors) in enumerate(N_MODEL_SPECS.items()):
        print("\n" + "="*80)
        print(f"[ORDINAL n] Fitting {model_name}: {predictors}")
        print("="*80)

        model = build_n_model(n_df2, predictors)

        with model:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                target_accept=args.target_accept,
                random_seed=args.seed + offset,
                idata_kwargs={"log_likelihood": True},
                return_inferencedata=True,
            )
            idata = pm.sample_posterior_predictive(
                idata,
                var_names=["n_obs"],
                random_seed=args.seed + 1000 + offset,
                extend_inferencedata=True,
            )

        path = n_idata_dir / f"{model_name}_ordinal_n_idata_ppc.nc"
        save_idata_atomic(idata, path)

        diag, summ = posterior_diag(
            model_name,
            idata,
            ["beta", "cut1", "cut_delta", "cutpoints"],
        )
        diag["family"] = "ordinal_n"
        diagnostics.append(diag)
        summ.to_csv(diag_dir / f"{model_name}_posterior_summary.csv")

        coef = coef_summary_from_idata(
            model_name,
            idata,
            ["beta"],
        )
        coef["family"] = "ordinal_n"
        coef_tables.append(coef)

        try:
            lr, pk = loo_diag(model_name, idata)
            lr["family"] = "ordinal_n"
            loo_rows.append(lr)

            meta = n_df2[
                [
                    "Patient_ID", "participant_id", "ecological_context_k2",
                    "time", "n_prev", "n", "dt"
                ]
            ].reset_index(drop=True)
            meta["obs_index"] = np.arange(len(meta))
            pk = pk.merge(meta, on="obs_index", how="left")
            pk["family"] = "ordinal_n"
            pareto_tables.append(pk)
        except Exception as e:
            print(f"[WARN] LOO failed for {model_name}: {e}")

        del idata, model
        gc.collect()

    # =============================================================
    # B. TRANSFORMED x OU SENSITIVITY
    # =============================================================
    for eps_idx, eps in enumerate(EPSILONS):
        tag = str(eps).replace(".", "p")
        x_input = tx_dir / f"wave2_x_logit_eps_{tag}_transition_table.csv"
        if not x_input.exists():
            raise FileNotFoundError(x_input)

        x_df = pd.read_csv(x_input)
        x_part, x_scaler = build_x_design(x_df)

        x_part.to_csv(
            design_dir / f"wave2_x_eps_{tag}_participant_design.csv",
            index=False,
        )
        x_scaler.to_csv(
            design_dir / f"wave2_x_eps_{tag}_scaler_parameters.csv",
            index=False,
        )

        for mod_idx, (base_name, predictors) in enumerate(X_MODEL_SPECS.items()):
            model_name = f"{base_name}_eps_{tag}"

            print("\n" + "="*80)
            print(f"[TRANSFORMED x] Fitting {model_name}: {predictors or 'None'}")
            print("="*80)

            model = build_x_model(x_df, x_part, predictors)

            seed_offset = 10000 + eps_idx*100 + mod_idx

            with model:
                idata = pm.sample(
                    draws=args.draws,
                    tune=args.tune,
                    chains=args.chains,
                    cores=args.cores,
                    target_accept=args.target_accept,
                    random_seed=args.seed + seed_offset,
                    idata_kwargs={"log_likelihood": True},
                    return_inferencedata=True,
                )
                idata = pm.sample_posterior_predictive(
                    idata,
                    var_names=["y_obs"],
                    random_seed=args.seed + 20000 + seed_offset,
                    extend_inferencedata=True,
                )

            path = x_idata_dir / f"{model_name}_ou_logit_x_idata_ppc.nc"
            save_idata_atomic(idata, path)

            vars_ = ["mu_0", "theta_0", "sigma_proc"]
            if predictors:
                vars_ += ["beta_mu", "beta_theta"]

            diag, summ = posterior_diag(model_name, idata, vars_)
            diag["family"] = "transformed_x"
            diag["epsilon"] = eps
            diagnostics.append(diag)
            summ.to_csv(diag_dir / f"{model_name}_posterior_summary.csv")

            coef = coef_summary_from_idata(
                model_name,
                idata,
                ["beta_mu", "beta_theta"],
            )
            if len(coef):
                coef["family"] = "transformed_x"
                coef["epsilon"] = eps
                coef_tables.append(coef)

            try:
                lr, pk = loo_diag(model_name, idata)
                lr["family"] = "transformed_x"
                lr["epsilon"] = eps
                loo_rows.append(lr)

                meta = x_df[
                    [
                        "Patient_ID", "participant_id", "ecological_context_k2",
                        "time", "x_prev_raw", "x_raw", "dt"
                    ]
                ].reset_index(drop=True)
                meta["obs_index"] = np.arange(len(meta))
                pk = pk.merge(meta, on="obs_index", how="left")
                pk["family"] = "transformed_x"
                pk["epsilon"] = eps
                pareto_tables.append(pk)
            except Exception as e:
                print(f"[WARN] LOO failed for {model_name}: {e}")

            del idata, model
            gc.collect()

    # =============================================================
    # SAVE AGGREGATES
    # =============================================================
    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(
        diag_dir / "wave2_secondary_diagnostics.csv",
        index=False,
    )

    if loo_rows:
        loo_df = pd.DataFrame(loo_rows)
        loo_df.to_csv(
            diag_dir / "wave2_secondary_loo_summary.csv",
            index=False,
        )
    else:
        loo_df = pd.DataFrame()

    if pareto_tables:
        pareto_df = pd.concat(pareto_tables, ignore_index=True)
        pareto_df.to_csv(
            diag_dir / "wave2_secondary_pareto_k.csv",
            index=False,
        )

    if coef_tables:
        coef_df = pd.concat(coef_tables, ignore_index=True)
        coef_df.to_csv(
            diag_dir / "wave2_secondary_coefficient_summary.csv",
            index=False,
        )
    else:
        coef_df = pd.DataFrame()

    print("\n[OK] Wave 2 secondary full-data fitting complete.")
    print("Outputs:", outroot)

    print("\nDiagnostics:")
    print(diagnostics_df.to_string(index=False))

    if len(loo_df):
        print("\nTransition-level LOO summary:")
        print(loo_df.to_string(index=False))

    if len(coef_df):
        print("\nCoefficient summary:")
        print(coef_df.to_string(index=False))

    print(
        "\n[NOTE] These are full-data/transition-level secondary diagnostics. "
        "We will decide whether true participant-level LOPO is warranted only "
        "after inspecting these results."
    )


if __name__ == "__main__":
    main()
