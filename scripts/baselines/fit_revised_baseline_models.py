from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at
import arviz as az


# =============================================================================
# Paths
# =============================================================================

ROOT = Path("/TME_OU_Branching")
RESULTS_DIR = ROOT / "results"
FIG3_ASSIGNMENTS = ROOT / "Figure_3" / "patient_ecological_context_assignments.csv"
LONGITUDINAL_PATH = ROOT / "kmt2a_longitudinal_clean.xlsx"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)

OUTFILES = {
    "ou_only": RESULTS_DIR / "ou_only_revised_context_idata.nc",
    "shuffled_context": RESULTS_DIR / "shuffled_context_revised_idata.nc",
    "random_walk": RESULTS_DIR / "random_walk_revised_idata.nc",
    "static_context": RESULTS_DIR / "static_context_revised_idata.nc",
}

OUT_AUDIT = RESULTS_DIR / "revised_baseline_model_observation_audit.csv"


# =============================================================================
# Model settings
# =============================================================================

CONTEXT_ORDER = ["E1", "E2", "E3", "E4"]

TME_COLS = [
    "Unknown_z",
    "T_z",
    "B_z",
    "Myeloid_z",
    "NK_z",
    "Stromal_z",
]

DIAGNOSIS_ORDER = ["B-ALL", "T-ALL", "ETP-ALL", "AML", "MPAL", "Unknown"]

DEFAULT_DRAWS = 2000
DEFAULT_TUNE = 3000
DEFAULT_CHAINS = 4
DEFAULT_CORES = 4
DEFAULT_TARGET_ACCEPT = 0.99
DEFAULT_SEED = 42


# =============================================================================
# Helpers
# =============================================================================

def normalize_patient_id(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_to_netcdf(idata: az.InferenceData, final_path: Path) -> None:
    """
    Save safely through a temporary file to avoid partial writes and stale locks.
    """
    tmp_path = final_path.with_name(final_path.stem + "_tmp.nc")

    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[INFO] Saving temporary file:\n{tmp_path}")
    az.to_netcdf(idata, tmp_path, engine="netcdf4")

    if final_path.exists():
        final_path.unlink()

    tmp_path.replace(final_path)
    print(f"[INFO] Saved:\n{final_path}")


def ou_transition_mean_var(y_prev, mu_i, theta_i, sigma_proc, dt):
    decay = at.exp(-theta_i * dt)
    mean = mu_i + (y_prev - mu_i) * decay
    var = (sigma_proc ** 2) / (2.0 * theta_i) * (1.0 - at.exp(-2.0 * theta_i * dt))
    var = at.clip(var, 1e-12, np.inf)
    return mean, var


def load_revised_inputs(seed: int = DEFAULT_SEED):
    """
    Loads Figure 3 assignments and longitudinal transitions.

    Returns a dictionary containing:
        patients, patient_ids, eco_idx, eco_idx_shuffled,
        X_cov, cov_cols, long_df, y_obs, y_prev, dt_obs, pat_idx
    """
    patients = pd.read_csv(FIG3_ASSIGNMENTS)
    patients["Patient_ID"] = patients["Patient_ID"].map(normalize_patient_id)

    if patients["Patient_ID"].duplicated().any():
        dupes = patients.loc[patients["Patient_ID"].duplicated(), "Patient_ID"].tolist()
        raise ValueError(f"Duplicated Patient_ID values in Figure 3 assignments: {dupes[:10]}")

    missing_cols = [c for c in TME_COLS if c not in patients.columns]
    if missing_cols:
        raise ValueError(f"Missing TME columns in Figure 3 assignments: {missing_cols}")

    patients["ecological_context"] = patients["ecological_context"].astype(str).str.strip()

    eco_cat = pd.Categorical(
        patients["ecological_context"],
        categories=CONTEXT_ORDER,
        ordered=True,
    )

    eco_idx = eco_cat.codes.astype("int64")
    if np.any(eco_idx < 0):
        bad = patients.loc[eco_idx < 0, ["Patient_ID", "ecological_context"]]
        raise ValueError(f"Unrecognized ecological_context values:\n{bad}")

    # Diagnosis-aware covariates, matching revised full model
    if "diagnosis" not in patients.columns:
        patients["diagnosis"] = "Unknown"

    patients["diagnosis"] = patients["diagnosis"].fillna("Unknown").astype(str).str.strip()
    patients["diagnosis_cat"] = pd.Categorical(
        patients["diagnosis"],
        categories=DIAGNOSIS_ORDER,
        ordered=False,
    )

    diag_df = pd.get_dummies(
        patients["diagnosis_cat"],
        prefix="diagnosis",
        dtype=float,
    )

    # B-ALL reference
    diag_df = diag_df.drop(columns=["diagnosis_B-ALL"], errors="ignore")

    X_cov_df = pd.concat(
        [
            patients[TME_COLS].astype(float).reset_index(drop=True),
            diag_df.reset_index(drop=True),
        ],
        axis=1,
    )

    cov_cols = X_cov_df.columns.tolist()
    X_cov = X_cov_df.to_numpy(dtype="float64")

    if not np.isfinite(X_cov).all():
        raise ValueError("Non-finite values detected in covariate matrix.")

    patient_ids = patients["Patient_ID"].astype(str).tolist()
    pat_index_map = {pid: i for i, pid in enumerate(patient_ids)}

    # Shuffled context labels preserve marginal context counts
    rng = np.random.default_rng(seed)
    eco_idx_shuffled = rng.permutation(eco_idx)

    # Longitudinal transitions
    long = pd.read_excel(LONGITUDINAL_PATH, sheet_name="Series")
    required_cols = ["Patient_ID", "series", "t", "value"]
    missing = [c for c in required_cols if c not in long.columns]
    if missing:
        raise ValueError(f"Longitudinal file missing columns: {missing}")

    long["Patient_ID"] = long["Patient_ID"].map(normalize_patient_id)
    long["pat_index"] = long["Patient_ID"].map(pat_index_map)

    n_dropped = long["pat_index"].isna().sum()
    if n_dropped > 0:
        print(f"[WARN] Dropping {n_dropped} longitudinal rows not present in Figure 3 assignments.")

    long = long.dropna(subset=["pat_index"]).copy()
    long["pat_index"] = long["pat_index"].astype(int)

    long = long.sort_values(["pat_index", "series", "t"]).copy()
    long["y_prev"] = long.groupby(["pat_index", "series"])["value"].shift(1)
    long["dt"] = long.groupby(["pat_index", "series"])["t"].diff()

    long = long.dropna(subset=["y_prev", "dt"]).reset_index(drop=True)
    long = long[long["dt"] > 0].reset_index(drop=True)
    long = long.rename(columns={"value": "y", "t": "time"})

    y_obs = long["y"].to_numpy(dtype="float64")
    y_prev = long["y_prev"].to_numpy(dtype="float64")
    dt_obs = long["dt"].to_numpy(dtype="float64")
    pat_idx = long["pat_index"].to_numpy(dtype="int64")

    if len(y_obs) == 0:
        raise ValueError("No valid longitudinal transitions retained.")

    # Audit file
    audit = long[["Patient_ID", "series", "time", "y_prev", "dt", "y", "pat_index"]].copy()
    audit["ecological_context"] = [patients.loc[i, "ecological_context"] for i in audit["pat_index"]]
    audit["diagnosis"] = [patients.loc[i, "diagnosis"] for i in audit["pat_index"]]
    audit.to_csv(OUT_AUDIT, index=False)

    print("[INFO] Revised baseline input summary")
    print(f"  patients: {len(patients)}")
    print(f"  covariates: {len(cov_cols)}")
    print(f"  transitions: {len(y_obs)}")
    print(f"  represented patients: {len(set(pat_idx))}")
    print("[INFO] Context counts:")
    print(patients["ecological_context"].value_counts().reindex(CONTEXT_ORDER))
    print("[INFO] Longitudinal transition counts by context:")
    print(audit.groupby("ecological_context").size().reindex(CONTEXT_ORDER).fillna(0).astype(int))
    print(f"[INFO] Saved observation audit:\n{OUT_AUDIT}")

    return {
        "patients": patients,
        "patient_ids": patient_ids,
        "eco_idx": eco_idx,
        "eco_idx_shuffled": eco_idx_shuffled,
        "X_cov": X_cov,
        "cov_cols": cov_cols,
        "long_df": long,
        "y_obs": y_obs,
        "y_prev": y_prev,
        "dt_obs": dt_obs,
        "pat_idx": pat_idx,
    }


def sample_and_ppc(model_name, model, draws, tune, chains, cores, target_accept, seed):
    with model:
        print(f"[INFO] Sampling model: {model_name}")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            idata_kwargs={"log_likelihood": True},
        )

        print(f"[INFO] Sampling posterior predictive for: {model_name}")

        idata = pm.sample_posterior_predictive(
            idata,
            var_names=["y_obs"],
            random_seed=seed,
            extend_inferencedata=True,
        )

    divergences = int(idata.sample_stats["diverging"].sum().values)
    print(f"[CHECK] {model_name}: divergences = {divergences}")

    if divergences > 0:
        print(
            f"[WARN] {model_name} has divergences. Consider increasing target_accept "
            "or simplifying priors if this model is used for final inference."
        )

    return idata


# =============================================================================
# Model definitions
# =============================================================================

def build_ou_only(data):
    y_obs = data["y_obs"]
    y_prev = data["y_prev"]
    dt_obs = data["dt_obs"]

    coords = {"obs": np.arange(len(y_obs))}

    with pm.Model(coords=coords) as model:
        mu_0 = pm.Normal("mu_0", 0.0, 1.0)
        log_theta_0 = pm.Normal("log_theta_0", 0.0, 1.0)
        theta = pm.Deterministic("theta", at.exp(log_theta_0) + 1e-6)

        sigma_proc = pm.HalfNormal("sigma_proc", 1.0)

        mean_tr, var_tr = ou_transition_mean_var(
            y_prev=y_prev,
            mu_i=mu_0,
            theta_i=theta,
            sigma_proc=sigma_proc,
            dt=dt_obs,
        )

        pm.Normal(
            "y_obs",
            mu=mean_tr,
            sigma=at.sqrt(var_tr),
            observed=y_obs,
            dims="obs",
        )

    return model


def build_shuffled_context_ou(data):
    y_obs = data["y_obs"]
    y_prev = data["y_prev"]
    dt_obs = data["dt_obs"]
    pat_idx = data["pat_idx"]

    patient_ids = data["patient_ids"]
    eco_idx_shuffled = data["eco_idx_shuffled"]
    X_cov = data["X_cov"]
    cov_cols = data["cov_cols"]

    coords = {
        "patient": patient_ids,
        "ecological_context": CONTEXT_ORDER,
        "cov": cov_cols,
        "obs": np.arange(len(y_obs)),
    }

    with pm.Model(coords=coords) as model:
        mu_0 = pm.Normal("mu_0", 0.0, 1.0)

        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.75)
        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="ecological_context")
        alpha_context = pm.Deterministic(
            "alpha_context",
            z_alpha * sigma_alpha,
            dims="ecological_context",
        )

        beta = pm.Normal("beta", 0.0, 0.35, dims="cov")

        mu_pat = pm.Deterministic(
            "mu_pat",
            mu_0 + alpha_context[eco_idx_shuffled] + at.dot(X_cov, beta),
            dims="patient",
        )

        theta_0 = pm.Normal("theta_0", 0.0, 1.0)

        sigma_gamma = pm.HalfNormal("sigma_gamma", 0.75)
        z_gamma = pm.Normal("z_gamma", 0.0, 1.0, dims="ecological_context")
        gamma_context = pm.Deterministic(
            "gamma_context",
            z_gamma * sigma_gamma,
            dims="ecological_context",
        )

        eta = pm.Normal("eta", 0.0, 0.35, dims="cov")

        log_theta_pat = pm.Deterministic(
            "log_theta_pat",
            theta_0 + gamma_context[eco_idx_shuffled] + at.dot(X_cov, eta),
            dims="patient",
        )

        theta_pat = pm.Deterministic(
            "theta_pat",
            at.exp(log_theta_pat) + 1e-6,
            dims="patient",
        )

        sigma_proc = pm.HalfNormal("sigma_proc", 1.0)

        mean_tr, var_tr = ou_transition_mean_var(
            y_prev=y_prev,
            mu_i=mu_pat[pat_idx],
            theta_i=theta_pat[pat_idx],
            sigma_proc=sigma_proc,
            dt=dt_obs,
        )

        pm.Normal(
            "y_obs",
            mu=mean_tr,
            sigma=at.sqrt(var_tr),
            observed=y_obs,
            dims="obs",
        )

    return model


def build_random_walk(data):
    y_obs = data["y_obs"]
    y_prev = data["y_prev"]
    dt_obs = data["dt_obs"]

    coords = {"obs": np.arange(len(y_obs))}

    with pm.Model(coords=coords) as model:
        rw_drift = pm.Normal("rw_drift", 0.0, 0.5)
        sigma_rw = pm.HalfNormal("sigma_rw", 1.0)

        mean_tr = y_prev + rw_drift * dt_obs
        sd_tr = sigma_rw * at.sqrt(dt_obs)
        sd_tr = at.clip(sd_tr, 1e-6, np.inf)

        pm.Normal(
            "y_obs",
            mu=mean_tr,
            sigma=sd_tr,
            observed=y_obs,
            dims="obs",
        )

    return model


def build_static_context(data):
    y_obs = data["y_obs"]
    pat_idx = data["pat_idx"]
    eco_idx = data["eco_idx"]

    ctx_for_obs = eco_idx[pat_idx]

    coords = {
        "ecological_context": CONTEXT_ORDER,
        "obs": np.arange(len(y_obs)),
    }

    with pm.Model(coords=coords) as model:
        mu_context = pm.Normal("mu_context", 0.0, 1.0, dims="ecological_context")
        sigma_obs = pm.HalfNormal("sigma_obs", 1.0)

        pm.Normal(
            "y_obs",
            mu=mu_context[ctx_for_obs],
            sigma=sigma_obs,
            observed=y_obs,
            dims="obs",
        )

    return model


MODEL_BUILDERS = {
    "ou_only": build_ou_only,
    "shuffled_context": build_shuffled_context_ou,
    "random_walk": build_random_walk,
    "static_context": build_static_context,
}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "ou_only", "shuffled_context", "random_walk", "static_context"],
        help="Which model to fit.",
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--cores", type=int, default=DEFAULT_CORES)
    parser.add_argument("--target_accept", type=float, default=DEFAULT_TARGET_ACCEPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refit and overwrite existing output files.",
    )

    args = parser.parse_args()

    data = load_revised_inputs(seed=args.seed)

    if args.model == "all":
        models_to_fit = ["ou_only", "shuffled_context", "random_walk", "static_context"]
    else:
        models_to_fit = [args.model]

    for model_key in models_to_fit:
        outpath = OUTFILES[model_key]

        if outpath.exists() and not args.force:
            print(f"[SKIP] {model_key}: file already exists:\n{outpath}")
            continue

        model = MODEL_BUILDERS[model_key](data)

        idata = sample_and_ppc(
            model_name=model_key,
            model=model,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            seed=args.seed,
        )

        safe_to_netcdf(idata, outpath)

        print(f"[DONE] {model_key}")

    print("\nAll requested baseline models completed.")


if __name__ == "__main__":
    main()
