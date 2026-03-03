#ou_ecotype_ou_branching_calibration.py

"""
Ecotype-aware OU–Branching calibration with PyMC.

Steps:
  1. Load patient-level TME + ecotype info from:
        patient_immune_ecotypes.csv
  2. Load longitudinal OU data from:
        longitudinal_data.csv
     (EDIT path/columns in load_longitudinal_data()).
  3. Build patient-level μ_i and θ_i priors that depend on:
        - immune ecotype (random intercepts)
        - TME covariates (regression)
  4. Fit OU model with PyMC and save posterior to .nc.

You only need to:
  - Set ROOT to your project folder
  - Point LONGITUDINAL_PATH to your real time-series file
  - Adjust column names in load_longitudinal_data() if needed
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at
import arviz as az

# =========================================================
# 0. Paths
# =========================================================
ROOT = Path("/")
PATIENT_PATH = ROOT / "patient_immune_ecotypes.csv"
LONGITUDINAL_PATH = ROOT / "kmt2a_longitudinal_clean.xlsx"  # <-- edit if needed
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True, parents=True)

OUT_IDATA = OUT_DIR / "tme_ou_ecotype_idata.nc"


# =========================================================
# 1. Load patient-level ecotypes + TME covariates
# =========================================================
patients = pd.read_csv(PATIENT_PATH)

# Robust ecotype coding (works for labels like 1..4 or strings)
eco_cat = pd.Categorical(patients["immune_ecotype"])
eco_idx = eco_cat.codes.astype("int64")  # 0..K-1
K = len(eco_cat.categories)

tme_cols = [
    "frac_unknown_z",
    "frac_T_given_known_z",
    "frac_B_given_known_z",
    "frac_myeloid_given_known_z",
    "frac_NK_given_known_z",
    "frac_stromal_given_known_z",
]
tme_cols = [c for c in tme_cols if c in patients.columns]

X_cov = patients[tme_cols].to_numpy(dtype="float64")
N_pat, P = X_cov.shape

print(f"[INFO] Loaded {N_pat} patients, {P} TME covariates, {K} ecotypes.")
print("[INFO] Ecotype categories:", list(eco_cat.categories))
print("[INFO] TME covariates:", tme_cols)

# Map Patient_ID -> row index
patient_ids = patients["Patient_ID"].astype(str).tolist()
pat_index_map = {pid: i for i, pid in enumerate(patient_ids)}


# =========================================================
# 2. Load longitudinal OU data
# =========================================================
def load_longitudinal_data(path: Path) -> pd.DataFrame:
    """
    Excel sheet 'Series' with columns:
        Patient_ID, series, t, value

    Treat each (Patient_ID, series) as an independent OU path.
    Returns df with: pat_index, y_prev, dt, y
    """
    df = pd.read_excel(path, sheet_name="Series")

    time_col = "t"
    value_col = "value"
    series_col = "series"

    df["Patient_ID"] = df["Patient_ID"].astype(str)
    df["pat_index"] = df["Patient_ID"].map(pat_index_map)

    df = df.dropna(subset=["pat_index"]).copy()
    df["pat_index"] = df["pat_index"].astype(int)

    df = df.sort_values(["pat_index", series_col, time_col]).copy()

    df["y_prev"] = df.groupby(["pat_index", series_col])[value_col].shift(1)
    df["dt"] = df.groupby(["pat_index", series_col])[time_col].diff()

    df = df.dropna(subset=["y_prev", "dt"]).reset_index(drop=True)
    df = df[df["dt"] > 0].reset_index(drop=True)

    df = df.rename(columns={value_col: "y", time_col: "time"})
    return df

long_df = load_longitudinal_data(LONGITUDINAL_PATH)
print(f"[INFO] Loaded longitudinal data with {len(long_df)} transitions.")

y_obs = long_df["y"].to_numpy(dtype="float64")
y_prev = long_df["y_prev"].to_numpy(dtype="float64")
dt_obs = long_df["dt"].to_numpy(dtype="float64")
pat_idx = long_df["pat_index"].to_numpy(dtype="int64")


# =========================================================
# 3. OU transition helper
# =========================================================
def ou_transition_mean_var(y_prev, mu_i, theta_i, sigma_proc, dt):
    decay = at.exp(-theta_i * dt)
    mean = mu_i + (y_prev - mu_i) * decay
    var = (sigma_proc**2) / (2.0 * theta_i) * (1.0 - at.exp(-2.0 * theta_i * dt))
    return mean, var


# =========================================================
# 4. Build + sample model (single run)
# =========================================================
coords = {
    "patient": patient_ids,
    "ecotype": [str(x) for x in eco_cat.categories],
    "cov": tme_cols,
}

with pm.Model(coords=coords) as ou_ecotype_model:

    # ---- μ model ----
    mu_0 = pm.Normal("mu_0", 0.0, 1.0)

    sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)
    z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="ecotype")
    alpha_ecotype = pm.Deterministic("alpha_ecotype", z_alpha * sigma_alpha, dims="ecotype")

    beta = pm.Normal("beta", 0.0, 0.5, dims="cov")  # slightly tighter than 1.0 helps stability

    mu_pat = pm.Deterministic(
        "mu_pat",
        mu_0 + alpha_ecotype[eco_idx] + at.dot(X_cov, beta),
        dims="patient",
    )

    # ---- θ model (log scale) ----
    theta_0 = pm.Normal("theta_0", 0.0, 1.0)

    sigma_gamma = pm.HalfNormal("sigma_gamma", 1.0)
    z_gamma = pm.Normal("z_gamma", 0.0, 1.0, dims="ecotype")
    gamma_ecotype = pm.Deterministic("gamma_ecotype", z_gamma * sigma_gamma, dims="ecotype")

    eta = pm.Normal("eta", 0.0, 0.5, dims="cov")

    log_theta_pat = pm.Deterministic(
        "log_theta_pat",
        theta_0 + gamma_ecotype[eco_idx] + at.dot(X_cov, eta),
        dims="patient",
    )
    theta_pat = pm.Deterministic("theta_pat", at.exp(log_theta_pat) + 1e-6, dims="patient")

    # ---- global process diffusion ----
    sigma_proc = pm.HalfNormal("sigma_proc", 1.0)

    # ---- likelihood on transitions ----
    mu_for_obs = mu_pat[pat_idx]
    theta_for_obs = theta_pat[pat_idx]

    mean_tr, var_tr = ou_transition_mean_var(
        y_prev=y_prev,
        mu_i=mu_for_obs,
        theta_i=theta_for_obs,
        sigma_proc=sigma_proc,
        dt=dt_obs,
    )

    # numerical floor to avoid sqrt(negative) from tiny rounding
    var_tr = at.clip(var_tr, 1e-12, np.inf)

    pm.Normal("y_obs", mu=mean_tr, sigma=at.sqrt(var_tr), observed=y_obs)

    idata = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=4,
        target_accept=0.95,   # helps with divergences
        random_seed=42,
    )

az.to_netcdf(idata, OUT_IDATA)
print(f"[INFO] Saved idata to: {OUT_IDATA}")

print(az.summary(idata, var_names=[
    "mu_0", "sigma_alpha", "alpha_ecotype",
    "theta_0", "sigma_gamma", "gamma_ecotype",
    "sigma_proc"
]))
