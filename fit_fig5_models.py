# fit_fig5_models.py

from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as at
import arviz as az

# ---------- disable C compiler backend ----------
pytensor.config.cxx = ""
pytensor.config.mode = "FAST_COMPILE"
# ----------------------------------------------

# =========================================================
# 0. Paths
# =========================================================
ROOT = Path("/")
PATIENT_PATH = ROOT / "patient_immune_ecotypes.csv"
LONGITUDINAL_PATH = ROOT / "kmt2a_longitudinal_clean.xlsx"  # your file
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

Y_RV_NAME = "y_obs"  # observed RV name used in pm.Normal(...)

# =========================================================
# 1. Load patient-level ecotypes + TME covariates
# =========================================================
patients = pd.read_csv(PATIENT_PATH)

eco_idx = patients["immune_ecotype"].astype(int).values  # (N_pat,)
K = patients["immune_ecotype"].nunique()

tme_cols = [
    "frac_unknown_z",
    "frac_T_given_known_z",
    "frac_B_given_known_z",
    "frac_myeloid_given_known_z",
    "frac_NK_given_known_z",
    "frac_stromal_given_known_z",
]
tme_cols = [c for c in tme_cols if c in patients.columns]
X_cov = patients[tme_cols].values.astype("float64")  # (N_pat, P)
N_pat, P = X_cov.shape

print(f"[INFO] Loaded {N_pat} patients, {P} covariates, {K} ecotypes.")
print("[INFO] Covariates:", tme_cols)

patient_ids = patients["Patient_ID"].tolist()
pat_index_map = {pid: i for i, pid in enumerate(patient_ids)}

# =========================================================
# 2. Load longitudinal OU data
# =========================================================
def load_longitudinal_data(path: Path) -> pd.DataFrame:
    """
    Uses sheet 'Series' with columns: Patient_ID, series, t, value
    Treats each (Patient_ID, series) as an independent OU path.
    Returns df with: pat_index, y_prev, dt, y
    """
    df = pd.read_excel(path, sheet_name="Series")

    time_col = "t"
    value_col = "value"
    series_col = "series"

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

y_obs = long_df["y"].values.astype("float64")
y_prev = long_df["y_prev"].values.astype("float64")
dt_obs = long_df["dt"].values.astype("float64")
pat_idx = long_df["pat_index"].values.astype("int64")  # (N_obs,)

# =========================================================
# 3. Transition helpers
# =========================================================
def ou_mean_var(y_prev_t, mu_i, theta_i, sigma_proc, dt):
    decay = at.exp(-theta_i * dt)
    mean = mu_i + (y_prev_t - mu_i) * decay
    var = (sigma_proc**2) / (2.0 * theta_i) * (1.0 - at.exp(-2.0 * theta_i * dt))
    return mean, var

def rw_mean_var(y_prev_t, sigma_proc, dt):
    mean = y_prev_t
    var = (sigma_proc**2) * dt
    return mean, var

# =========================================================
# 4. Model builders (return pm.Model)
# =========================================================
def build_full_model(eco_idx_used: np.ndarray) -> pm.Model:
    """
    Your current 'full' model: ecotype random effects + TME covariate regression
    for mu and log-theta. Global sigma_proc. OU transition likelihood.
    """
    with pm.Model() as m:
        # μ_i
        mu_0 = pm.Normal("mu_0", mu=0.0, sigma=1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
        alpha_ecotype = pm.Normal("alpha_ecotype", mu=0.0, sigma=sigma_alpha, shape=K)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=P)

        mu_pat = pm.Deterministic(
            "mu_pat",
            mu_0 + alpha_ecotype[eco_idx_used] + at.dot(X_cov, beta)  # (N_pat,)
        )

        # θ_i (log scale)
        theta_0 = pm.Normal("theta_0", mu=0.0, sigma=1.0)
        sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=1.0)
        gamma_ecotype = pm.Normal("gamma_ecotype", mu=0.0, sigma=sigma_gamma, shape=K)
        eta = pm.Normal("eta", mu=0.0, sigma=1.0, shape=P)

        log_theta_pat = pm.Deterministic(
            "log_theta_pat",
            theta_0 + gamma_ecotype[eco_idx_used] + at.dot(X_cov, eta)
        )
        theta_pat = pm.Deterministic("theta_pat", at.exp(log_theta_pat))

        # global diffusion
        sigma_proc = pm.HalfNormal("sigma_proc", sigma=1.0)

        # likelihood per transition
        mu_for_obs = mu_pat[pat_idx]
        theta_for_obs = theta_pat[pat_idx]

        mean_tr, var_tr = ou_mean_var(
            y_prev_t=y_prev,
            mu_i=mu_for_obs,
            theta_i=theta_for_obs,
            sigma_proc=sigma_proc,
            dt=dt_obs,
        )

        pm.Normal(Y_RV_NAME, mu=mean_tr, sigma=at.sqrt(var_tr), observed=y_obs)
    return m


def build_ou_only_model() -> pm.Model:
    """
    OU baseline with NO TME/ecotype structure:
    patient-specific mu_i and theta_i with simple hierarchical priors.
    """
    with pm.Model() as m:
        # patient-specific mu_i (no ecotype, no covariates)
        mu0 = pm.Normal("mu0", mu=0.0, sigma=1.0)
        tau_mu = pm.HalfNormal("tau_mu", sigma=1.0)
        mu_pat = pm.Normal("mu_pat", mu=mu0, sigma=tau_mu, shape=N_pat)

        # patient-specific theta_i (log scale)
        theta0 = pm.Normal("theta0", mu=0.0, sigma=1.0)
        tau_theta = pm.HalfNormal("tau_theta", sigma=1.0)
        log_theta_pat = pm.Normal("log_theta_pat", mu=theta0, sigma=tau_theta, shape=N_pat)
        theta_pat = pm.Deterministic("theta_pat", at.exp(log_theta_pat))

        sigma_proc = pm.HalfNormal("sigma_proc", sigma=1.0)

        mu_for_obs = mu_pat[pat_idx]
        theta_for_obs = theta_pat[pat_idx]

        mean_tr, var_tr = ou_mean_var(
            y_prev_t=y_prev,
            mu_i=mu_for_obs,
            theta_i=theta_for_obs,
            sigma_proc=sigma_proc,
            dt=dt_obs,
        )

        pm.Normal(Y_RV_NAME, mu=mean_tr, sigma=at.sqrt(var_tr), observed=y_obs)
    return m


def build_rw_model() -> pm.Model:
    """
    Random-walk (Brownian) baseline:
      y_t | y_{t-1} ~ Normal(y_prev, sigma_proc*sqrt(dt))
    """
    with pm.Model() as m:
        sigma_proc = pm.HalfNormal("sigma_proc", sigma=1.0)
        mean_tr, var_tr = rw_mean_var(y_prev_t=y_prev, sigma_proc=sigma_proc, dt=dt_obs)
        pm.Normal(Y_RV_NAME, mu=mean_tr, sigma=at.sqrt(var_tr), observed=y_obs)
    return m


def build_static_ecotype_model(eco_idx_used: np.ndarray) -> pm.Model:
    """
    Static ecotype-only baseline (no dynamics):
      y ~ Normal(beta_k, sigma)
    where k is ecotype of the patient for that observation.
    """
    eco_for_obs = eco_idx_used[pat_idx]  # ecotype label per transition

    with pm.Model() as m:
        beta_k = pm.Normal("beta_ecotype", mu=0.0, sigma=1.0, shape=K)
        sigma = pm.HalfNormal("sigma_obs", sigma=1.0)
        pm.Normal(Y_RV_NAME, mu=beta_k[eco_for_obs], sigma=sigma, observed=y_obs)
    return m


# =========================================================
# 5. Fit + PPC + save helper
# =========================================================
def fit_save(model: pm.Model, out_path: Path, seed: int = 42,
             draws: int = 1000, tune: int = 1000, chains: int = 4, cores: int = 4,
             target_accept: float = 0.9):
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},  # needed for LOO/WAIC
        )
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=[Y_RV_NAME],
            random_seed=seed,
            extend_inferencedata=True,
        )

    az.to_netcdf(idata, out_path)
    print(f"[OK] Wrote {out_path}")


# =========================================================
# 6. Run all five
# =========================================================
if __name__ == "__main__":
    # FULL
    fit_save(build_full_model(eco_idx), OUT_DIR / "full.nc", seed=1)

    # OU-only (no TME)
    fit_save(build_ou_only_model(), OUT_DIR / "ou_only.nc", seed=2)

    # Shuffled TME labels (destroys ecotype signal)
    rng = np.random.default_rng(123)
    eco_idx_shuf = rng.permutation(eco_idx)
    fit_save(build_full_model(eco_idx_shuf), OUT_DIR / "shuffled.nc", seed=3)

    # Random-walk latent baseline
    fit_save(build_rw_model(), OUT_DIR / "rw.nc", seed=4)

    # Static ecotype-only
    fit_save(build_static_ecotype_model(eco_idx), OUT_DIR / "static.nc", seed=5)
