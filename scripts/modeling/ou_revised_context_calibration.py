from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at
import arviz as az


# =========================================================
# 0. Paths
# =========================================================

ROOT = Path("/TME_OU_Branching")

# This is the source-of-truth assignment file from revised Figure 3
PATIENT_PATH = ROOT / "Figure_3" / "patient_ecological_context_assignments.csv"

# Longitudinal trait data
LONGITUDINAL_PATH = ROOT / "kmt2a_longitudinal_clean.xlsx"

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# New revised-context trace file
OUT_IDATA = OUT_DIR / "ou_revised_ecological_context_idata_ppc.nc"

# Exact patient order used in this model
OUT_ORDER = OUT_DIR / "model_patient_order_revised_context.csv"

# Model input audit table
OUT_MODEL_INPUT = OUT_DIR / "model_input_revised_context.csv"


# =========================================================
# 1. Load revised ecological contexts + covariates
# =========================================================

patients = pd.read_csv(PATIENT_PATH)
patients["Patient_ID"] = patients["Patient_ID"].astype(str).str.strip()

# Check duplicates
if patients["Patient_ID"].duplicated().any():
    dupes = patients.loc[patients["Patient_ID"].duplicated(), "Patient_ID"].tolist()
    raise ValueError(f"Duplicated Patient_ID values in Figure 3 assignments: {dupes[:10]}")

# Stable context order from revised Figure 3
context_order = ["E1", "E2", "E3", "E4"]

if "ecological_context" not in patients.columns:
    raise ValueError("Missing required column: ecological_context")

patients["ecological_context"] = patients["ecological_context"].astype(str).str.strip()

eco_cat = pd.Categorical(
    patients["ecological_context"],
    categories=context_order,
    ordered=True,
)

eco_idx = eco_cat.codes.astype("int64")
K = len(eco_cat.categories)

if np.any(eco_idx < 0):
    bad = patients.loc[eco_idx < 0, ["Patient_ID", "ecological_context"]]
    raise ValueError(f"Unrecognized ecological_context labels:\n{bad}")

# Revised Figure 3 standardized TME feature names
tme_cols = [
    "Unknown_z",
    "T_z",
    "B_z",
    "Myeloid_z",
    "NK_z",
    "Stromal_z",
]

missing_tme = [c for c in tme_cols if c not in patients.columns]
if missing_tme:
    raise ValueError(f"Missing revised TME columns: {missing_tme}")

# Optional broad diagnosis-aware covariates
# Keep subdiagnosis out of the primary model because many subdiagnosis groups are tiny.
INCLUDE_DIAGNOSIS_COVARIATES = True

if "diagnosis" not in patients.columns:
    patients["diagnosis"] = "Unknown"

patients["diagnosis"] = patients["diagnosis"].fillna("Unknown").astype(str).str.strip()

if INCLUDE_DIAGNOSIS_COVARIATES:
    diagnosis_order = ["B-ALL", "T-ALL", "ETP-ALL", "AML", "MPAL", "Unknown"]

    patients["diagnosis_cat"] = pd.Categorical(
        patients["diagnosis"],
        categories=diagnosis_order,
        ordered=False,
    )

    diag_df = pd.get_dummies(
        patients["diagnosis_cat"],
        prefix="diagnosis",
        dtype=float,
    )

    # Use B-ALL as reference because it is the dominant diagnosis.
    diag_df = diag_df.drop(columns=["diagnosis_B-ALL"], errors="ignore")

    X_cov_df = pd.concat(
        [
            patients[tme_cols].astype(float).reset_index(drop=True),
            diag_df.reset_index(drop=True),
        ],
        axis=1,
    )
else:
    X_cov_df = patients[tme_cols].astype(float).copy()

cov_cols = X_cov_df.columns.tolist()
X_cov = X_cov_df.to_numpy(dtype="float64")

if not np.isfinite(X_cov).all():
    bad_cols = X_cov_df.columns[~np.isfinite(X_cov_df).all(axis=0)].tolist()
    raise ValueError(f"Non-finite covariate values detected in columns: {bad_cols}")

N_pat, P = X_cov.shape

patient_ids = patients["Patient_ID"].astype(str).tolist()
pat_index_map = {pid: i for i, pid in enumerate(patient_ids)}

# Save exact model patient order
pd.DataFrame({"Patient_ID": patient_ids}).to_csv(OUT_ORDER, index=False)

# Save model input audit file
audit_cols = ["Patient_ID", "diagnosis", "subdiagnosis", "ecological_context"]
audit_cols = [c for c in audit_cols if c in patients.columns]

model_input = pd.concat(
    [
        patients[audit_cols].reset_index(drop=True),
        X_cov_df.reset_index(drop=True),
    ],
    axis=1,
)

model_input.to_csv(OUT_MODEL_INPUT, index=False)

print(f"[INFO] Loaded {N_pat} patients, {P} covariates, {K} ecological contexts.")
print("[INFO] Ecological context categories:", list(eco_cat.categories))
print("[INFO] Context counts:")
print(patients["ecological_context"].value_counts().reindex(context_order))
print("[INFO] Covariates:", cov_cols)
print(f"[INFO] Saved model patient order to: {OUT_ORDER}")
print(f"[INFO] Saved model input audit table to: {OUT_MODEL_INPUT}")


# =========================================================
# 2. Load longitudinal OU data
# =========================================================

def load_longitudinal_data(path: Path) -> pd.DataFrame:
    """
    Excel sheet 'Series' with columns:
        Patient_ID, series, t, value

    Treat each (Patient_ID, series) as an independent OU path.
    Returns a dataframe with:
        pat_index, y_prev, dt, y
    """
    df = pd.read_excel(path, sheet_name="Series")

    required_cols = ["Patient_ID", "series", "t", "value"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Longitudinal file is missing columns: {missing}")

    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()
    df["pat_index"] = df["Patient_ID"].map(pat_index_map)

    n_missing = df["pat_index"].isna().sum()
    if n_missing > 0:
        print(f"[WARN] Dropping {n_missing} longitudinal rows with Patient_ID not in Figure 3 assignments.")

    df = df.dropna(subset=["pat_index"]).copy()
    df["pat_index"] = df["pat_index"].astype(int)

    df = df.sort_values(["pat_index", "series", "t"]).copy()

    df["y_prev"] = df.groupby(["pat_index", "series"])["value"].shift(1)
    df["dt"] = df.groupby(["pat_index", "series"])["t"].diff()

    df = df.dropna(subset=["y_prev", "dt"]).reset_index(drop=True)
    df = df[df["dt"] > 0].reset_index(drop=True)

    df = df.rename(columns={"value": "y", "t": "time"})
    return df


long_df = load_longitudinal_data(LONGITUDINAL_PATH)

if len(long_df) == 0:
    raise ValueError("No longitudinal transitions retained after preprocessing.")

print(f"[INFO] Loaded longitudinal data with {len(long_df)} transitions.")
print("[INFO] Longitudinal patients represented:",
      long_df["pat_index"].nunique(), "of", N_pat)

y_obs = long_df["y"].to_numpy(dtype="float64")
y_prev = long_df["y_prev"].to_numpy(dtype="float64")
dt_obs = long_df["dt"].to_numpy(dtype="float64")
pat_idx = long_df["pat_index"].to_numpy(dtype="int64")

if not np.isfinite(y_obs).all():
    raise ValueError("Non-finite y_obs values detected.")
if not np.isfinite(y_prev).all():
    raise ValueError("Non-finite y_prev values detected.")
if not np.isfinite(dt_obs).all():
    raise ValueError("Non-finite dt values detected.")


# =========================================================
# 3. OU transition helper
# =========================================================

def ou_transition_mean_var(y_prev, mu_i, theta_i, sigma_proc, dt):
    """
    Closed-form OU transition.

    Y_t | Y_{t-dt} ~ Normal(
        mu + (y_prev - mu) * exp(-theta * dt),
        sigma_proc^2 / (2 theta) * (1 - exp(-2 theta dt))
    )
    """
    decay = at.exp(-theta_i * dt)
    mean = mu_i + (y_prev - mu_i) * decay
    var = (sigma_proc**2) / (2.0 * theta_i) * (1.0 - at.exp(-2.0 * theta_i * dt))
    return mean, var


# =========================================================
# 4. Build and sample revised ecological-context OU model
# =========================================================

coords = {
    "patient": patient_ids,
    "ecological_context": context_order,
    "cov": cov_cols,
    "obs": np.arange(len(y_obs)),
}

with pm.Model(coords=coords) as ou_context_model:

    # -----------------------------------------------------
    # μ model: participant-level local attractor
    # -----------------------------------------------------
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
        mu_0 + alpha_context[eco_idx] + at.dot(X_cov, beta),
        dims="patient",
    )

    # -----------------------------------------------------
    # θ model: log effective mean-reversion strength
    # -----------------------------------------------------
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
        theta_0 + gamma_context[eco_idx] + at.dot(X_cov, eta),
        dims="patient",
    )

    theta_pat = pm.Deterministic(
        "theta_pat",
        at.exp(log_theta_pat) + 1e-6,
        dims="patient",
    )

    # -----------------------------------------------------
    # Shared process diffusion
    # -----------------------------------------------------
    sigma_proc = pm.HalfNormal("sigma_proc", 1.0)

    # -----------------------------------------------------
    # Likelihood on observed transitions
    # -----------------------------------------------------
    mu_for_obs = mu_pat[pat_idx]
    theta_for_obs = theta_pat[pat_idx]

    mean_tr, var_tr = ou_transition_mean_var(
        y_prev=y_prev,
        mu_i=mu_for_obs,
        theta_i=theta_for_obs,
        sigma_proc=sigma_proc,
        dt=dt_obs,
    )

    # Numerical floor
    var_tr = at.clip(var_tr, 1e-12, np.inf)

    pm.Normal(
        "y_obs",
        mu=mean_tr,
        sigma=at.sqrt(var_tr),
        observed=y_obs,
        dims="obs",
    )

    print("[INFO] Starting MCMC sampling...")

    idata = pm.sample(
        draws=2000,
        tune=3000,
        chains=4,
        cores=4,
        target_accept=0.99,
        random_seed=42,
        idata_kwargs={"log_likelihood": True},
    )

    print("[INFO] Sampling posterior predictive values...")

    idata = pm.sample_posterior_predictive(
        idata,
        var_names=["y_obs"],
        random_seed=42,
        extend_inferencedata=True,
    )

# =========================================================
# 5. Save and summarize
# =========================================================

OUT_IDATA_TMP = OUT_DIR / "ou_revised_ecological_context_idata_ppc_tmp.nc"

# Avoid stale temporary files
if OUT_IDATA_TMP.exists():
    OUT_IDATA_TMP.unlink()

print(f"[INFO] Saving revised-context idata with PPC to temporary file:\n{OUT_IDATA_TMP}")

az.to_netcdf(
    idata,
    OUT_IDATA_TMP,
    engine="netcdf4",
)

# Replace final PPC file atomically if possible
if OUT_IDATA.exists():
    OUT_IDATA.unlink()

OUT_IDATA_TMP.replace(OUT_IDATA)

print(f"[INFO] Saved revised-context idata with PPC to:\n{OUT_IDATA}")

divergences = int(idata.sample_stats["diverging"].sum().values)
print(f"[CHECK] Total divergences: {divergences}")

if divergences > 0:
    print(
        "[WARN] Divergences detected. For final manuscript output, consider rerunning with "
        "target_accept=0.995 and/or tune=4000."
    )

summary_vars = [
    "mu_0",
    "sigma_alpha",
    "alpha_context",
    "theta_0",
    "sigma_gamma",
    "gamma_context",
    "sigma_proc",
]

print(
    az.summary(
        idata,
        var_names=summary_vars,
        round_to=3,
    )
)
