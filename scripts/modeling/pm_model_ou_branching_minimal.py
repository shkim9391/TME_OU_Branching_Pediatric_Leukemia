import pytensor
pytensor.config.cxx = ""
pytensor.config.mode = "FAST_COMPILE"  # no C compilation

import numpy as np
import pymc as pm
import pytensor.tensor as pt


# -------------------------
# 1. Load Ep (patient-level TME design matrix)
# -------------------------
Ep = np.load(
    "/TME_OU_Branching/derived_features/Ep_baseline_z.npy"
)
P, K = Ep.shape
print("Ep shape:", Ep.shape)  # (P, 6)

# For this minimal test, assume one record per patient
patient_idx = np.arange(P, dtype="int32")

# -------------------------
# 2. Build PyMC model
# -------------------------
with pm.Model() as model:

    # just use Ep directly (no pm.Data)
    E_data = pt.as_tensor_variable(Ep)  # shape (P, K)
    idx    = patient_idx                # shape (P,)

    # ----- θ_p regression -----
    alpha_theta = pm.Normal("alpha_theta", 0.0, 1.0)
    beta_theta  = pm.Normal("beta_theta", 0.0, 1.0, shape=K)
    sigma_theta = pm.HalfNormal("sigma_theta", 1.0)

    mu_theta = alpha_theta + pt.dot(E_data, beta_theta)   # (P,)
    theta_p  = pm.Normal("theta_p", mu=mu_theta, sigma=sigma_theta, shape=P)
    theta_for_record = theta_p[idx]   # here: same shape as theta_p

    # ----- log b0_p regression -----
    alpha_b0 = pm.Normal("alpha_b0", 0.0, 1.0)
    beta_b0  = pm.Normal("beta_b0", 0.0, 1.0, shape=K)
    sigma_b0 = pm.HalfNormal("sigma_b0", 1.0)

    mu_logb0 = alpha_b0 + pt.dot(E_data, beta_b0)         # (P,)
    log_b0_p  = pm.Normal("log_b0_p", mu=mu_logb0, sigma=sigma_b0, shape=P)
    log_b0_for_record = log_b0_p[idx]

    # ----- dummy likelihood just to test the graph -----
    # pretend we observe a noisy proxy for theta_p
    y_obs = np.zeros(P, dtype="float32")
    pm.Normal("y", mu=theta_for_record, sigma=1.0, observed=y_obs)

    # sample a few draws to check everything works
    idata = pm.sample(draws=200, tune=200, chains=2, target_accept=0.9)

print(idata.posterior["beta_theta"].mean(dim=("chain","draw")))
print(idata.posterior["beta_b0"].mean(dim=("chain","draw")))
