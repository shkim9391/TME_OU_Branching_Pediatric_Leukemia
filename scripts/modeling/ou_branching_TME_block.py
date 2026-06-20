from pathlib import Path
import numpy as np

import pytensor
# disable C++ backend so it runs on your Mac without compiler headaches
pytensor.config.cxx = ""
pytensor.config.mode = "FAST_COMPILE"

import pymc as pm
import pytensor.tensor as pt


# ---------------------------------------------------------------------
# 1. Load patient-level TME design matrix Ep
# ---------------------------------------------------------------------
ROOT = Path("/TME_OU_Branching")
FEAT_DIR = ROOT / "derived_features"

Ep_path = FEAT_DIR / "Ep_baseline_z.npy"
Ep = np.load(Ep_path)     # shape (P, K)
P, K = Ep.shape
print(f"Loaded Ep from {Ep_path}")
print("Ep shape:", Ep.shape)   # e.g. (100, 6)

# For now assume one record per patient; if you have more later,
# build a longer patient_idx with repeats.
patient_idx = np.arange(P, dtype="int32")


# ---------------------------------------------------------------------
# 2. Build PyMC model: TME → θ_p, log b0_p
# ---------------------------------------------------------------------
with pm.Model() as ou_branching_model:

    # Put Ep into the computation graph
    E_data = pt.as_tensor_variable(Ep)   # (P, K)

    # ------------------------------------------------
    # θ_p regression (OU drift target)
    # ------------------------------------------------
    alpha_theta = pm.Normal("alpha_theta", mu=0.0, sigma=1.0)
    beta_theta  = pm.Normal("beta_theta", mu=0.0, sigma=1.0, shape=K)
    sigma_theta = pm.HalfNormal("sigma_theta", sigma=1.0)

    # patient-level means
    mu_theta = alpha_theta + pt.dot(E_data, beta_theta)   # (P,)
    theta_p  = pm.Normal("theta_p", mu=mu_theta, sigma=sigma_theta, shape=P)

    # For record-level indexing (if needed)
    theta_for_record = theta_p[patient_idx]

    # ------------------------------------------------
    # log b0_p regression (baseline birth rate)
    # ------------------------------------------------
    alpha_b0 = pm.Normal("alpha_b0", mu=0.0, sigma=1.0)
    beta_b0  = pm.Normal("beta_b0",  mu=0.0, sigma=1.0, shape=K)
    sigma_b0 = pm.HalfNormal("sigma_b0", sigma=1.0)

    mu_logb0 = alpha_b0 + pt.dot(E_data, beta_b0)         # (P,)
    log_b0_p = pm.Normal("log_b0_p", mu=mu_logb0, sigma=sigma_b0, shape=P)

    # Optional: positive birth rates
    b0_p = pm.Deterministic("b0_p", pt.exp(log_b0_p))

    log_b0_for_record = log_b0_p[patient_idx]

    # ------------------------------------------------
    # 3. DUMMY likelihood (for testing only!)
    # ------------------------------------------------
    # This is just to make pm.sample() work. It treats theta_p as if it's
    # "observed" around zeros. Replace this whole block with your real
    # OU–Branching latent states and observation model.
    #
    y_obs = np.zeros(P, dtype="float32")
    pm.Normal("y_dummy", mu=theta_for_record, sigma=1.0, observed=y_obs)

    # ------------------------------------------------
    # 4. Sampling – just to test the block
    # ------------------------------------------------
    print("Sampling… (this is just testing the TME block, not the real model)")
    idata = pm.sample(
        draws=400,
        tune=400,
        chains=2,
        target_accept=0.9,
        random_seed=123,
        progressbar=True,
    )

    # Quick look at TME coefficients
    beta_theta_mean = idata.posterior["beta_theta"].mean(dim=("chain", "draw")).values
    beta_b0_mean    = idata.posterior["beta_b0"].mean(dim=("chain", "draw")).values

    print("\nPosterior mean beta_theta:", beta_theta_mean)
    print("Posterior mean beta_b0:   ", beta_b0_mean)
