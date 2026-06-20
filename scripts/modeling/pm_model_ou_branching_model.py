import pymc as pm
import numpy as np
import pytensor.tensor as pt   # instead of aesara.tensor as at

Ep = np.load(
    "/TME_OU_Branching/derived_features/Ep_baseline_z.npy"
)
P, K = Ep.shape


# patient_idx: length = number of patient-specific records in your OU–Branching likelihood
# e.g. each patient appears once -> patient_idx = np.arange(P)
# or multiple records per patient -> patient_idx with repeats
patient_idx = ...  # np.array of ints in [0, P-1]

with pm.Model() as ou_branching_model:

    # ---------------------------
    # put design matrix in the model
    # ---------------------------
    E_data = pm.Data("E_patient", Ep)        # shape (P, K)
    idx     = pm.Data("patient_idx", patient_idx)

    # ---------------------------
    # θ_p (OU drift target) regression
    # ---------------------------
    # hyperpriors on regression coefficients
    alpha_theta = pm.Normal("alpha_theta", mu=0.0, sigma=1.0)           # intercept
    beta_theta  = pm.Normal("beta_theta", mu=0.0, sigma=1.0, shape=K)   # slopes
    sigma_theta = pm.HalfNormal("sigma_theta", sigma=1.0)

    # mean θ_p as linear function of Ep
    mu_theta = alpha_theta + at.dot(E_data, beta_theta)   # shape (P,)

    # patient-level θ_p with residual variation
    theta_p = pm.Normal(
        "theta_p", mu=mu_theta, sigma=sigma_theta, shape=P
    )   # shape (P,)

    # When you need θ for each record in the likelihood:
    theta_for_record = theta_p[idx]   # broadcasts to your data rows

    # ---------------------------
    # log b0_p (baseline birth rate) regression
    # ---------------------------
    alpha_b0 = pm.Normal("alpha_b0", mu=0.0, sigma=1.0)
    beta_b0  = pm.Normal("beta_b0",  mu=0.0, sigma=1.0, shape=K)
    sigma_b0 = pm.HalfNormal("sigma_b0", sigma=1.0)

    mu_logb0 = alpha_b0 + at.dot(E_data, beta_b0)        # shape (P,)
    log_b0_p = pm.Normal(
        "log_b0_p", mu=mu_logb0, sigma=sigma_b0, shape=P
    )

    log_b0_for_record = log_b0_p[idx]

    # ---------------------------
    # (Optional) correlation between θ and log b0
    # ---------------------------
    # If you want θ_p and log_b0_p to share structure, you can instead
    # model them jointly as a multivariate regression; keeping it separate
    # here for clarity.

    # ---------------------------
    # The rest of the OU–Branching model goes here
    # ---------------------------
    # Example placeholders:
    # other patient-level params: alpha_p, sigma_p, etc.
    # individual-level latent states x_{i,t} driven by θ_for_record, log_b0_for_record
    #
    # likelihood:
    #   pm.DensityDist("likelihood", logp_fn, observed={...})
