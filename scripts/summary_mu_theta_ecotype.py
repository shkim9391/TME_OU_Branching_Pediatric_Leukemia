#summary_mu_theta_ecotype.py

import numpy as np
import pandas as pd
import arviz as az

ROOT = "/"
TRACE_PATH = ROOT + "/results/ou_ecotype_ou_branching_trace.nc"

# Load trace
idata = az.from_netcdf(TRACE_PATH)

mu_0 = idata.posterior["mu_0"].values            # (chains, draws)
alpha = idata.posterior["alpha_ecotype"].values  # (chains, draws, K)
theta_0 = idata.posterior["theta_0"].values
gamma = idata.posterior["gamma_ecotype"].values  # (chains, draws, K)

K = alpha.shape[-1]

rows = []
for k in range(K):
    mu_k = mu_0 + alpha[..., k]
    theta_k = np.exp(theta_0 + gamma[..., k])

    rows.append({
        "ecotype": k,
        "mu_mean": float(mu_k.mean()),
        "mu_hdi_low": float(np.percentile(mu_k, 2.5)),
        "mu_hdi_high": float(np.percentile(mu_k, 97.5)),
        "theta_mean": float(theta_k.mean()),
        "theta_hdi_low": float(np.percentile(theta_k, 2.5)),
        "theta_hdi_high": float(np.percentile(theta_k, 97.5)),
    })

summary_df = pd.DataFrame(rows)

# >>> ADD THESE LINES <<<
print(summary_df)

out_path = ROOT + "/results/mu_theta_by_ecotype.csv"
summary_df.to_csv(out_path, index=False)
print("\nSaved to:", out_path)
