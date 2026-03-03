#plot_mu_thetha_ecotype_violin.py

"""
Violin plots of posterior μ and θ by immune ecotype.
"""

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/")
TRACE_PATH = ROOT / "results" / "ou_ecotype_ou_branching_trace.nc"

# ---------------------------------------------------------
# 1. Load posterior
# ---------------------------------------------------------
idata = az.from_netcdf(TRACE_PATH)

mu_0 = idata.posterior["mu_0"].values             # (chains, draws)
alpha = idata.posterior["alpha_ecotype"].values   # (chains, draws, K)
theta_0 = idata.posterior["theta_0"].values
gamma = idata.posterior["gamma_ecotype"].values   # (chains, draws, K)

chains, draws = mu_0.shape
K = alpha.shape[-1]

# Flatten chains+draws for convenience
mu_0_flat = mu_0.reshape(-1)
theta_0_flat = theta_0.reshape(-1)

mu_samples = []
theta_samples = []
ecotype_labels = []

for k in range(K):
    alpha_k = alpha[..., k].reshape(-1)   # (chains*draws,)
    gamma_k = gamma[..., k].reshape(-1)

    # μ_k and θ_k samples across all chains/draws
    mu_k = mu_0_flat + alpha_k
    theta_k = np.exp(theta_0_flat + gamma_k)

    mu_samples.append(mu_k)
    theta_samples.append(theta_k)
    ecotype_labels.append(f"Ecotype {k+1}")

# ---------------------------------------------------------
# 2. Violin plot: μ by ecotype
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.violinplot(
    mu_samples,
    positions=range(K),
    showmeans=True,
    showextrema=False,
)
plt.xticks(range(K), ecotype_labels)
plt.ylabel("Drift mean μ")
plt.title("Posterior μ by immune ecotype")
plt.tight_layout()

mu_fig_path = ROOT / "results" / "violin_mu_by_ecotype.png"
plt.savefig(mu_fig_path, dpi=600)
plt.show()
print("Saved μ violin figure to:", mu_fig_path)

# ---------------------------------------------------------
# 3. Violin plot: θ by ecotype (log10 scale for readability)
# ---------------------------------------------------------
theta_samples_log10 = [np.log10(ts) for ts in theta_samples]

plt.figure(figsize=(7, 5))
plt.violinplot(
    theta_samples_log10,
    positions=range(K),
    showmeans=True,
    showextrema=False,
)
plt.xticks(range(K), ecotype_labels)
plt.ylabel("log10(θ)")
plt.title("Posterior selection strength θ by immune ecotype")
plt.tight_layout()

theta_fig_path = ROOT / "results" / "violin_theta_log10_by_ecotype.png"
plt.savefig(theta_fig_path, dpi=600)
plt.show()
print("Saved θ violin figure to:", theta_fig_path)
