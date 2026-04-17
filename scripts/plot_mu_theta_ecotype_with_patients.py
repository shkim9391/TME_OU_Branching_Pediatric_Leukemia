#plot_mu_theta_ecotype_with_patients.py

"""
Violin plots of μ and θ by ecotype with jittered patient-level posterior means.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/")
TRACE_PATH = ROOT / "results" / "ou_ecotype_ou_branching_trace.nc"
PATIENT_PATH = ROOT / "patient_immune_ecotypes.csv"

# ---------------------------------------------------------
# 1. Load posterior + patient info
# ---------------------------------------------------------
idata = az.from_netcdf(TRACE_PATH)
patients = pd.read_csv(PATIENT_PATH)

# Patient-level parameters: already include ecotype + covariates
mu_pat = idata.posterior["mu_pat"].values      # (chains, draws, N_pat)
theta_pat = idata.posterior["theta_pat"].values  # (chains, draws, N_pat)

chains, draws, N_pat = mu_pat.shape
eco = patients["immune_ecotype"].values.astype(int)
K = np.unique(eco).size
ecotype_labels = [f"Ecotype {k+1}" for k in range(K)]

# Posterior means per patient
mu_pat_mean = mu_pat.mean(axis=(0, 1))            # (N_pat,)
theta_pat_mean = theta_pat.mean(axis=(0, 1))      # (N_pat,)
log10_theta_pat_mean = np.log10(theta_pat_mean)

# ---------------------------------------------------------
# 2. Group-level samples for violins (from hyperparameters)
# ---------------------------------------------------------
mu_0 = idata.posterior["mu_0"].values             # (chains, draws)
alpha = idata.posterior["alpha_ecotype"].values   # (chains, draws, K)
theta_0 = idata.posterior["theta_0"].values
gamma = idata.posterior["gamma_ecotype"].values   # (chains, draws, K)

mu_0_flat = mu_0.reshape(-1)
theta_0_flat = theta_0.reshape(-1)

mu_samples = []
theta_samples_log10 = []

for k in range(K):
    alpha_k = alpha[..., k].reshape(-1)
    gamma_k = gamma[..., k].reshape(-1)

    mu_k = mu_0_flat + alpha_k
    theta_k = np.exp(theta_0_flat + gamma_k)

    mu_samples.append(mu_k)
    theta_samples_log10.append(np.log10(theta_k))

# ---------------------------------------------------------
# 3. μ: violins + jittered patient means
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
parts = plt.violinplot(
    mu_samples,
    positions=range(K),
    showmeans=True,
    showextrema=False,
)

# soften violins
for pc in parts["bodies"]:
    pc.set_alpha(0.3)

# jittered patient points
rng = np.random.default_rng(0)
for k in range(K):
    idx = np.where(eco == k)[0]
    xs = k + rng.normal(scale=0.04, size=len(idx))
    ys = mu_pat_mean[idx]
    plt.scatter(xs, ys, s=14, alpha=0.8, edgecolor="k", linewidths=0.2)

plt.xticks(range(K), ecotype_labels)
plt.ylabel("Drift mean μ")
plt.title("Posterior μ by immune ecotype\n(violins + patient posterior means)")
plt.tight_layout()

out_mu = ROOT / "results" / "violin_mu_with_patients.png"
plt.savefig(out_mu, dpi=600)
plt.show()
print("Saved μ plot with patients to:", out_mu)

# ---------------------------------------------------------
# 4. θ: violins + jittered patient means (log10 scale)
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
parts = plt.violinplot(
    theta_samples_log10,
    positions=range(K),
    showmeans=True,
    showextrema=False,
)
for pc in parts["bodies"]:
    pc.set_alpha(0.3)

rng = np.random.default_rng(1)
for k in range(K):
    idx = np.where(eco == k)[0]
    xs = k + rng.normal(scale=0.04, size=len(idx))
    ys = log10_theta_pat_mean[idx]
    plt.scatter(xs, ys, s=14, alpha=0.8, edgecolor="k", linewidths=0.2)

plt.xticks(range(K), ecotype_labels)
plt.ylabel("log10(θ)")
plt.title("Posterior selection strength θ by immune ecotype\n(violins + patient posterior means)")
plt.tight_layout()

out_theta = ROOT / "results" / "violin_theta_log10_with_patients.png"
plt.savefig(out_theta, dpi=600)
plt.show()
print("Saved θ plot with patients to:", out_theta)
