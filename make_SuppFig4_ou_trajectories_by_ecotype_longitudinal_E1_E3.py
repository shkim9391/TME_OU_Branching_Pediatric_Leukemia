#make_SuppFig4_ou_trajectories_by_ecotype_longitudinal_E1_E3.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# ------------------------------------------------------
# Paths
# ------------------------------------------------------
idata_ppc_path = "/results/ou_ecotype_ou_branching_trace_with_ppc.nc"
patient_csv    = "/Supplementary/SFig1/patient_immune_ecotypes.csv"
posterior_draws_path = "/Supplementary/SFig1/posterior_draws.csv"

# Get longitudinal indexing arrays from your calibration script
import sys
sys.path.insert(0, "/")
from ou_ecotype_ou_branching_calibration import pat_idx, dt_obs, y_obs

outdir = "/Supplementary/SFig4"
os.makedirs(outdir, exist_ok=True)
outpng = os.path.join(outdir, "SuppFig4_OU_trajectory_examples_by_ecotype_longitudinal_E1_E3.png")

# ------------------------------------------------------
# Load idata with PPC
# ------------------------------------------------------
idata = az.from_netcdf(idata_ppc_path)
y_ppc = idata.posterior_predictive["y_obs"].values     # (chain, draw, obs_dim)
y_ppc_2d = y_ppc.reshape((-1, y_ppc.shape[-1]))        # (samples, obs_dim)

# ------------------------------------------------------
# Load patient metadata
# ------------------------------------------------------
meta = pd.read_csv(patient_csv)
ecotype_map = {0: "E1", 1: "E2", 2: "E3", 3: "E4"}
meta["ecotype_label"] = meta["immune_ecotype"].map(ecotype_map)

# ------------------------------------------------------
# Compute n_obs per patient using pat_idx
# ------------------------------------------------------
pat_idx = np.asarray(pat_idx).astype(int)
dt_obs = np.asarray(dt_obs).astype(float)
y_obs  = np.asarray(y_obs).astype(float)

n_pat = meta.shape[0]
counts = np.bincount(pat_idx, minlength=n_pat)  # counts per patient index
meta["n_obs"] = counts

# Longitudinal patients: at least 2 obs
meta_long = meta[meta["n_obs"] >= 2].copy()
print("Longitudinal patients:", meta_long.shape[0], "out of", meta.shape[0])

# ------------------------------------------------------
# Patient-level (mu, log10theta) means for choosing representatives
# ------------------------------------------------------
if not os.path.exists(posterior_draws_path):
    raise FileNotFoundError(f"Missing {posterior_draws_path}. Needed to choose centroid-representative patients.")

df = pd.read_csv(posterior_draws_path)
df["ecotype"] = df["ecotype_label"]

patient_means = (
    df.groupby(["Patient_ID", "ecotype"], observed=True)
      .agg(mu_mean=("mu_sample", "mean"),
           log10_theta_mean=("log10_theta_sample", "mean"))
      .reset_index()
)

# Filter to longitudinal patients only
patient_means = patient_means[patient_means["Patient_ID"].isin(meta_long["Patient_ID"])].copy()

# ------------------------------------------------------
# Choose representative per ecotype among longitudinal patients
# (nearest to ecotype centroid in (mu, log10theta))
# ------------------------------------------------------
rep = {}
available_ecotypes = []
for e in ["E1", "E2", "E3", "E4"]:
    sub = patient_means[patient_means["ecotype"] == e].copy()
    if sub.empty:
        continue
    available_ecotypes.append(e)
    c_mu = sub["mu_mean"].mean()
    c_th = sub["log10_theta_mean"].mean()
    sub["dist2"] = (sub["mu_mean"] - c_mu)**2 + (sub["log10_theta_mean"] - c_th)**2
    rep_pid = sub.sort_values("dist2").iloc[0]["Patient_ID"]
    rep[e] = rep_pid

print("Representative longitudinal patients by ecotype:", rep)

# ------------------------------------------------------
# Helper: get obs indices for a given Patient_ID
# ------------------------------------------------------
pid_list = meta["Patient_ID"].tolist()

def patient_obs_indices(pid_str: str):
    p_i = pid_list.index(pid_str)          # patient index used in pat_idx
    idx = np.where(pat_idx == p_i)[0]      # obs indices
    return np.sort(idx)

# ------------------------------------------------------
# Plot
# ------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
})

ecotype_colors = {"E1":"#4C72B0","E2":"#55A868","E3":"#C44E52","E4":"#8172B3"}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
ecotypes_to_plot = ["E1", "E3"]
panel_labels = ["A", "B"]

for ax, lab, e in zip(axes, panel_labels, ecotypes_to_plot):
    ax.text(0.01, 0.99, lab, transform=ax.transAxes, va="top", ha="left",
            fontsize=14, fontweight="bold")

    pid = rep[e]
    idx = patient_obs_indices(pid)
    idx = np.sort(idx)

    t = np.cumsum(dt_obs[idx])
    y = y_obs[idx]

    yrep = y_ppc_2d[:, idx]
    mean = np.nanmean(yrep, axis=0)
    lo = np.nanpercentile(yrep, 5, axis=0)
    hi = np.nanpercentile(yrep, 95, axis=0)

    ax.fill_between(t, lo, hi, alpha=0.25, color=ecotype_colors[e], linewidth=0)
    ax.plot(t, mean, linewidth=2.0, color=ecotype_colors[e], label="PPC mean")
    ax.plot(t, y, marker="o", markersize=4.5, linewidth=1.2, color="black", label="Observed")

    ax.set_title(f"{e}: {pid}", pad=6)
    ax.set_xlabel("Cumulative time (arb. units)")
    ax.set_ylabel(r"$y(t)$")
    ax.grid(True, linestyle="--", alpha=0.35)

# single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))

fig.tight_layout()
plt.savefig(outpng, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {outpng}")
