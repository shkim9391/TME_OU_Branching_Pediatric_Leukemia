#plot_mu_theta_scatter_with_centroids.py

"""
Patient-level (μ, θ) scatter with:
- color: immune ecotype
- marker: diagnosis (B-ALL/T-ALL/AML)
- labels: selected outliers
- ecotype-wise centroids (μ̄, θ̄) with error bars
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path("/")
TRACE_PATH = ROOT / "results" / "ou_ecotype_ou_branching_trace.nc"
PATIENT_PATH = ROOT / "patient_immune_ecotypes.csv"

LABEL_PATIENTS = ["P90", "P91", "P92", "P93", "P94", "P96", "P97", "P100", "P54", "P89"]

# --------------------------- load data ---------------------------
idata = az.from_netcdf(TRACE_PATH)
patients = pd.read_csv(PATIENT_PATH)

mu_pat = idata.posterior["mu_pat"].values        # (chains, draws, N_pat)
theta_pat = idata.posterior["theta_pat"].values  # (chains, draws, N_pat)

mu_mean = mu_pat.mean(axis=(0, 1))
theta_mean = theta_pat.mean(axis=(0, 1))
log10_theta_mean = np.log10(theta_mean)

eco = patients["immune_ecotype"].values.astype(int)
diagnosis_raw = patients["diagnosis"].fillna("").astype(str).values
patient_ids = patients["Patient_ID"].astype(str).values

def diag_group(d: str) -> str:
    d_low = d.lower()
    if "myeloid" in d_low:
        return "AML"
    if "t-cell" in d_low or "t cell" in d_low:
        return "T-ALL"
    return "B-ALL"

diag_grp = np.array([diag_group(d) for d in diagnosis_raw])

# --- ecotype display offset (0-based -> 1-based) ---
ECO_OFFSET = 1 if eco.min() == 0 else 0
def eco_disp(k: int) -> int:
    return int(k) + ECO_OFFSET

# --- stable color mapping (use rank index, not raw k) ---
ecotypes = np.sort(np.unique(eco))
cmap = plt.cm.tab10
eco_colors = {k: cmap(i / max(1, len(ecotypes) - 1)) for i, k in enumerate(ecotypes)}
diag_groups = ["B-ALL", "T-ALL", "AML"]
markers = {"B-ALL": "o", "T-ALL": "s", "AML": "D"}

plt.figure(figsize=(7, 6))

# --------------------------- scatter points ---------------------------
for k in ecotypes:
    idx_k = np.where(eco == k)[0]
    for g in diag_groups:
        idx = idx_k[diag_grp[idx_k] == g]
        if idx.size == 0:
            continue
        plt.scatter(
            mu_mean[idx],
            log10_theta_mean[idx],
            color=eco_colors[k],
            marker=markers[g],
            s=40,
            alpha=0.9,
            edgecolor="k",
            linewidths=0.3,
        )

# --------------------------- label outliers ---------------------------
for pid in LABEL_PATIENTS:
    m = np.where(patient_ids == pid)[0]
    if m.size == 0:
        continue
    i = m[0]
    x = mu_mean[i]
    y = log10_theta_mean[i]
    plt.text(
        x,
        y,
        pid,
        fontsize=8,
        ha="left",
        va="bottom",
        color="black",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, linewidth=0.2),
    )

# --------------------------- ecotype centroids ---------------------------
for k in ecotypes:
    idx = np.where(eco == k)[0]
    if idx.size == 0:
        continue
    mu_k = mu_mean[idx]
    log10_theta_k = log10_theta_mean[idx]

    mu_bar = mu_k.mean()
    mu_sd = mu_k.std(ddof=1)
    th_bar = log10_theta_k.mean()
    th_sd = log10_theta_k.std(ddof=1)

    # centroid marker
    plt.errorbar(
        mu_bar,
        th_bar,
        xerr=mu_sd,
        yerr=th_sd,
        fmt="o",
        ms=10,
        mfc="yellow",
        mec="k",
        ecolor="k",
        elinewidth=1.2,
        capsize=5,
        zorder=5,
    )
    plt.text(
        mu_bar,
        th_bar,
        f"E{eco_disp(k)}",
        fontsize=8,
        ha="center",
        va="center",
        color="black",
        fontweight="bold",
    )

# --------------------------- legends & labels ---------------------------
plt.xlabel("Drift mean μ (patient posterior mean)")
plt.ylabel("log10(θ) (patient posterior mean)")
plt.title("Patient-level OU parameters\ncolor: ecotype, marker: diagnosis, centroids per ecotype")

# legend: ecotype colors
eco_handles = [
    Line2D([0], [0], marker="o", color="w",
           label=f"Ecotype {eco_disp(k)}",
           markerfacecolor=eco_colors[k],
           markeredgecolor="k",
           markersize=8)
    for k in ecotypes
]
legend1 = plt.legend(handles=eco_handles, title="Immune ecotype", loc="upper right")
plt.gca().add_artist(legend1)

# legend: diagnosis markers
diag_handles = [
    Line2D([0], [0], marker=markers[g], color="w",
           label=g,
           markerfacecolor="gray",
           markeredgecolor="k",
           markersize=8)
    for g in diag_groups
]
plt.legend(handles=diag_handles, title="Diagnosis", loc="lower left")

plt.tight_layout()

out_path = ROOT / "results" / "scatter_mu_theta_ecotype_centroids.png"
plt.savefig(out_path, dpi=600)
plt.show()
print("Saved μ–θ scatter with centroids to:", out_path)
