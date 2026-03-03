#SFig2.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Paths
# ------------------------------------------------------
posterior_path = "/SFig1/posterior_draws.csv"
outdir = "/Supplementary/SFig2"
os.makedirs(outdir, exist_ok=True)

# ------------------------------------------------------
# Load posterior draws
# ------------------------------------------------------
df = pd.read_csv(posterior_path)

# Normalize ecotype labels
if "ecotype_label" in df.columns:
    df["ecotype"] = df["ecotype_label"]
elif "ecotype" not in df.columns:
    raise ValueError("Expected 'ecotype_label' or 'ecotype' in posterior_draws.csv")

ecotype_order = ["E1", "E2", "E3", "E4"]

# ------------------------------------------------------
# Patient-level posterior means (point per patient)
# ------------------------------------------------------
# Uses posterior draws to compute per-patient posterior mean for μ and log10(θ)
patient_means = (
    df.groupby(["Patient_ID", "ecotype"], observed=True)
      .agg(
          mu_mean=("mu_sample", "mean"),
          log10_theta_mean=("log10_theta_sample", "mean"),
      )
      .reset_index()
)

# Optional: bring in diagnosis if present (kept simple; not used for shapes unless you want)
if "diagnosis" in df.columns:
    diag_map = (
        df[["Patient_ID", "diagnosis"]]
        .drop_duplicates()
        .set_index("Patient_ID")["diagnosis"]
        .to_dict()
    )
    patient_means["diagnosis"] = patient_means["Patient_ID"].map(diag_map)
else:
    patient_means["diagnosis"] = "NA"

# ------------------------------------------------------
# Ecotype centroids + spread (across patients, not across draws)
# ------------------------------------------------------
centroids = (
    patient_means.groupby("ecotype", observed=True)
    .agg(
        mu_cent=("mu_mean", "mean"),
        mu_sd=("mu_mean", "std"),
        th_cent=("log10_theta_mean", "mean"),
        th_sd=("log10_theta_mean", "std"),
        n=("Patient_ID", "nunique"),
    )
    .reindex(ecotype_order)
    .reset_index()
)

# If any SD is NaN (e.g., only 1 patient), set to 0 for plotting
centroids["mu_sd"] = centroids["mu_sd"].fillna(0.0)
centroids["th_sd"] = centroids["th_sd"].fillna(0.0)

# ------------------------------------------------------
# Plot style
# ------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
})

ecotype_colors = {
    "E1": "#4C72B0",
    "E2": "#55A868",
    "E3": "#C44E52",
    "E4": "#8172B3",
}

# ------------------------------------------------------
# Make figure
# ------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))

# --------------------------
# Panel A: scatter + centroid
# --------------------------
ax = axes[0]
for e in ecotype_order:
    sub = patient_means[patient_means["ecotype"] == e]
    ax.scatter(
        sub["mu_mean"].values,
        sub["log10_theta_mean"].values,
        s=22,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
        label=e,
        color=ecotype_colors[e],
        zorder=2,
    )

# Centroids with error bars (±1 SD across patients)
for _, r in centroids.iterrows():
    e = r["ecotype"]
    ax.errorbar(
        r["mu_cent"],
        r["th_cent"],
        xerr=r["mu_sd"],
        yerr=r["th_sd"],
        fmt="o",
        markersize=7,
        capsize=4,
        elinewidth=1.4,
        markeredgecolor="black",
        markeredgewidth=0.8,
        color=ecotype_colors[e],
        zorder=3,
    )

ax.set_xlabel(r"Patient posterior mean $\hat{\mu}$")
ax.set_ylabel(r"Patient posterior mean $\widehat{\log_{10}\theta}$")
ax.set_title(r"Joint TME-modulated OU parameters by ecotype", fontsize=12, pad=6)
ax.grid(True, linestyle="--", alpha=0.35)
ax.text(0.01, 0.99, "A", transform=ax.transAxes, va="top", ha="left",
        fontsize=14, fontweight="bold")

# Legend (compact, inside)

ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", title="Ecotype")

# --------------------------
# Panel B: density context (hexbin) + points
# --------------------------
ax = axes[1]

# Light hexbin per ecotype to show density (Matplotlib-only)
# We plot a hexbin for each ecotype with a small alpha so they can overlap.
# (No explicit color-setting beyond ecotype colors.)
for e in ecotype_order:
    sub = patient_means[patient_means["ecotype"] == e]
    ax.hexbin(
        sub["mu_mean"].values,
        sub["log10_theta_mean"].values,
        gridsize=25,
        mincnt=1,
        linewidths=0.0,
        alpha=0.22,
        cmap=None,   # keep default colormap behavior
    )
    # Overlay points again for crispness
    ax.scatter(
        sub["mu_mean"].values,
        sub["log10_theta_mean"].values,
        s=18,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
        color=ecotype_colors[e],
        zorder=2,
    )

# Add centroid markers again
for _, r in centroids.iterrows():
    e = r["ecotype"]
    ax.scatter(
        r["mu_cent"],
        r["th_cent"],
        s=70,
        marker="X",
        edgecolor="black",
        linewidth=0.8,
        color=ecotype_colors[e],
        zorder=3,
    )

ax.set_xlabel(r"Patient posterior mean $\hat{\mu}$")
ax.set_ylabel(r"Patient posterior mean $\widehat{\log_{10}\theta}$")
ax.set_title(r"Ecotype density + centroid summary", fontsize=12, pad=6)
ax.grid(True, linestyle="--", alpha=0.35)
ax.text(0.01, 0.99, "B", transform=ax.transAxes, va="top", ha="left",
        fontsize=14, fontweight="bold")

fig.tight_layout()
outpath = os.path.join(outdir, "SuppFig2_joint_mu_theta_by_ecotype.png")
plt.savefig(outpath, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {outpath}")
