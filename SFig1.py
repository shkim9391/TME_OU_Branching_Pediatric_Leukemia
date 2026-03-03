#SFig1.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Paths
# ------------------------------------------------------
posterior_path = "/SFig1/posterior_draws.csv"
outdir = "/Supplementary/SFig1"
os.makedirs(outdir, exist_ok=True)

# ------------------------------------------------------
# Load data
# ------------------------------------------------------
df = pd.read_csv(posterior_path)
df["ecotype"] = df["ecotype_label"]

ecotype_order = ["E1", "E2", "E3", "E4"]

# ------------------------------------------------------
# Patient-level posterior means (for jittered points)
# ------------------------------------------------------
patient_means = (
    df.groupby(["Patient_ID", "ecotype"], observed=True)
      .agg(
          mu_mean=("mu_sample", "mean"),
          log10_theta_mean=("log10_theta_sample", "mean")
      )
      .reset_index()
)

# ------------------------------------------------------
# Plot style
# ------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
})

ecotype_colors = {
    "E1": "#4C72B0",
    "E2": "#55A868",
    "E3": "#C44E52",
    "E4": "#8172B3",
}

# ------------------------------------------------------
# Helper function
# ------------------------------------------------------
def violin_panel(ax, value_col, mean_col, ylabel, title, panel_label):
    xs = np.arange(len(ecotype_order)) + 1
    data = [df.loc[df.ecotype == e, value_col].values for e in ecotype_order]

    vp = ax.violinplot(
        data, positions=xs, widths=0.7,
        showmeans=False, showmedians=False, showextrema=False
    )

    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(ecotype_colors[ecotype_order[i]])
        body.set_edgecolor("black")
        body.set_alpha(0.7)

        median = np.median(data[i])
        ax.hlines(median, xs[i]-0.22, xs[i]+0.22, lw=1.6, color="black", zorder=4)

    rng = np.random.default_rng(42)
    for i, e in enumerate(ecotype_order):
        sub = patient_means[patient_means.ecotype == e]
        xj = xs[i] + rng.uniform(-0.15, 0.15, size=len(sub))
        ax.scatter(
            xj, sub[mean_col],
            s=16, color="black",
            edgecolor="white", linewidth=0.4, zorder=3
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(ecotype_order)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.text(0.01, 0.99, panel_label, transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top", ha="left")

# ------------------------------------------------------
# Build figure
# ------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

violin_panel(
    axes[0],
    value_col="mu_sample",
    mean_col="mu_mean",
    ylabel=r"OU drift mean $\mu$",
    title=r"Posterior drift mean $\mu$ by TME ecotype",
    panel_label="A",
)

violin_panel(
    axes[1],
    value_col="log10_theta_sample",
    mean_col="log10_theta_mean",
    ylabel=r"$\log_{10}$ selection strength $\theta$",
    title=r"Posterior selection strength $\theta$ by TME ecotype",
    panel_label="B",
)

fig.tight_layout()
outpath = os.path.join(outdir, "SuppFig1_posterior_mu_theta_by_ecotype.png")
plt.savefig(outpath, bbox_inches="tight")
plt.close(fig)

print(f"Saved {outpath}")
