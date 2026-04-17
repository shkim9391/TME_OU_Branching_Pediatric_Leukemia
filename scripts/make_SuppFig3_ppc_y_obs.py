#make_SuppFig3_ppc_y_obs.py

import os
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# ------------------------------------------------------
# Paths
# ------------------------------------------------------
idata_ppc_path = "/results/ou_ecotype_ou_branching_trace_with_ppc.nc"
outdir = "/Supplementary/SFig3"
os.makedirs(outdir, exist_ok=True)

outpng = os.path.join(outdir, "SuppFig3_posterior_predictive_checks_y_obs.png")

# ------------------------------------------------------
# Load idata with PPC
# ------------------------------------------------------
idata = az.from_netcdf(idata_ppc_path)
print("Groups:", idata.groups())

# Observed
y_obs = np.asarray(idata.observed_data["y_obs"].values).ravel()

# Posterior predictive (chain, draw, obs)
y_ppc = idata.posterior_predictive["y_obs"].values
y_ppc_2d = y_ppc.reshape((-1, y_ppc.shape[-1]))

ppc_mean = y_ppc_2d.mean(axis=0)
ppc_sd   = y_ppc_2d.std(axis=0, ddof=1)
ppc_sd = np.where(ppc_sd == 0, np.nan, ppc_sd)

z = (y_obs - ppc_mean) / ppc_sd
z = z[np.isfinite(z)]

# ------------------------------------------------------
# Style
# ------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
})

# ------------------------------------------------------
# Plot
# ------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))

# A: Distribution overlay
ax = axes[0]
lo = np.nanpercentile(y_obs, 0.5)
hi = np.nanpercentile(y_obs, 99.5)
bins = np.linspace(lo, hi, 40)

ax.hist(y_obs, bins=bins, density=True, histtype="step", linewidth=2.0,
        label="Observed $y_{obs}$")

rng = np.random.default_rng(0)
n_overlay = min(50, y_ppc_2d.shape[0])
idx = rng.choice(y_ppc_2d.shape[0], size=n_overlay, replace=False)
for k in idx:
    ax.hist(y_ppc_2d[k, :], bins=bins, density=True, histtype="step",
            linewidth=0.8, alpha=0.18)

ax.hist(y_ppc_2d.mean(axis=0), bins=bins, density=True, histtype="step",
        linewidth=2.0, linestyle="--", label="PPC mean")

ax.set_title("Posterior predictive distribution vs observed", pad=6)
ax.set_xlabel(r"$y_{obs}$")
ax.set_ylabel("Density")
ax.grid(True, linestyle="--", alpha=0.35)
ax.text(0.01, 0.99, "A", transform=ax.transAxes, va="top", ha="left",
        fontsize=14, fontweight="bold")
ax.legend(frameon=False, loc="upper right")

# B: z-score calibration
ax = axes[1]
ax.hist(z, bins=40, density=True, histtype="step", linewidth=2.0, label="PPC z-scores")

x = np.linspace(-4, 4, 300)
ref = (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
ax.plot(x, ref, linewidth=2.0, linestyle="--", label="N(0,1) reference")

ax.set_title("Calibration via standardized residuals", pad=6)
ax.set_xlabel(r"$z_i=(y_i-\mathbb{E}[y_i^{rep}])/\mathrm{SD}(y_i^{rep})$")
ax.set_ylabel("Density")
ax.grid(True, linestyle="--", alpha=0.35)
ax.text(0.01, 0.99, "B", transform=ax.transAxes, va="top", ha="left",
        fontsize=14, fontweight="bold")
ax.legend(frameon=False, loc="upper right")

fig.tight_layout()
plt.savefig(outpng, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {outpng}")
