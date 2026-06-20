from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arviz as az


# ============================================================
# User settings
# ============================================================

BASE_DIR = Path(
    "/TME_OU_Branching"
)

POSSIBLE_INPUT_DIRS = [
    BASE_DIR,
    BASE_DIR / "results",
    BASE_DIR / "data",
    BASE_DIR / "Figure_4",
    BASE_DIR / "BMC_Bioinformatics",
    Path.cwd(),
]

OUT_DIR = BASE_DIR / "BMC_Bioinformatics" / "Supplementary_Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = "SuppFigS3_mcmc_diagnostics"

TRACE_FILES = [
    "tme_ou_ecotype_idata_E88_FINAL.nc",
    "tme_ou_ecotype_idata.nc",
    "ou_ecotype_ou_branching_trace.nc",
    "tme_ou_ecotype_trace.nc",
    "idata.nc",
    "trace.nc",
]


# ============================================================
# Helper functions
# ============================================================

def find_file(possible_names, required=True):
    """Find first matching file across possible directories."""
    for input_dir in POSSIBLE_INPUT_DIRS:
        for name in possible_names:
            candidate = input_dir / name
            if candidate.exists():
                print(f"Found: {candidate}")
                return candidate

    if required:
        raise FileNotFoundError(
            "Could not find any of these files:\n"
            + "\n".join(possible_names)
            + "\nSearched in:\n"
            + "\n".join(str(x) for x in POSSIBLE_INPUT_DIRS)
        )

    return None


def flatten_values(x):
    """Flatten xarray/NumPy values and remove non-finite entries."""
    arr = np.asarray(x).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def add_panel_label(ax, label):
    ax.text(
        -0.10,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def choose_trace_variables(idata, max_vars=4):
    """
    Choose representative variables for trace plots.
    Preference order matches likely model variable names.
    """
    posterior_vars = list(idata.posterior.data_vars)

    preferred_exact = [
        "mu_hyper",
        "mu_global",
        "mu_ecotype",
        "mu_context",
        "theta_hyper",
        "log_theta_hyper",
        "eta_context",
        "sigma_proc",
        "sigma_obs",
        "tau_mu",
        "tau_theta",
    ]

    preferred_contains = [
        "mu",
        "theta",
        "sigma",
        "tau",
        "eta",
    ]

    selected = []

    for v in preferred_exact:
        if v in posterior_vars and v not in selected:
            selected.append(v)

    for key in preferred_contains:
        for v in posterior_vars:
            if key.lower() in v.lower() and v not in selected:
                selected.append(v)
            if len(selected) >= max_vars:
                return selected[:max_vars]

    if len(selected) < max_vars:
        for v in posterior_vars:
            if v not in selected:
                selected.append(v)
            if len(selected) >= max_vars:
                break

    return selected[:max_vars]


def summarize_variable_scalar(trace_da):
    """
    Convert a posterior variable to one scalar series per chain/draw.
    If the variable has additional dimensions, average over those dims.
    """
    da = trace_da

    extra_dims = [d for d in da.dims if d not in ["chain", "draw"]]
    if extra_dims:
        da = da.mean(dim=extra_dims)

    return da


# ============================================================
# Load trace
# ============================================================

trace_file = find_file(TRACE_FILES)

print(f"\nLoading inference data: {trace_file}")
idata = az.from_netcdf(trace_file)

if not hasattr(idata, "posterior"):
    raise ValueError("The inference data object does not contain a posterior group.")

posterior_vars = list(idata.posterior.data_vars)
print("\nPosterior variables:")
print(posterior_vars)

n_chains = int(idata.posterior.sizes.get("chain", 1))
n_draws = int(idata.posterior.sizes.get("draw", 0))

print(f"\nChains: {n_chains}")
print(f"Posterior draws per chain: {n_draws}")


# ============================================================
# Compute diagnostics
# ============================================================

print("\nComputing R-hat, bulk ESS, and tail ESS...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    rhat_ds = az.rhat(idata, method="rank")
    ess_bulk_ds = az.ess(idata, method="bulk")
    ess_tail_ds = az.ess(idata, method="tail")

diagnostic_rows = []

for var in posterior_vars:
    rhat_vals = flatten_values(rhat_ds[var].values) if var in rhat_ds else np.array([])
    ess_bulk_vals = flatten_values(ess_bulk_ds[var].values) if var in ess_bulk_ds else np.array([])
    ess_tail_vals = flatten_values(ess_tail_ds[var].values) if var in ess_tail_ds else np.array([])

    diagnostic_rows.append({
        "variable": var,
        "n_parameters": int(max(len(rhat_vals), len(ess_bulk_vals), len(ess_tail_vals))),
        "rhat_median": np.nanmedian(rhat_vals) if len(rhat_vals) else np.nan,
        "rhat_max": np.nanmax(rhat_vals) if len(rhat_vals) else np.nan,
        "n_rhat_gt_1_01": int(np.sum(rhat_vals > 1.01)) if len(rhat_vals) else 0,
        "n_rhat_gt_1_05": int(np.sum(rhat_vals > 1.05)) if len(rhat_vals) else 0,
        "ess_bulk_median": np.nanmedian(ess_bulk_vals) if len(ess_bulk_vals) else np.nan,
        "ess_bulk_min": np.nanmin(ess_bulk_vals) if len(ess_bulk_vals) else np.nan,
        "ess_tail_median": np.nanmedian(ess_tail_vals) if len(ess_tail_vals) else np.nan,
        "ess_tail_min": np.nanmin(ess_tail_vals) if len(ess_tail_vals) else np.nan,
    })

diag_df = pd.DataFrame(diagnostic_rows)

diag_path = OUT_DIR / "SuppFigS3_mcmc_diagnostics_summary.csv"
diag_df.to_csv(diag_path, index=False)

all_rhat = np.concatenate([
    flatten_values(rhat_ds[var].values)
    for var in rhat_ds.data_vars
])

all_ess_bulk = np.concatenate([
    flatten_values(ess_bulk_ds[var].values)
    for var in ess_bulk_ds.data_vars
])

all_ess_tail = np.concatenate([
    flatten_values(ess_tail_ds[var].values)
    for var in ess_tail_ds.data_vars
])

print("\nDiagnostic summary:")
print(diag_df)

print("\nOverall:")
print(f"R-hat median: {np.nanmedian(all_rhat):.4f}")
print(f"R-hat max: {np.nanmax(all_rhat):.4f}")
print(f"n R-hat > 1.01: {np.sum(all_rhat > 1.01)}")
print(f"n R-hat > 1.05: {np.sum(all_rhat > 1.05)}")
print(f"Bulk ESS median: {np.nanmedian(all_ess_bulk):.1f}")
print(f"Bulk ESS min: {np.nanmin(all_ess_bulk):.1f}")
print(f"Tail ESS median: {np.nanmedian(all_ess_tail):.1f}")
print(f"Tail ESS min: {np.nanmin(all_ess_tail):.1f}")

print("\nSampler diagnostics:")

if hasattr(idata, "sample_stats"):
    ss = idata.sample_stats

    if "diverging" in ss:
        n_div = int(np.asarray(ss["diverging"]).sum())
        print(f"Divergences: {n_div}")
    else:
        print("Divergences: not available")

    if "tree_depth" in ss:
        tree_depth = np.asarray(ss["tree_depth"]).ravel()
        print(f"Max tree depth: {np.nanmax(tree_depth):.0f}")
    else:
        print("Tree depth: not available")

    print("BFMI: not reported")
else:
    print("sample_stats group not available")


# ============================================================
# Sampler diagnostics, if available
# ============================================================

sampler_text_lines = []

if hasattr(idata, "sample_stats"):
    ss = idata.sample_stats

    if "diverging" in ss:
        n_div = int(np.asarray(ss["diverging"]).sum())
        sampler_text_lines.append(f"Divergences: {n_div}")

    if "tree_depth" in ss:
        tree_depth = np.asarray(ss["tree_depth"]).ravel()
        sampler_text_lines.append(f"Max tree depth: {np.nanmax(tree_depth):.0f}")

    if "energy" in ss:
        try:
            bfmi_vals = az.bfmi(idata)
            bfmi_arr = np.asarray(bfmi_vals).ravel()
            bfmi_str = ", ".join([f"{x:.2f}" for x in bfmi_arr])
            sampler_text_lines.append(f"BFMI: {bfmi_str}")
        except Exception:
            pass

if not sampler_text_lines:
    sampler_text_lines.append("Sampler diagnostics unavailable")


# ============================================================
# Plot
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 13,
})

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8.8),
    constrained_layout=False,
)

axA, axB = axes[0, 0], axes[0, 1]
axC, axD = axes[1, 0], axes[1, 1]


# ------------------------------------------------------------
# Panel A: R-hat distribution
# ------------------------------------------------------------

axA.hist(
    all_rhat,
    bins=30,
    edgecolor="black",
    linewidth=0.6,
)

axA.axvline(1.01, linestyle="--", linewidth=1.2, label=r"$\hat{R}=1.01$")
axA.axvline(1.05, linestyle=":", linewidth=1.2, label=r"$\hat{R}=1.05$")

axA.set_title(r"Rank-normalized $\hat{R}$")
axA.set_xlabel(r"$\hat{R}$")
axA.set_ylabel("Number of monitored parameters")
axA.legend(frameon=False)
remove_spines(axA)
add_panel_label(axA, "A")


# ------------------------------------------------------------
# Panel B: Bulk ESS distribution
# ------------------------------------------------------------

axB.hist(
    all_ess_bulk,
    bins=30,
    edgecolor="black",
    linewidth=0.6,
)

axB.axvline(400, linestyle="--", linewidth=1.2, label="ESS = 400")

axB.set_title("Bulk effective sample size")
axB.set_xlabel("Bulk ESS")
axB.set_ylabel("Number of monitored parameters")
axB.legend(frameon=False)
remove_spines(axB)
add_panel_label(axB, "B")


# ------------------------------------------------------------
# Panel C: Tail ESS distribution
# ------------------------------------------------------------

axC.hist(
    all_ess_tail,
    bins=30,
    edgecolor="black",
    linewidth=0.6,
)

axC.axvline(400, linestyle="--", linewidth=1.2, label="ESS = 400")

axC.set_title("Tail effective sample size")
axC.set_xlabel("Tail ESS")
axC.set_ylabel("Number of monitored parameters")
axC.legend(frameon=False)
remove_spines(axC)
add_panel_label(axC, "C")


# ------------------------------------------------------------
# Panel D: Representative trace plots
# ------------------------------------------------------------

trace_vars = choose_trace_variables(idata, max_vars=4)
print("\nTrace variables selected for Panel D:")
print(trace_vars)

offset = 0.0
yticks = []
yticklabels = []

for i, var in enumerate(trace_vars):
    da = summarize_variable_scalar(idata.posterior[var])
    arr = da.values  # shape chain x draw

    # Standardize each variable for visual overlay
    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]

    if len(flat) == 0:
        continue

    mean_val = np.nanmean(flat)
    sd_val = np.nanstd(flat)
    if sd_val == 0 or not np.isfinite(sd_val):
        sd_val = 1.0

    arr_z = (arr - mean_val) / sd_val

    # Offset each variable vertically
    y_offset = i * 4.0
    yticks.append(y_offset)
    yticklabels.append(var)

    for chain in range(arr_z.shape[0]):
        axD.plot(
            np.arange(arr_z.shape[1]),
            arr_z[chain, :] + y_offset,
            linewidth=0.5,
            alpha=0.75,
        )

axD.set_title("Representative standardized trace plots")
axD.set_xlabel("Posterior draw")
axD.set_ylabel("Variable")
axD.set_yticks(yticks)
axD.set_yticklabels(yticklabels)
remove_spines(axD)
add_panel_label(axD, "D")

# Add sampler diagnostics text box
sampler_text = "\n".join(sampler_text_lines)
axD.text(
    1.02,
    0.02,
    sampler_text,
    transform=axD.transAxes,
    ha="left",
    va="bottom",
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgray"),
)


# ============================================================
# Final formatting and save
# ============================================================

fig.suptitle(
    "Supplementary Figure S3. MCMC convergence diagnostics for the ecological-context-modulated OU model",
    y=0.98,
    fontweight="bold",
)

fig.tight_layout(rect=[0, 0, 1, 0.95])

png_path = OUT_DIR / f"{OUT_PREFIX}.png"
pdf_path = OUT_DIR / f"{OUT_PREFIX}.pdf"
svg_path = OUT_DIR / f"{OUT_PREFIX}.svg"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")

plt.close(fig)

print("\nSaved files:")
print(png_path)
print(pdf_path)
print(svg_path)

print("\nSaved diagnostic table:")
print(diag_path)
