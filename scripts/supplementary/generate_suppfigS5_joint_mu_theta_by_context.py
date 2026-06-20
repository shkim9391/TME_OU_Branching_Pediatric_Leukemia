from pathlib import Path
import re
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

TRACE_FILE = BASE_DIR / "results" / "ou_revised_ecological_context_idata_ppc.nc"
CONTEXT_FILE = BASE_DIR / "Figure_3" / "patient_ecological_context_assignments.csv"

OUT_DIR = BASE_DIR / "BMC_Bioinformatics" / "Supplementary_Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = "SuppFigS5_joint_mu_theta_by_context"

CONTEXT_ORDER = ["E1", "E2", "E3", "E4"]

CONTEXT_COLORS = {
    "E1": "#4C78A8",
    "E2": "#59A14F",
    "E3": "#F28E2B",
    "E4": "#B279A2",
}

RANDOM_SEED = 123


# ============================================================
# Helper functions
# ============================================================

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def clean_context(x):
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip()

    numeric_to_context = {
        "0": "E1",
        "1": "E2",
        "2": "E3",
        "3": "E4",
    }

    if s in numeric_to_context:
        return numeric_to_context[s]

    if s.upper() in ["E1", "E2", "E3", "E4"]:
        return s.upper()

    return s


def natural_patient_key(pid):
    s = str(pid)
    m = re.search(r"(\d+)$", s)
    if m:
        return int(m.group(1))
    return s


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


def get_patient_dimension(da):
    non_sample_dims = [d for d in da.dims if d not in ["chain", "draw"]]

    if len(non_sample_dims) != 1:
        raise ValueError(
            f"Expected exactly one patient dimension, but got dims: {da.dims}"
        )

    return non_sample_dims[0]


def extract_patient_ids_from_idata(idata, da, patient_dim, context_df, patient_id_col):
    n_pat = da.sizes[patient_dim]

    if patient_dim in da.coords:
        coord_vals = da.coords[patient_dim].values
        coord_vals_str = [str(x) for x in coord_vals]
        if not all(s.isdigit() for s in coord_vals_str):
            print(f"Using patient IDs from posterior coordinate: {patient_dim}")
            return coord_vals_str

    for coord_name in ["Patient_ID", "patient_id", "participant_id"]:
        if coord_name in idata.posterior.coords:
            coord_vals = idata.posterior.coords[coord_name].values
            if len(coord_vals) == n_pat:
                print(f"Using patient IDs from posterior coordinate: {coord_name}")
                return [str(x) for x in coord_vals]

    context_ordered = context_df.copy()
    context_ordered[patient_id_col] = context_ordered[patient_id_col].astype(str)
    context_ordered = context_ordered.drop_duplicates(subset=[patient_id_col])

    if len(context_ordered) != n_pat:
        raise ValueError(
            f"Posterior has {n_pat} patients, but context table has "
            f"{len(context_ordered)} unique patients. Cannot safely align."
        )

    print(
        "WARNING: No patient IDs found in posterior coordinates. "
        "Using the row order of the context assignment table."
    )

    return context_ordered[patient_id_col].astype(str).tolist()


def summarize_patient_draws(draws_2d):
    return {
        "mean": np.nanmean(draws_2d, axis=0),
        "median": np.nanmedian(draws_2d, axis=0),
        "q025": np.nanquantile(draws_2d, 0.025, axis=0),
        "q975": np.nanquantile(draws_2d, 0.975, axis=0),
    }


def safe_std(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return 0.0
    return float(np.nanstd(vals, ddof=1))


# ============================================================
# Load data
# ============================================================

trace_file = TRACE_FILE
context_file = CONTEXT_FILE

for f in [trace_file, context_file]:
    if not f.exists():
        raise FileNotFoundError(f"Required input file not found: {f}")

print(f"\nLoading inference data: {trace_file}")
print(f"Using context file: {context_file}")

idata = az.from_netcdf(trace_file)
context_df = pd.read_csv(context_file)

print("\nPosterior variables:")
print(list(idata.posterior.data_vars))

print("\nContext table columns:")
print(context_df.columns.tolist())


# ============================================================
# Identify columns and variables
# ============================================================

patient_id_col = pick_col(
    context_df,
    ["Patient_ID", "patient_id", "participant_id", "Participant_ID", "sample_id"],
)

context_col = pick_col(
    context_df,
    [
        "ecological_context",
        "candidate_ecological_context",
        "context_label",
        "context",
        "immune_ecotype",
        "ecotype",
        "ecotype_label",
    ],
)

if "mu_pat" not in idata.posterior:
    raise KeyError("Expected posterior variable 'mu_pat' was not found.")

if "log_theta_pat" in idata.posterior:
    log_theta_var = "log_theta_pat"
elif "theta_pat" in idata.posterior:
    log_theta_var = "theta_pat"
else:
    raise KeyError("Expected posterior variable 'log_theta_pat' or 'theta_pat' was not found.")

mu_da = idata.posterior["mu_pat"]
theta_da = idata.posterior[log_theta_var]

patient_dim_mu = get_patient_dimension(mu_da)
patient_dim_theta = get_patient_dimension(theta_da)

n_pat_mu = mu_da.sizes[patient_dim_mu]
n_pat_theta = theta_da.sizes[patient_dim_theta]

if n_pat_mu != n_pat_theta:
    raise ValueError(
        f"Patient dimension sizes differ between mu and theta variables: "
        f"{n_pat_mu} vs {n_pat_theta}"
    )

n_pat = n_pat_mu

print(f"\nPatient dimension for mu_pat: {patient_dim_mu}")
print(f"Patient dimension for {log_theta_var}: {patient_dim_theta}")
print(f"Number of patients in posterior: {n_pat}")

context_df = context_df.copy()
context_df[patient_id_col] = context_df[patient_id_col].astype(str)
context_df["ecological_context"] = context_df[context_col].apply(clean_context)
context_df = context_df.drop_duplicates(subset=[patient_id_col]).copy()

patient_ids = extract_patient_ids_from_idata(
    idata=idata,
    da=mu_da,
    patient_dim=patient_dim_mu,
    context_df=context_df,
    patient_id_col=patient_id_col,
)

patient_map_df = pd.DataFrame({
    "Patient_ID": patient_ids,
    "posterior_patient_index": np.arange(len(patient_ids)),
})

patient_map_df = patient_map_df.merge(
    context_df[[patient_id_col, "ecological_context"]],
    left_on="Patient_ID",
    right_on=patient_id_col,
    how="left",
)

patient_map_df["ecological_context"] = patient_map_df["ecological_context"].fillna("Unknown")

if patient_map_df["ecological_context"].eq("Unknown").any():
    missing = patient_map_df.loc[
        patient_map_df["ecological_context"].eq("Unknown"),
        "Patient_ID",
    ].tolist()
    raise ValueError(
        "Some posterior patients could not be matched to context labels:\n"
        + ", ".join(missing[:20])
    )

patient_map_df = patient_map_df[
    patient_map_df["ecological_context"].isin(CONTEXT_ORDER)
].copy()

print("\nContext counts used for S5:")
print(patient_map_df["ecological_context"].value_counts().reindex(CONTEXT_ORDER, fill_value=0))


# ============================================================
# Extract draws and summaries
# ============================================================

mu_stacked = mu_da.stack(sample=("chain", "draw")).transpose("sample", patient_dim_mu)
theta_stacked = theta_da.stack(sample=("chain", "draw")).transpose("sample", patient_dim_theta)

mu_draws = np.asarray(mu_stacked.values, dtype=float)
theta_draws_raw = np.asarray(theta_stacked.values, dtype=float)

if log_theta_var == "theta_pat":
    theta_draws = np.log(theta_draws_raw)
else:
    theta_draws = theta_draws_raw

idx = patient_map_df["posterior_patient_index"].to_numpy(dtype=int)

mu_draws = mu_draws[:, idx]
theta_draws = theta_draws[:, idx]

mu_summary = summarize_patient_draws(mu_draws)
theta_summary = summarize_patient_draws(theta_draws)

patient_summary_df = patient_map_df[["Patient_ID", "posterior_patient_index", "ecological_context"]].copy()

for key, vals in mu_summary.items():
    patient_summary_df[f"mu_{key}"] = vals

for key, vals in theta_summary.items():
    patient_summary_df[f"log_theta_{key}"] = vals

patient_summary_path = OUT_DIR / "SuppFigS5_patient_joint_parameter_summary.csv"
patient_summary_df.to_csv(patient_summary_path, index=False)

context_rows = []

for context in CONTEXT_ORDER:
    sub = patient_summary_df[patient_summary_df["ecological_context"] == context]

    context_rows.append({
        "ecological_context": context,
        "n_patients": int(len(sub)),
        "mu_mean_centroid": float(sub["mu_mean"].mean()) if len(sub) else np.nan,
        "log_theta_mean_centroid": float(sub["log_theta_mean"].mean()) if len(sub) else np.nan,
        "mu_mean_sd": safe_std(sub["mu_mean"].values) if len(sub) else np.nan,
        "log_theta_mean_sd": safe_std(sub["log_theta_mean"].values) if len(sub) else np.nan,
        "mu_mean_median": float(sub["mu_mean"].median()) if len(sub) else np.nan,
        "log_theta_mean_median": float(sub["log_theta_mean"].median()) if len(sub) else np.nan,
    })

context_summary_df = pd.DataFrame(context_rows)
context_summary_path = OUT_DIR / "SuppFigS5_context_joint_parameter_summary.csv"
context_summary_df.to_csv(context_summary_path, index=False)

print("\nContext-level joint parameter summary:")
print(context_summary_df)


# ============================================================
# Plot: two-panel version
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
    1,
    2,
    figsize=(12, 5.2),
    constrained_layout=False,
)

axA, axB = axes[0], axes[1]


# ------------------------------------------------------------
# Shared axis limits
# ------------------------------------------------------------

x_vals = patient_summary_df["mu_mean"].to_numpy(dtype=float)
y_vals = patient_summary_df["log_theta_mean"].to_numpy(dtype=float)

x_pad = 0.15 * (np.nanmax(x_vals) - np.nanmin(x_vals) + 1e-6)
y_pad = 0.15 * (np.nanmax(y_vals) - np.nanmin(y_vals) + 1e-6)

xlim = (np.nanmin(x_vals) - x_pad, np.nanmax(x_vals) + x_pad)
ylim = (np.nanmin(y_vals) - y_pad, np.nanmax(y_vals) + y_pad)


# ------------------------------------------------------------
# Panel A: Patient-level posterior means
# ------------------------------------------------------------

for context in CONTEXT_ORDER:
    sub = patient_summary_df[patient_summary_df["ecological_context"] == context]
    if sub.empty:
        continue

    axA.scatter(
        sub["mu_mean"],
        sub["log_theta_mean"],
        s=48,
        color=CONTEXT_COLORS.get(context, "gray"),
        edgecolor="black",
        linewidth=0.45,
        alpha=0.88,
        label=f"{context} (n={len(sub)})",
    )

axA.set_title(r"Patient-level posterior means")
axA.set_xlabel(r"Posterior mean $\mu_i$")
axA.set_ylabel(r"Posterior mean $\log(\theta_i)$")
axA.set_xlim(xlim)
axA.set_ylim(ylim)
axA.legend(frameon=False, loc="best")
remove_spines(axA)
add_panel_label(axA, "A")


# ------------------------------------------------------------
# Panel B: Context centroids and dispersion bars
# ------------------------------------------------------------

for _, row in context_summary_df.iterrows():
    context = row["ecological_context"]
    n = row["n_patients"]

    if n == 0:
        continue

    x = row["mu_mean_centroid"]
    y = row["log_theta_mean_centroid"]
    xerr = row["mu_mean_sd"]
    yerr = row["log_theta_mean_sd"]

    axB.errorbar(
        x,
        y,
        xerr=xerr if np.isfinite(xerr) else 0.0,
        yerr=yerr if np.isfinite(yerr) else 0.0,
        fmt="o",
        markersize=9,
        color=CONTEXT_COLORS.get(context, "gray"),
        ecolor=CONTEXT_COLORS.get(context, "gray"),
        markeredgecolor="black",
        markeredgewidth=0.8,
        elinewidth=1.5,
        capsize=4,
        label=f"{context} (n={n})",
        zorder=4,
    )

    axB.text(
        x,
        y,
        f" {context}",
        fontsize=9,
        va="center",
        ha="left",
        fontweight="bold",
    )

axB.set_title("Context centroids and between-patient dispersion")
axB.set_xlabel(r"Mean of posterior $\mu_i$")
axB.set_ylabel(r"Mean of posterior $\log(\theta_i)$")
axB.set_xlim(xlim)
axB.set_ylim(ylim)
axB.legend(frameon=False, loc="best")
remove_spines(axB)
add_panel_label(axB, "B")


# ============================================================
# Final formatting and save
# ============================================================

fig.suptitle(
    "Supplementary Figure S5. Joint posterior structure of patient-level OU parameters by ecological context",
    y=0.99,
    fontweight="bold",
)

fig.tight_layout(rect=[0, 0, 1, 0.92])

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

print("\nSaved tables:")
print(patient_summary_path)
print(context_summary_path)
