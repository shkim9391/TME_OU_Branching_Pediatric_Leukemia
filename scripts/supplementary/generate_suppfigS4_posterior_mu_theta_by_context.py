from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az


# ============================================================
# User settings
# ============================================================

BASE_DIR = Path("/TME_OU_Branching")

TRACE_FILE = BASE_DIR / "results" / "ou_revised_ecological_context_idata_ppc.nc"
CONTEXT_FILE = BASE_DIR / "Figure_3" / "patient_ecological_context_assignments.csv"

OUT_DIR = BASE_DIR / "BMC_Bioinformatics" / "Supplementary_Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = "SuppFigS4_posterior_mu_theta_by_context"

TRACE_FILES = [
    "tme_ou_ecotype_idata_E88_FINAL.nc",
    "tme_ou_ecotype_idata.nc",
    "ou_ecotype_ou_branching_trace.nc",
    "tme_ou_ecotype_trace.nc",
    "idata.nc",
    "trace.nc",
]

CONTEXT_FILES = [
    "patient_ecological_context_assignments.csv",
    "ecological_context_master_table.csv",
    "patient_context_assignments.csv",
    "patient_immune_ecotypes.csv",
]

CONTEXT_ORDER = ["E1", "E2", "E3", "E4"]

CONTEXT_COLORS = {
    "E1": "#4C78A8",
    "E2": "#59A14F",
    "E3": "#F28E2B",
    "E4": "#B279A2",
}

RANDOM_SEED = 123
MAX_POSTERIOR_DRAWS_PER_CONTEXT = 25000


# ============================================================
# Helper functions
# ============================================================

def pick_col(df, candidates, required=True):
    """Pick first available column from candidate names."""
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
    """Standardize ecological-context labels."""
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
    """
    Natural sort key for patient IDs such as P1, P2, P10, P100.
    """
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
    """
    Identify patient dimension from posterior variable.
    Expected dims are usually ('chain', 'draw', 'patient') or similar.
    """
    non_sample_dims = [d for d in da.dims if d not in ["chain", "draw"]]

    if len(non_sample_dims) != 1:
        raise ValueError(
            f"Expected exactly one patient dimension for variable, but got: {da.dims}"
        )

    return non_sample_dims[0]


def extract_patient_ids_from_idata(idata, da, patient_dim, context_df, patient_id_col):
    """
    Try to extract Patient_ID labels from inference data coordinates.
    If unavailable, fall back to context table order.
    """
    n_pat = da.sizes[patient_dim]

    # Case 1: patient dimension coordinate exists and looks like patient IDs.
    if patient_dim in da.coords:
        coord_vals = da.coords[patient_dim].values
        coord_vals_str = [str(x) for x in coord_vals]

        # If these look like P1/P2 or other non-integer IDs, use them.
        if not all(s.isdigit() for s in coord_vals_str):
            print(f"Using patient IDs from posterior coordinate: {patient_dim}")
            return coord_vals_str

    # Case 2: try common coordinate names.
    for coord_name in ["Patient_ID", "patient_id", "participant_id"]:
        if coord_name in idata.posterior.coords:
            coord_vals = idata.posterior.coords[coord_name].values
            if len(coord_vals) == n_pat:
                print(f"Using patient IDs from posterior coordinate: {coord_name}")
                return [str(x) for x in coord_vals]

    # Case 3: fallback to context-table order.
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


def summarize_patient_parameter(draws_2d):
    """
    draws_2d shape: n_draws_total x n_patients.
    Returns patient-level summary dataframe columns.
    """
    return {
        "mean": np.nanmean(draws_2d, axis=0),
        "median": np.nanmedian(draws_2d, axis=0),
        "q025": np.nanquantile(draws_2d, 0.025, axis=0),
        "q25": np.nanquantile(draws_2d, 0.25, axis=0),
        "q75": np.nanquantile(draws_2d, 0.75, axis=0),
        "q975": np.nanquantile(draws_2d, 0.975, axis=0),
    }


def prepare_context_draws(draws_2d, contexts, context_order, max_draws=25000, seed=123):
    """
    Concatenate posterior draws across patients within each context.
    draws_2d shape: n_draws_total x n_patients.
    contexts length: n_patients.
    """
    rng = np.random.default_rng(seed)

    out = []

    contexts = np.asarray(contexts)

    for context in context_order:
        idx = np.where(contexts == context)[0]

        if len(idx) == 0:
            out.append(np.array([]))
            continue

        vals = draws_2d[:, idx].ravel()
        vals = vals[np.isfinite(vals)]

        if len(vals) > max_draws:
            vals = rng.choice(vals, size=max_draws, replace=False)

        out.append(vals)

    return out


def plot_violin_with_patient_means(ax, context_draws, patient_df, value_col, ylabel, title):
    """Violin plots of pooled posterior draws plus patient-level posterior means."""
    positions = np.arange(1, len(CONTEXT_ORDER) + 1)

    nonempty_data = []
    nonempty_positions = []
    nonempty_contexts = []

    for pos, context, vals in zip(positions, CONTEXT_ORDER, context_draws):
        if len(vals) > 0:
            nonempty_data.append(vals)
            nonempty_positions.append(pos)
            nonempty_contexts.append(context)

    violins = ax.violinplot(
        nonempty_data,
        positions=nonempty_positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, context in zip(violins["bodies"], nonempty_contexts):
        body.set_facecolor(CONTEXT_COLORS.get(context, "gray"))
        body.set_edgecolor("black")
        body.set_alpha(0.65)
        body.set_linewidth(0.7)

    # Add median and IQR bars based on pooled posterior draws
    for pos, context, vals in zip(positions, CONTEXT_ORDER, context_draws):
        if len(vals) == 0:
            continue

        q25, med, q75 = np.nanquantile(vals, [0.25, 0.5, 0.75])

        ax.plot(
            [pos - 0.22, pos + 0.22],
            [med, med],
            color="black",
            linewidth=1.4,
            zorder=4,
        )

        ax.plot(
            [pos, pos],
            [q25, q75],
            color="black",
            linewidth=2.0,
            zorder=4,
        )

    # Overlay patient-level posterior means
    rng = np.random.default_rng(RANDOM_SEED)

    for pos, context in zip(positions, CONTEXT_ORDER):
        sub = patient_df[patient_df["ecological_context"] == context].copy()

        if sub.empty:
            continue

        jitter = rng.normal(0, 0.045, size=len(sub))

        ax.scatter(
            np.full(len(sub), pos) + jitter,
            sub[value_col],
            s=32,
            color=CONTEXT_COLORS.get(context, "gray"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.90,
            zorder=5,
        )


    ax.set_xticks(positions)
    ax.set_xticklabels(CONTEXT_ORDER)
    ax.set_xlabel("Candidate ecological context")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    remove_spines(ax)


# ============================================================
# Load input data
# ============================================================

trace_file = TRACE_FILE
context_file = CONTEXT_FILE

for f in [trace_file, context_file]:
    if not f.exists():
        raise FileNotFoundError(f"Required input file not found: {f}")

print(f"\nLoading inference data: {trace_file}")
idata = az.from_netcdf(trace_file)

context_df = pd.read_csv(context_file)

print("\nPosterior variables:")
print(list(idata.posterior.data_vars))

print("\nContext table columns:")
print(context_df.columns.tolist())


# ============================================================
# Identify columns and posterior variables
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

patient_dim = patient_dim_mu
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
    patient_dim=patient_dim,
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

print("\nContext counts used for S4:")
print(patient_map_df["ecological_context"].value_counts().reindex(CONTEXT_ORDER, fill_value=0))


# ============================================================
# Extract posterior draws
# ============================================================

# Ensure patient dimension is last after stacking chain/draw.
mu_stacked = mu_da.stack(sample=("chain", "draw")).transpose("sample", patient_dim_mu)
theta_stacked = theta_da.stack(sample=("chain", "draw")).transpose("sample", patient_dim_theta)

mu_draws = np.asarray(mu_stacked.values, dtype=float)
theta_draws_raw = np.asarray(theta_stacked.values, dtype=float)

if log_theta_var == "theta_pat":
    # If only theta is available, plot log(theta).
    theta_draws = np.log(theta_draws_raw)
    theta_label = r"$\log(\theta_i)$"
else:
    theta_draws = theta_draws_raw
    theta_label = r"$\log(\theta_i)$"

# Reorder/subset by patient_map_df posterior indices
idx = patient_map_df["posterior_patient_index"].to_numpy(dtype=int)
mu_draws = mu_draws[:, idx]
theta_draws = theta_draws[:, idx]
contexts = patient_map_df["ecological_context"].to_numpy()


# ============================================================
# Patient-level and context-level summaries
# ============================================================

mu_summary = summarize_patient_parameter(mu_draws)
theta_summary = summarize_patient_parameter(theta_draws)

patient_summary_df = patient_map_df[["Patient_ID", "posterior_patient_index", "ecological_context"]].copy()

for key, vals in mu_summary.items():
    patient_summary_df[f"mu_{key}"] = vals

for key, vals in theta_summary.items():
    patient_summary_df[f"log_theta_{key}"] = vals

patient_summary_path = OUT_DIR / "SuppFigS4_patient_parameter_summary.csv"
patient_summary_df.to_csv(patient_summary_path, index=False)

context_rows = []

for context in CONTEXT_ORDER:
    sub = patient_summary_df[patient_summary_df["ecological_context"] == context]

    context_rows.append({
        "ecological_context": context,
        "n_patients": int(len(sub)),
        "mu_patient_mean_mean": float(sub["mu_mean"].mean()) if len(sub) else np.nan,
        "mu_patient_mean_median": float(sub["mu_mean"].median()) if len(sub) else np.nan,
        "log_theta_patient_mean_mean": float(sub["log_theta_mean"].mean()) if len(sub) else np.nan,
        "log_theta_patient_mean_median": float(sub["log_theta_mean"].median()) if len(sub) else np.nan,
    })

context_summary_df = pd.DataFrame(context_rows)
context_summary_path = OUT_DIR / "SuppFigS4_context_parameter_summary.csv"
context_summary_df.to_csv(context_summary_path, index=False)

print("\nContext-level parameter summary:")
print(context_summary_df)


# ============================================================
# Prepare violin data
# ============================================================

mu_context_draws = prepare_context_draws(
    mu_draws,
    contexts,
    CONTEXT_ORDER,
    max_draws=MAX_POSTERIOR_DRAWS_PER_CONTEXT,
    seed=RANDOM_SEED,
)

theta_context_draws = prepare_context_draws(
    theta_draws,
    contexts,
    CONTEXT_ORDER,
    max_draws=MAX_POSTERIOR_DRAWS_PER_CONTEXT,
    seed=RANDOM_SEED + 1,
)


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
    1,
    2,
    figsize=(12, 5.2),
    constrained_layout=False,
)

axA, axB = axes[0], axes[1]

plot_violin_with_patient_means(
    ax=axA,
    context_draws=mu_context_draws,
    patient_df=patient_summary_df,
    value_col="mu_mean",
    ylabel=r"Posterior $\mu_i$",
    title=r"OU latent optimum $\mu_i$ by context",
)

add_panel_label(axA, "A")

plot_violin_with_patient_means(
    ax=axB,
    context_draws=theta_context_draws,
    patient_df=patient_summary_df,
    value_col="log_theta_mean",
    ylabel=theta_label,
    title=r"OU mean-reversion strength by context",
)

add_panel_label(axB, "B")

fig.suptitle(
    "Supplementary Figure S4. Posterior distributions of ecological-context-modulated OU parameters",
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
