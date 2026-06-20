import os
import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

try:
    import arviz as az
except ImportError as e:
    raise ImportError(
        "This script requires arviz. Install with: pip install arviz"
    ) from e


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = "/Desktop/TME_OU_Branching"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG3_DIR = os.path.join(BASE_DIR, "Figure_3")
OUTDIR = os.path.join(BASE_DIR, "Figure_4")
os.makedirs(OUTDIR, exist_ok=True)

SUPPORT_CSV = os.path.join(
    RESULTS_DIR,
    "longitudinal_support_revised_context.csv"
)

TRACE_NC = os.path.join(
    RESULTS_DIR,
    "ou_revised_ecological_context_idata_ppc.nc"
)

TRACE_PATIENT_ORDER_CSV = os.path.join(
    RESULTS_DIR,
    "model_patient_order_revised_context.csv"
)

ASSIGNMENTS_CSV = os.path.join(
    FIG3_DIR,
    "patient_ecological_context_assignments.csv"
)

MASTER_CONTEXT_CSV = os.path.join(
    FIG3_DIR,
    "ecological_context_master_table.csv"
)

COLOR_KEY_JSON = os.path.join(
    FIG3_DIR,
    "ecological_context_color_key.json"
)

OUT_PNG = os.path.join(OUTDIR, "Figure4_OU_parameter_summaries_by_context.png")
OUT_TIFF = os.path.join(OUTDIR, "Figure4_OU_parameter_summaries_by_context.tiff")
OUT_SUMMARY = os.path.join(OUTDIR, "Figure4_patient_OU_parameter_summaries.csv")


# =============================================================================
# Manual variable-name overrides
# =============================================================================
# Leave as None for automatic detection.
# If automatic detection fails, inspect:
#   print(idata.posterior.data_vars)
# and set these names manually.
MU_VAR = None
THETA_VAR = None

MU_CANDIDATES = [
    "mu_i",
    "mu_patient",
    "mu_participant",
    "patient_mu",
    "participant_mu",
    "mu_individual",
    "mu",
]

THETA_CANDIDATES = [
    "theta_i",
    "theta_patient",
    "theta_participant",
    "patient_theta",
    "participant_theta",
    "theta_individual",
    "theta",
    "log_theta_i",
    "log_theta_patient",
    "log_theta",
    "log10_theta_i",
    "log10_theta_patient",
    "log10_theta",
]


# =============================================================================
# Plot settings
# =============================================================================

RANDOM_SEED = 123
JITTER_WIDTH = 0.13
POINT_SIZE = 28
INTERVAL_ALPHA = 0.22
POINT_ALPHA = 0.88

FIGSIZE = (14.5, 5.2)
DPI = 600

CONTEXT_ORDER = ["E1", "E2", "E3", "E4"]

# Fallback colors if JSON is absent.
FALLBACK_CONTEXT_COLORS = {
    "E1": "#4C78A8",  # blue
    "E2": "#59A14F",  # green
    "E3": "#F28E2B",  # orange
    "E4": "#B279A2",  # purple
}

DIAGNOSIS_MARKERS = {
    "B-ALL": "o",
    "T-ALL": "s",
    "ETP-ALL": "^",
    "AML": "D",
    "MPAL": "X",
    "Unknown": "v",
}

DIAGNOSIS_SHORT_MAP = {
    "Acute myeloid leukemia": "AML",
    "B-cell acute lymphoblastic leukemia": "B-ALL",
    "Early T-cell precursor T-cell acute lymphoblastic leukemia": "ETP-ALL",
    "Mixed phenotype acute leukemia": "MPAL",
    "T-cell acute lymphoblastic leukemia": "T-ALL",
}


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# =============================================================================
# Helper functions
# =============================================================================

def normalize_patient_id(x) -> str:
    """Normalize patient IDs for robust merging."""
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Avoid converting P17-like strings. Only strip .0 from numeric IDs.
    if s.endswith(".0") and s[:-2].replace(".", "", 1).isdigit():
        s = s[:-2]
    return s


def short_diagnosis(x) -> str:
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip()
    return DIAGNOSIS_SHORT_MAP.get(x, x)


def find_patient_id_column(df: pd.DataFrame) -> str:
    candidates = [
        "Patient_ID",
        "participant_id",
        "Participant_ID",
        "patient_id",
        "patient",
        "participant",
        "Patient",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a patient ID column. Available columns: {list(df.columns)}"
    )


def read_context_colors() -> Dict[str, str]:
    if os.path.exists(COLOR_KEY_JSON):
        with open(COLOR_KEY_JSON, "r") as f:
            colors = json.load(f)
        return {str(k): str(v) for k, v in colors.items()}
    return FALLBACK_CONTEXT_COLORS.copy()


def load_context_assignments() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """Load revised Figure 3 context assignment and master table."""
    if not os.path.exists(ASSIGNMENTS_CSV):
        raise FileNotFoundError(
            f"Missing assignments file from revised Figure 3:\n{ASSIGNMENTS_CSV}"
        )
    if not os.path.exists(MASTER_CONTEXT_CSV):
        raise FileNotFoundError(
            f"Missing master context file from revised Figure 3:\n{MASTER_CONTEXT_CSV}"
        )

    assign = pd.read_csv(ASSIGNMENTS_CSV)
    master = pd.read_csv(MASTER_CONTEXT_CSV)

    pid_col = find_patient_id_column(assign)
    assign = assign.rename(columns={pid_col: "Patient_ID"})
    assign["Patient_ID"] = assign["Patient_ID"].map(normalize_patient_id)

    if "ecological_context" not in assign.columns:
        raise ValueError("Assignments CSV must contain column: ecological_context")

    assign["ecological_context"] = assign["ecological_context"].astype(str).str.strip()

    if "diagnosis" not in assign.columns:
        assign["diagnosis"] = "Unknown"
    assign["diagnosis"] = assign["diagnosis"].map(short_diagnosis)

    # Context labels from master table.
    context_label = {}
    for _, row in master.iterrows():
        eid = str(row["ecological_context"]).strip()
        label = str(row.get("context_label", eid)).strip()
        n = row.get("n_participants", np.nan)
        if pd.notna(n):
            context_label[eid] = f"{eid}: {label} (n={int(n)})"
        else:
            context_label[eid] = f"{eid}: {label}"

    # Fallback labels if needed.
    for eid in CONTEXT_ORDER:
        if eid not in context_label:
            n = int((assign["ecological_context"] == eid).sum())
            context_label[eid] = f"{eid} (n={n})"

    colors = read_context_colors()
    for eid in CONTEXT_ORDER:
        colors.setdefault(eid, FALLBACK_CONTEXT_COLORS[eid])

    return assign, master, context_label, colors


def is_patient_like_da(da, expected_n_max: int) -> bool:
    """Return True if variable has exactly one non-sample dimension."""
    dims = list(da.dims)
    other_dims = [d for d in dims if d not in ["chain", "draw", "sample"]]
    if len(other_dims) != 1:
        return False
    n = da.sizes[other_dims[0]]
    return 1 < n <= expected_n_max


def find_parameter_var(idata, candidates: List[str], expected_n_max: int, manual_name: Optional[str], label: str) -> str:
    """Find a patient-level posterior variable in idata.posterior."""
    posterior = idata.posterior

    if manual_name is not None:
        if manual_name not in posterior.data_vars:
            raise ValueError(f"Manual {label} variable '{manual_name}' not found in trace.")
        if not is_patient_like_da(posterior[manual_name], expected_n_max):
            raise ValueError(f"Manual {label} variable '{manual_name}' does not look patient-level.")
        return manual_name

    # First try explicit candidates.
    for v in candidates:
        if v in posterior.data_vars and is_patient_like_da(posterior[v], expected_n_max):
            return v

    # Then scan by substring.
    lowercase_label = label.lower()
    possible = []
    for v in posterior.data_vars:
        vl = v.lower()
        if lowercase_label in vl and is_patient_like_da(posterior[v], expected_n_max):
            possible.append(v)

    if possible:
        print(f"Auto-detected possible {label} variables: {possible}")
        return possible[0]

    # Print helpful debug info.
    print("\nAvailable posterior variables:")
    for v in posterior.data_vars:
        print(f"  {v}: dims={posterior[v].dims}, shape={posterior[v].shape}")

    raise ValueError(
        f"Could not auto-detect a patient-level {label} variable. "
        f"Set {label.upper()}_VAR manually near the top of the script."
    )


def extract_patient_samples(idata, var_name: str) -> Tuple[np.ndarray, List[str], str]:
    """
    Extract posterior samples for a patient-level variable.

    Returns
    -------
    samples : ndarray, shape [n_samples, n_patients]
    coord_ids : list[str], patient coordinate values if available, else index strings
    patient_dim : str
    """
    da = idata.posterior[var_name]

    # Stack chain/draw into sample.
    if "chain" in da.dims and "draw" in da.dims:
        da = da.stack(sample=("chain", "draw"))
    elif "sample" not in da.dims:
        raise ValueError(f"Variable {var_name} has no chain/draw/sample dimensions.")

    other_dims = [d for d in da.dims if d != "sample"]
    if len(other_dims) != 1:
        raise ValueError(
            f"Expected exactly one patient dimension for {var_name}; got dims {da.dims}"
        )

    patient_dim = other_dims[0]
    da = da.transpose("sample", patient_dim)
    samples = da.values

    if patient_dim in da.coords:
        coord_vals = da[patient_dim].values
    else:
        coord_vals = np.arange(samples.shape[1])

    coord_ids = [normalize_patient_id(x) for x in coord_vals]

    return samples, coord_ids, patient_dim


def load_trace_patient_order(n_patients: int, trace_coord_ids: List[str], assignments: pd.DataFrame) -> List[str]:
    """
    Determine patient order for trace dimension.

    Priority:
      1. Non-numeric trace coordinate IDs matching assignments.
      2. TRACE_PATIENT_ORDER_CSV, if available and length matches.
      3. Assignments order, only if length matches exactly.
    """
    assignment_ids = set(assignments["Patient_ID"].map(normalize_patient_id))

    # If trace coords look like real patient IDs and overlap assignment IDs, use them.
    overlap = sum(pid in assignment_ids for pid in trace_coord_ids)
    if overlap >= max(1, int(0.5 * n_patients)):
        print(f"Using patient IDs from trace coordinate; overlap = {overlap}/{n_patients}")
        return trace_coord_ids

    # Use old patient_immune_ecotypes.csv only for patient order.
    if os.path.exists(TRACE_PATIENT_ORDER_CSV):
        order_df = pd.read_csv(TRACE_PATIENT_ORDER_CSV)
        try:
            pid_col = find_patient_id_column(order_df)
            order_ids = order_df[pid_col].map(normalize_patient_id).tolist()
            if len(order_ids) >= n_patients:
                order_ids = order_ids[:n_patients]
                overlap = sum(pid in assignment_ids for pid in order_ids)
                if overlap >= max(1, int(0.5 * n_patients)):
                    print(
                        f"Using patient order from {TRACE_PATIENT_ORDER_CSV}; "
                        f"overlap = {overlap}/{n_patients}"
                    )
                    return order_ids
        except Exception as e:
            print(f"Could not use TRACE_PATIENT_ORDER_CSV for ordering: {e}")

    raise ValueError(
        "Trace has no patient-ID coordinate and no verified model patient-order file was used. "
        "Do not assume assignments CSV order. Provide the exact Patient_ID order used during model fitting."
    )

    raise ValueError(
        "Could not determine patient order for trace dimension.\n"
        "Fix by ensuring the trace has patient ID coordinates or by providing "
        "TRACE_PATIENT_ORDER_CSV with Patient_ID in the same order as the trace."
    )


def theta_samples_to_log10(theta_samples: np.ndarray, theta_var_name: str) -> np.ndarray:
    """
    Convert theta posterior samples to log10(theta).

    If the variable name contains 'log10', assume it is already log10(theta).
    If it contains 'log' but not 'log10', assume natural log(theta) and divide by ln(10).
    Otherwise assume theta is positive and take log10(theta).
    """
    name = theta_var_name.lower()

    if "log10" in name:
        return theta_samples

    if "log" in name:
        return theta_samples / np.log(10.0)

    if np.nanmin(theta_samples) <= 0:
        print(
            f"Warning: {theta_var_name} contains non-positive values. "
            "Treating it as already log10(theta)."
        )
        return theta_samples

    return np.log10(theta_samples)


def summarize_parameter(samples: np.ndarray, prefix: str) -> pd.DataFrame:
    """Summarize posterior samples per patient."""
    return pd.DataFrame({
        f"{prefix}_mean": np.nanmean(samples, axis=0),
        f"{prefix}_q025": np.nanquantile(samples, 0.025, axis=0),
        f"{prefix}_q500": np.nanquantile(samples, 0.500, axis=0),
        f"{prefix}_q975": np.nanquantile(samples, 0.975, axis=0),
    })


def jittered_positions(x_base: np.ndarray, width: float, rng: np.random.Generator) -> np.ndarray:
    return x_base + rng.uniform(-width, width, size=len(x_base))


def add_panel_label(ax, label: str, x: float = -0.20, y: float = 1.20) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def plot_parameter_panel(
    ax,
    df: pd.DataFrame,
    y_mean: str,
    y_low: str,
    y_high: str,
    ylabel: str,
    title: str,
    context_colors: Dict[str, str],
    rng: np.random.Generator,
) -> None:
    """Plot participant posterior means and 95% intervals by context."""
    context_to_x = {eid: i for i, eid in enumerate(CONTEXT_ORDER)}

    for eid in CONTEXT_ORDER:
        sub = df[df["ecological_context"] == eid].copy()
        if sub.empty:
            continue

        x0 = context_to_x[eid]
        x = jittered_positions(np.full(len(sub), x0, dtype=float), JITTER_WIDTH, rng)

        # Individual posterior intervals.
        for xi, (_, row) in zip(x, sub.iterrows()):
            ax.plot(
                [xi, xi],
                [row[y_low], row[y_high]],
                color=context_colors[eid],
                alpha=INTERVAL_ALPHA,
                lw=1.2,
                zorder=1,
            )

        # Participant posterior means.
        for xi, (_, row) in zip(x, sub.iterrows()):
            diag = row.get("diagnosis", "Unknown")
            marker = DIAGNOSIS_MARKERS.get(diag, "o")
            ax.scatter(
                xi,
                row[y_mean],
                s=POINT_SIZE,
                marker=marker,
                color=context_colors[eid],
                edgecolor="black",
                linewidth=0.35,
                alpha=POINT_ALPHA,
                zorder=2,
            )

        # Context mean and SD of participant posterior means.
        mean_val = sub[y_mean].mean()
        sd_val = sub[y_mean].std(ddof=1) if len(sub) > 1 else 0.0
        ax.errorbar(
            x0,
            mean_val,
            yerr=sd_val,
            fmt="D",
            color=context_colors[eid],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
            capsize=4,
            lw=1.5,
            zorder=4,
        )

    xticklabels = []
    for eid in CONTEXT_ORDER:
        n = int((df["ecological_context"] == eid).sum())
        xticklabels.append(f"{eid}\n$n_{{long}}$={n}")

    ax.set_xticks(np.arange(len(CONTEXT_ORDER)))
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18)
    ax.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax.grid(axis="y", color="0.90", lw=0.8)
    ax.set_axisbelow(True)


def plot_joint_panel(
    ax,
    df: pd.DataFrame,
    context_colors: Dict[str, str],
) -> None:
    """Plot joint posterior mean μ_i vs log10 θ_i."""
    # Participant-level points.
    for eid in CONTEXT_ORDER:
        sub = df[df["ecological_context"] == eid].copy()
        if sub.empty:
            continue

        for _, row in sub.iterrows():
            diag = row.get("diagnosis", "Unknown")
            marker = DIAGNOSIS_MARKERS.get(diag, "o")
            ax.scatter(
                row["mu_mean"],
                row["log10_theta_mean"],
                s=38,
                marker=marker,
                color=context_colors[eid],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.88,
                zorder=2,
            )

    # Context centroids with color-matched error bars.
    for eid in CONTEXT_ORDER:
        sub = df[df["ecological_context"] == eid].copy()
        if sub.empty:
            continue

        x_mean = sub["mu_mean"].mean()
        y_mean = sub["log10_theta_mean"].mean()
        x_sd = sub["mu_mean"].std(ddof=1) if len(sub) > 1 else 0.0
        y_sd = sub["log10_theta_mean"].std(ddof=1) if len(sub) > 1 else 0.0

        ax.errorbar(
            x_mean,
            y_mean,
            xerr=x_sd,
            yerr=y_sd,
            fmt="D",
            color=context_colors[eid],
            ecolor=context_colors[eid],
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=10,
            capsize=5,
            lw=1.8,
            zorder=5,
        )

        ax.text(
            x_mean,
            y_mean,
            f" {eid}",
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="left",
            color="black",
            zorder=6,
        )

    ax.set_xlabel(r"Posterior mean local attractor $\mu_i$")
    ax.set_ylabel(r"Posterior mean $\log_{10}$ effective mean reversion $\theta_i$")
    ax.set_title("Likelihood-supported joint OU summaries", pad=18)
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.95, 1.45)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.grid(color="0.92", lw=0.8)
    ax.set_axisbelow(True)


def make_legends(fig, context_labels: Dict[str, str], context_colors: Dict[str, str], diagnoses: List[str]) -> None:
    """Add context and diagnosis legends to the full figure."""
    context_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            label=context_labels[eid],
            markerfacecolor=context_colors[eid],
            markeredgecolor="black",
            markersize=8,
        )
        for eid in CONTEXT_ORDER
    ]

    diag_handles = []
    for d in diagnoses:
        diag_handles.append(
            Line2D(
                [0], [0],
                marker=DIAGNOSIS_MARKERS.get(d, "o"),
                color="black",
                label=d,
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=7,
                linestyle="None",
            )
        )

    leg1 = fig.legend(
        handles=context_handles,
        title="Candidate ecological context (full cohort n)",
        loc="upper center",
        bbox_to_anchor=(0.50, 0.02),
        ncol=2,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    leg2 = fig.legend(
        handles=diag_handles,
        title="Diagnosis marker",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.98),
        frameon=False,
        handletextpad=0.6,
        labelspacing=0.4,
    )

    fig.add_artist(leg1)
    fig.add_artist(leg2)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # Load context assignments.
    assignments, master, context_labels, context_colors = load_context_assignments()
    expected_n_max = len(assignments)

    # Load trace.
    if not os.path.exists(TRACE_NC):
        raise FileNotFoundError(f"Missing trace file:\n{TRACE_NC}")

    print(f"Loading trace: {TRACE_NC}")
    idata = az.from_netcdf(TRACE_NC)

    # Find variables.
    mu_var = find_parameter_var(
        idata,
        candidates=MU_CANDIDATES,
        expected_n_max=expected_n_max,
        manual_name=MU_VAR,
        label="mu",
    )

    theta_var = find_parameter_var(
        idata,
        candidates=THETA_CANDIDATES,
        expected_n_max=expected_n_max,
        manual_name=THETA_VAR,
        label="theta",
    )

    print(f"Using μ variable:     {mu_var}")
    print(f"Using θ variable:     {theta_var}")

    # Extract samples.
    mu_samples, mu_coord_ids, mu_patient_dim = extract_patient_samples(idata, mu_var)
    theta_samples, theta_coord_ids, theta_patient_dim = extract_patient_samples(idata, theta_var)

    if mu_samples.shape[1] != theta_samples.shape[1]:
        raise ValueError(
            f"μ and θ have different patient counts: "
            f"{mu_samples.shape[1]} vs {theta_samples.shape[1]}"
        )

    n_patients = mu_samples.shape[1]

    # Convert theta to log10(theta).
    log10_theta_samples = theta_samples_to_log10(theta_samples, theta_var)

    # Determine patient order.
    patient_ids = load_trace_patient_order(
        n_patients=n_patients,
        trace_coord_ids=mu_coord_ids,
        assignments=assignments,
    )

    # Summaries.
    mu_summary = summarize_parameter(mu_samples, "mu")
    theta_summary = summarize_parameter(log10_theta_samples, "log10_theta")

    summ = pd.concat([mu_summary, theta_summary], axis=1)
    summ["Patient_ID"] = patient_ids
    summ["Patient_ID"] = summ["Patient_ID"].map(normalize_patient_id)

    # Merge with revised Figure 3 context assignments.
    keep_cols = ["Patient_ID", "diagnosis", "subdiagnosis", "ecological_context"]
    for c in keep_cols:
        if c not in assignments.columns:
            assignments[c] = "Unknown"

    plot_df = summ.merge(assignments[keep_cols], on="Patient_ID", how="left")
    
    # -------------------------------------------------------------------------
    # Restrict main Figure 4 to participants with direct longitudinal support
    # -------------------------------------------------------------------------
    
    if not os.path.exists(SUPPORT_CSV):
        raise FileNotFoundError(
            f"Missing longitudinal support file:\n{SUPPORT_CSV}\n"
            "Run check_longitudinal_support.py before generating Figure 4."
        )
    
    support = pd.read_csv(SUPPORT_CSV)
    support["Patient_ID"] = support["Patient_ID"].map(normalize_patient_id)
    
    support_cols = [
        "Patient_ID",
        "n_longitudinal_rows",
        "has_longitudinal_rows",
        "n_transitions",
        "has_transition",
    ]
    
    support_cols = [c for c in support_cols if c in support.columns]
    
    plot_df = plot_df.merge(
        support[support_cols],
        on="Patient_ID",
        how="left"
    )
    
    plot_df["n_transitions"] = plot_df["n_transitions"].fillna(0).astype(int)
    plot_df["has_transition"] = plot_df["n_transitions"] > 0
    
    # Keep a full copy for audit if needed
    plot_df_all_modeled = plot_df.copy()
    
    # Main Figure 4 should show likelihood-supported participants only
    plot_df = plot_df[plot_df["has_transition"]].copy()
    
    print("\nLongitudinally supported participants in Figure 4:")
    print(
        plot_df.groupby("ecological_context")
        .agg(
            n_participants=("Patient_ID", "size"),
            total_transitions=("n_transitions", "sum"),
        )
        .reindex(CONTEXT_ORDER)
        .fillna(0)
        .astype(int)
    )    

    # Diagnostics for missing context assignments.
    missing_context = plot_df["ecological_context"].isna().sum()
    if missing_context > 0:
        print(
            f"Warning: {missing_context} trace patients did not match revised Figure 3 "
            "context assignments. They will be dropped."
        )
        print(plot_df.loc[plot_df["ecological_context"].isna(), "Patient_ID"].head(20).tolist())

    plot_df = plot_df.dropna(subset=["ecological_context"]).copy()
    plot_df["ecological_context"] = plot_df["ecological_context"].astype(str)
    plot_df["diagnosis"] = plot_df["diagnosis"].map(short_diagnosis).fillna("Unknown")

    # Keep only defined context order.
    plot_df = plot_df[plot_df["ecological_context"].isin(CONTEXT_ORDER)].copy()

    # Save per-patient summaries.
    plot_df.to_csv(OUT_SUMMARY, index=False)
    print(f"Saved patient parameter summaries: {OUT_SUMMARY}")

    # Print context counts.
    print("\nContext counts in Figure 4:")
    print(plot_df["ecological_context"].value_counts().reindex(CONTEXT_ORDER).fillna(0).astype(int))

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.25],
        wspace=0.34,
        left=0.07,
        right=0.86,
        top=0.86,
        bottom=0.24,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    plot_parameter_panel(
        ax=ax_a,
        df=plot_df,
        y_mean="mu_mean",
        y_low="mu_q025",
        y_high="mu_q975",
        ylabel=r"Local attractor $\mu_i$",
        title=r"Likelihood-supported $\mu_i$ by context",
        context_colors=context_colors,
        rng=rng,
    )
    add_panel_label(ax_a, "A")

    plot_parameter_panel(
        ax=ax_b,
        df=plot_df,
        y_mean="log10_theta_mean",
        y_low="log10_theta_q025",
        y_high="log10_theta_q975",
        ylabel=r"$\log_{10}$ effective mean reversion $\theta_i$",
        title=r"Likelihood-supported $\log_{10}\theta_i$ by context",
        context_colors=context_colors,
        rng=rng,
    )
    add_panel_label(ax_b, "B")

    plot_joint_panel(
        ax=ax_c,
        df=plot_df,
        context_colors=context_colors,
    )
    add_panel_label(ax_c, "C")

    diagnoses = [d for d in ["B-ALL", "T-ALL", "ETP-ALL", "AML", "MPAL", "Unknown"]
                 if d in set(plot_df["diagnosis"])]
    make_legends(fig, context_labels, context_colors, diagnoses)

    # Main figure title is optional; I usually leave it out for journal figures.
    # fig.suptitle("OU parameter summaries by candidate ecological context", y=0.98)

    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.12)

    # Save TIFF with LZW compression.
    img = Image.open(OUT_PNG).convert("RGB")
    img.save(OUT_TIFF, dpi=(DPI, DPI), compression="tiff_lzw")

    print(f"Saved Figure 4 PNG:  {OUT_PNG}")
    print(f"Saved Figure 4 TIFF: {OUT_TIFF}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
