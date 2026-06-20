from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =============================================================================
# Paths
# =============================================================================

ROOT = Path("/TME_OU_Branching")
RESULTS_DIR = ROOT / "results"
OUTDIR = ROOT / "Figure_5"
OUTDIR.mkdir(exist_ok=True, parents=True)

OUT_METRICS = RESULTS_DIR / "model_comparison_revised_context.csv"
OUT_PNG = OUTDIR / "Figure5_model_comparison_revised.png"
OUT_TIFF = OUTDIR / "Figure5_model_comparison_revised.tiff"
OUT_PDF = OUTDIR / "Figure5_model_comparison_revised.pdf"


# =============================================================================
# Model files
# =============================================================================
# Edit paths if your baseline files have different names.
# The full model should be the revised-context model.

MODEL_FILES = {
    "context_ou": RESULTS_DIR / "ou_revised_ecological_context_idata_ppc.nc",
    "ou_only": RESULTS_DIR / "ou_only_revised_context_idata.nc",
    "shuffled_context": RESULTS_DIR / "shuffled_context_revised_idata.nc",
    "random_walk": RESULTS_DIR / "random_walk_revised_idata.nc",
    "static_context": RESULTS_DIR / "static_context_revised_idata.nc",
}

# If you want to temporarily use old baseline filenames, uncomment and edit:
# MODEL_FILES["ou_only"] = RESULTS_DIR / "ou_only.nc"
# MODEL_FILES["random_walk"] = RESULTS_DIR / "rw.nc"
# MODEL_FILES["shuffled_context"] = RESULTS_DIR / "shuffled.nc"
# MODEL_FILES["static_context"] = RESULTS_DIR / "static.nc"


MODEL_LABELS = {
    "context_ou": "Ecological-context OU",
    "ou_only": "OU-only",
    "shuffled_context": "Shuffled-context OU",
    "random_walk": "Random-walk latent",
    "static_context": "Static context-only",
}

# Top-to-bottom visual order
PLOT_ORDER = [
    "static_context",
    "random_walk",
    "shuffled_context",
    "ou_only",
    "context_ou",
]

BASELINE = "ou_only"

OBS_VAR = "y_obs"
PPC_CANDIDATES = ["y_obs", "y_rep"]

Q_LOW, Q_HIGH = 0.025, 0.975


# =============================================================================
# Style
# =============================================================================

MODEL_COLORS = {
    "context_ou": "#4C78A8",
    "ou_only": "#9E9E9E",
    "shuffled_context": "#72B7B2",
    "random_walk": "#F58518",
    "static_context": "#BDBDBD",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# =============================================================================
# Helpers
# =============================================================================

def _to_numpy(x):
    try:
        return x.values
    except Exception:
        return np.asarray(x)


def add_panel_label(ax, label: str, x: float = -0.16, y: float = 1.14):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def load_idata(path: Path) -> az.InferenceData:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file:\n{path}\n\n"
            "Make sure all baseline and ablation models have been fit using the same "
            "revised-context observation set."
        )
    return az.from_netcdf(path)


def get_observed_y(idata: az.InferenceData, obs_var: str = OBS_VAR) -> np.ndarray:
    if not hasattr(idata, "observed_data") or obs_var not in idata.observed_data:
        raise ValueError(f"Missing observed_data['{obs_var}'] in idata.")
    return _to_numpy(idata.observed_data[obs_var]).ravel()


def get_posterior_predictive_y(idata: az.InferenceData) -> np.ndarray:
    if not hasattr(idata, "posterior_predictive"):
        raise ValueError(
            "Missing posterior_predictive group in idata. "
            "Run pm.sample_posterior_predictive(..., extend_inferencedata=True)."
        )

    for var in PPC_CANDIDATES:
        if var in idata.posterior_predictive:
            ypp = _to_numpy(idata.posterior_predictive[var])
            ypp = ypp.reshape((-1, ypp.shape[-1]))
            return ypp

    raise ValueError(
        f"Could not find posterior predictive variable. Tried: {PPC_CANDIDATES}. "
        f"Available variables: {list(idata.posterior_predictive.data_vars)}"
    )


def posterior_predictive_stats(idata: az.InferenceData) -> dict:
    y = get_observed_y(idata, OBS_VAR)
    ypp = get_posterior_predictive_y(idata)

    if ypp.shape[-1] != len(y):
        raise ValueError(
            f"Posterior predictive obs dimension does not match observed data: "
            f"{ypp.shape[-1]} vs {len(y)}"
        )

    pred_mean = ypp.mean(axis=0)
    rmse = float(np.sqrt(np.mean((pred_mean - y) ** 2)))

    lo = np.quantile(ypp, Q_LOW, axis=0)
    hi = np.quantile(ypp, Q_HIGH, axis=0)
    coverage = float(np.mean((y >= lo) & (y <= hi)))

    alpha = 1.0 - (Q_HIGH - Q_LOW)
    interval_score = float(
        np.mean(
            (hi - lo)
            + (2 / alpha) * np.maximum(lo - y, 0)
            + (2 / alpha) * np.maximum(y - hi, 0)
        )
    )

    return {
        "rmse": rmse,
        "coverage_95": coverage,
        "interval_score_95": interval_score,
    }


def loo_pointwise(idata: az.InferenceData) -> az.ELPDData:
    if not hasattr(idata, "log_likelihood"):
        raise ValueError(
            "Missing log_likelihood group in idata. "
            "Fit model with idata_kwargs={'log_likelihood': True}."
        )
    return az.loo(idata, pointwise=True)


def delta_elpd_and_se(loo_a: az.ELPDData, loo_b: az.ELPDData) -> tuple[float, float]:
    diff_i = _to_numpy(loo_a["loo_i"]) - _to_numpy(loo_b["loo_i"])
    diff_i = diff_i.ravel()
    n = diff_i.size
    delta = float(diff_i.sum())
    se = float(np.sqrt(n * np.var(diff_i, ddof=1))) if n > 1 else 0.0
    return delta, se


def pareto_k_summary(loo: az.ELPDData) -> dict:
    if "pareto_k" not in loo:
        return {
            "pareto_k_max": np.nan,
            "pareto_k_gt_0_7": np.nan,
            "pareto_k_gt_1_0": np.nan,
        }

    pk = _to_numpy(loo["pareto_k"]).ravel()
    return {
        "pareto_k_max": float(np.nanmax(pk)),
        "pareto_k_gt_0_7": int(np.sum(pk > 0.7)),
        "pareto_k_gt_1_0": int(np.sum(pk > 1.0)),
    }


def check_same_observations(y_reference: np.ndarray, y_current: np.ndarray, model_key: str):
    if len(y_current) != len(y_reference):
        raise ValueError(
            f"{model_key} has different number of observations: "
            f"{len(y_current)} vs {len(y_reference)}"
        )

    if not np.allclose(y_current, y_reference, equal_nan=True):
        raise ValueError(
            f"{model_key} observed y_obs values differ from the baseline model. "
            "All Figure 5 models must be evaluated on the same observations in the same order."
        )


# =============================================================================
# Main
# =============================================================================

def main():
    idatas = {}
    loos = {}
    rows = []

    print("[INFO] Loading models and computing metrics...")

    y_reference = None

    for model_key, path in MODEL_FILES.items():
        print(f"\n[MODEL] {model_key}: {path}")
        idata = load_idata(path)
        idatas[model_key] = idata

        y = get_observed_y(idata, OBS_VAR)
        if y_reference is None:
            y_reference = y.copy()
            print(f"[INFO] Reference observation count: n={len(y_reference)}")
        else:
            check_same_observations(y_reference, y, model_key)

        loo = loo_pointwise(idata)
        loos[model_key] = loo

        stats = posterior_predictive_stats(idata)
        pk_stats = pareto_k_summary(loo)

        row = {
            "model_key": model_key,
            "model_label": MODEL_LABELS[model_key],
            "elpd_loo": float(loo["elpd_loo"]),
            "se_elpd_loo": float(loo["se"]),
            "p_loo": float(loo["p_loo"]),
        }
        row.update(stats)
        row.update(pk_stats)
        rows.append(row)

    if BASELINE not in loos:
        raise ValueError(f"Missing baseline model: {BASELINE}")

    df = pd.DataFrame(rows).set_index("model_key")

    # ΔELPD versus OU-only
    delta_list = []
    se_list = []
    for model_key in df.index:
        d, se = delta_elpd_and_se(loos[model_key], loos[BASELINE])
        delta_list.append(d)
        se_list.append(se)

    df["delta_elpd"] = delta_list
    df["se_delta_elpd"] = se_list

    # Ensure exact baseline visual value
    df.loc[BASELINE, "delta_elpd"] = 0.0
    df.loc[BASELINE, "se_delta_elpd"] = 0.0

    df_plot = df.loc[PLOT_ORDER].copy()
    df_plot.to_csv(OUT_METRICS)

    print("\n[INFO] Metrics:")
    print(df_plot[[
        "model_label",
        "elpd_loo",
        "delta_elpd",
        "se_delta_elpd",
        "rmse",
        "coverage_95",
        "pareto_k_max",
        "pareto_k_gt_0_7",
    ]])

    print(f"\n[INFO] Saved metrics: {OUT_METRICS}")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12.2, 4.5))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.35, 1.0, 1.0],
        wspace=0.38,
        left=0.17,
        right=0.98,
        top=0.84,
        bottom=0.22,
    )

    y_pos = np.arange(len(df_plot))
    labels = df_plot["model_label"].tolist()
    colors = [MODEL_COLORS[k] for k in df_plot.index]

    # Panel A: ΔELPD
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(
        y_pos,
        df_plot["delta_elpd"],
        xerr=df_plot["se_delta_elpd"],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        capsize=3,
    )
    ax1.axvline(0, color="black", linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_title("Predictive fit")
    ax1.set_xlabel("ΔELPD vs OU-only (higher = better)")
    ax1.grid(axis="x", color="0.9", linewidth=0.8)
    ax1.set_axisbelow(True)
    add_panel_label(ax1, "A", x=-0.30, y=1.15)

    # Inset zoom for OU variants only
    near_keys = [
        "context_ou",
        "ou_only",
        "shuffled_context",
    ]

    near = df_plot.loc[near_keys].copy()
    near_y = np.arange(len(near))
    near_colors = [MODEL_COLORS[k] for k in near.index]
    near_labels = ["Context OU", "OU-only", "Shuffled"]

    near_abs = np.nanmax(np.abs(near["delta_elpd"]) + near["se_delta_elpd"])
    near_abs = max(15.0, float(near_abs) * 1.25)

    axins = inset_axes(
        ax1,
        width="52%",
        height="42%",
        loc="lower left",
        bbox_to_anchor=(0.18, 0.08, 1, 1),
        bbox_transform=ax1.transAxes,
        borderpad=0.0,
    )

    axins.barh(
        near_y,
        near["delta_elpd"],
        xerr=near["se_delta_elpd"],
        color=near_colors,
        edgecolor="black",
        linewidth=0.3,
        capsize=2,
    )

    axins.axvline(0, color="black", linewidth=0.8)
    axins.set_xlim(-near_abs, near_abs)
    axins.set_yticks(near_y)
    axins.set_yticklabels(near_labels, fontsize=7)
    axins.invert_yaxis()
    axins.tick_params(axis="x", labelsize=7)
    axins.set_title("OU variants", fontsize=8, pad=2)
    axins.grid(axis="x", color="0.9", linewidth=0.6)
    axins.set_axisbelow(True)

    # Do not call ax1.indicate_inset_zoom(); connector lines are visually distracting.

    # Panel B: RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(
        y_pos,
        df_plot["rmse"],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", left=False)
    ax2.set_title("Point accuracy")
    ax2.set_xlabel("RMSE (lower = better)")
    ax2.grid(axis="x", color="0.9", linewidth=0.8)
    ax2.set_axisbelow(True)
    add_panel_label(ax2, "B", x=-0.22, y=1.15)

    # Panel C: coverage
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(
        df_plot["coverage_95"],
        y_pos,
        s=55,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    ax3.axvline(0.95, linestyle="--", color="black", linewidth=1)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([])
    ax3.invert_yaxis()
    ax3.tick_params(axis="y", left=False)
    ax3.set_xlim(0, 1.02)
    ax3.set_title("Uncertainty calibration")
    ax3.set_xlabel("Empirical 95% coverage")
    ax3.grid(axis="x", color="0.9", linewidth=0.8)
    ax3.set_axisbelow(True)
    add_panel_label(ax3, "C", x=-0.22, y=1.15)

    # Small note
    fig.text(
        0.17,
        0.06,
        "All comparisons use the same 434 longitudinal transitions. "
        "ΔELPD is relative to the OU-only baseline using pointwise PSIS-LOO differences;\n "
        "LOO comparisons are interpreted cautiously when Pareto-k diagnostics indicate influential observations.",
        ha="left",
        va="center",
        fontsize=9,
        color="0.25",
    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    img = Image.open(OUT_PNG).convert("RGB")
    img.save(OUT_TIFF, dpi=(600, 600), compression="tiff_lzw")

    print(f"\n[INFO] Saved:")
    print(f"  PNG : {OUT_PNG}")
    print(f"  TIFF: {OUT_TIFF}")
    print(f"  PDF : {OUT_PDF}")


if __name__ == "__main__":
    main()
