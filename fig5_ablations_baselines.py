#fig5_ablations_baselines.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt


VAR_OBS = "y_obs"
VAR_PPC = "y_rep"
Q_LOW, Q_HIGH = 0.025, 0.975


def _to_numpy(x):
    # ArviZ/xarray -> numpy
    try:
        return x.values
    except Exception:
        return np.asarray(x)


def posterior_predictive_stats(idata: az.InferenceData, var: str) -> dict:
    """
    Returns RMSE and 95% interval coverage using posterior predictive.
    Expects posterior_predictive[var] with dims (chain, draw, obs) or (draw, obs).
    """
    y = _to_numpy(idata.observed_data[var]).ravel()

    if not hasattr(idata, "posterior_predictive") or var not in idata.posterior_predictive:
        raise ValueError(f"Missing posterior_predictive['{var}'] in idata. "
                         "Run pm.sample_posterior_predictive and save it to the netCDF.")

    ypp = _to_numpy(idata.posterior_predictive[var])
    # collapse chain/draw dims
    ypp = ypp.reshape((-1, ypp.shape[-1]))

    pred_mean = ypp.mean(axis=0)
    rmse = float(np.sqrt(np.mean((pred_mean - y) ** 2)))

    lo = np.quantile(ypp, Q_LOW, axis=0)
    hi = np.quantile(ypp, Q_HIGH, axis=0)
    coverage = float(np.mean((y >= lo) & (y <= hi)))

    # A simple 95% interval score proxy (WIS-like component)
    # interval_score = (hi-lo) + (2/Q_LOW) * (lo-y)_+ + (2/(1-Q_HIGH)) * (y-hi)_+
    # With symmetric 95%, use alpha=0.05:
    alpha = 1 - (Q_HIGH - Q_LOW)
    interval_score = float(
        np.mean((hi - lo) + (2/alpha) * np.maximum(lo - y, 0) + (2/alpha) * np.maximum(y - hi, 0))
    )

    return {"rmse": rmse, "coverage95": coverage, "interval_score95": interval_score}


def _get_stat(obj, keys):
    # Works across ArviZ versions: attribute first, then dict-like indexing
    for k in keys:
        if hasattr(obj, k):
            return float(getattr(obj, k))
        try:
            return float(obj[k])
        except Exception:
            pass
    raise AttributeError(f"Could not find any of {keys} in {type(obj)}")

def loo_pointwise(idata: az.InferenceData) -> az.ELPDData:
    # pointwise=True is required to get loo_i for ΔELPD SE
    return az.loo(idata, pointwise=True)

def delta_elpd_and_se(loo_a: az.ELPDData, loo_b: az.ELPDData) -> tuple[float, float]:
    # ΔELPD = sum_i (loo_i[a] - loo_i[b])
    diff_i = _to_numpy(loo_a["loo_i"]) - _to_numpy(loo_b["loo_i"])
    diff_i = diff_i.ravel()
    n = diff_i.size
    delta = float(diff_i.sum())
    se = float(np.sqrt(n * np.var(diff_i, ddof=1))) if n > 1 else 0.0
    return delta, se

def load_idata(path: Path) -> az.InferenceData:
    return az.from_netcdf(path)
   
def main():
    results_dir = Path("results")
    model_files = {
        "TME-OU (full)": results_dir / "full.nc",
        "OU only (no TME)": results_dir / "ou_only.nc",
        "Random-walk latent": results_dir / "rw.nc",
        "Shuffled TME": results_dir / "shuffled.nc",
        "Static ecotype-only": results_dir / "static.nc",
    }

    loos = {}
    rows = []

    for name, fp in model_files.items():
        idata = load_idata(fp)

        loo = loo_pointwise(idata)   # <-- must exist above
        loos[name] = loo

        m = {
            "model": name,
            "elpd_loo": float(loo["elpd_loo"]),
            "p_loo": float(loo["p_loo"]),
        }
        m.update(posterior_predictive_stats(idata, VAR_OBS))
        rows.append(m)

    df = pd.DataFrame(rows).set_index("model")

    # ... keep the rest of your code here (baseline, ΔELPD+SE, plotting, saving)
    
    baseline = "OU only (no TME)"
    delta_list, se_list = [], []
    for name in df.index:
        d, se = delta_elpd_and_se(loos[name], loos[baseline])
        delta_list.append(d)
        se_list.append(se)
    
    df["delta_elpd_loo"] = delta_list
    df["delta_elpd_loo_se"] = se_list

    df.reset_index().to_csv("Fig5_metrics.csv", index=False)

    order = [
        "TME-OU (full)",
        "OU only (no TME)",
        "Shuffled TME",
        "Random-walk latent",
        "Static ecotype-only",
    ]
    d = df.loc[order].copy()

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # Panel A: ΔELPD (LOO)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(d.index, d["delta_elpd_loo"], xerr=d["delta_elpd_loo_se"], capsize=3)
    ax1.axvline(0, linewidth=1)
    ax1.set_title("A  Predictive fit (ΔELPD, PSIS-LOO)")
    ax1.set_xlabel("ΔELPD vs OU-only (higher = better)")
    
    # Main axis shows full range
    # (optionally you can force a little padding)
    ax1.margins(x=0.05)
    
    # Inset zoom
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    axins = inset_axes(
        ax1,
        width="52%",
        height="45%",
        loc="lower left",
        bbox_to_anchor=(0.18, 0.08, 1, 1),  # (x0, y0, w, h) in ax1 axes-fraction
        bbox_transform=ax1.transAxes,
        borderpad=0.0,
    )
    axins.barh(d.index, d["delta_elpd_loo"], xerr=d["delta_elpd_loo_se"], capsize=2)
    axins.axvline(0, linewidth=1)
    axins.set_xlim(-25, 5)
    axins.tick_params(axis="y", labelleft=False, left=False)
    axins.set_title("Zoom", fontsize=10)
    
    # Optional: show where inset corresponds
    ax1.indicate_inset_zoom(axins, edgecolor="black")

    # Panel B: RMSE (hide y labels)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(d.index, d["rmse"])
    ax2.set_title("B  Point accuracy (RMSE)")
    ax2.set_xlabel("RMSE (lower = better)")
    ax2.tick_params(axis="y", left=False, labelleft=False)

    # Panel C: Coverage (hide y labels)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(d["coverage95"], np.arange(len(d.index)))
    ax3.axvline(0.95, linestyle="--", linewidth=1)
    ax3.set_yticks(np.arange(len(d.index)))
    ax3.tick_params(axis="y", left=False, labelleft=False)
    ax3.set_xlim(0, 1)
    ax3.set_title("C  Uncertainty calibration")
    ax3.set_xlabel("Empirical 95% coverage (target ≈ 0.95)")

    fig.tight_layout()
    fig.savefig("Fig5_ablations_baselines.pdf", bbox_inches="tight")
    fig.savefig("Fig5_ablations_baselines.png", dpi=600, bbox_inches="tight")
    print("Wrote: Fig5_ablations_baselines.pdf/png and Fig5_metrics.csv")

if __name__ == "__main__":
    main()
