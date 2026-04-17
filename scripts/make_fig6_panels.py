# make_fig6_panels.py

from __future__ import annotations

"""
Generate Figure 6 panels (A, C, D, E) and an optional composite.

Inputs expected in --root (default: current directory or set ROOT below):
  - results/full.nc
  - kmt2a_longitudinal_clean.xlsx  (sheet: "Series" with columns Patient_ID, series, t, value)
  - STable2_tumor_TME_covariate_matrix.csv
  - STable3_posterior_by_ecotype.csv
  - STable4_posterior_by_patient.csv
Optional:
  - Fig6B.png  (schematic panel)

Outputs written to --outdir (default: ./figures):
  - Fig6A_cohort_summary.(png/pdf)
  - Fig6C_posteriors_by_ecotype.(png/pdf)
  - Fig6D_example_trajectories.(png/pdf)
  - Fig6E_k_sensitivity.(png/pdf)
  - Fig6_composite.(png/pdf)  [if --composite]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.cluster import KMeans


# -----------------------------
# Helpers
# -----------------------------
def save_both(fig, out_png: Path, out_pdf: Path, dpi: int = 600):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def mean_ci_normal(x: np.ndarray):
    """Mean +/- 1.96*SE (quick CI; avoids bootstrap runtime)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mu = float(np.mean(x))
    if n == 1:
        return mu, mu, mu, 1
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    lo = mu - 1.96 * se
    hi = mu + 1.96 * se
    return mu, lo, hi, n


def load_netcdf_groups(nc_path: Path, var: str = "y_obs"):
    """
    Load observed and posterior predictive arrays from a PyMC/ArviZ netCDF
    WITHOUT importing arviz.
    """
    obs_ds = xr.open_dataset(nc_path, group="observed_data")
    ppc_ds = xr.open_dataset(nc_path, group="posterior_predictive")

    y_obs = obs_ds[var].values.astype(float).ravel()
    ypp = ppc_ds[var].values  # (chain, draw, obs)
    ypp = ypp.reshape((-1, ypp.shape[-1]))  # (samples, obs)
    return y_obs, ypp

def make_fig6B_schematic(schematic_png: Path, outdir: Path):
    """
    Panel 6B = schematic image. Copy it into outdir with standard names.
    """
    if not schematic_png.exists():
        raise FileNotFoundError(f"Schematic not found: {schematic_png}")

    out_png = outdir / "Fig6B_schematic.png"
    out_pdf = outdir / "Fig6B_schematic.pdf"

    # Copy PNG bytes
    out_png.write_bytes(schematic_png.read_bytes())

    # Also write a simple PDF wrapper
    img = plt.imread(schematic_png)
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    save_both(fig, out_png, out_pdf)  # overwrites PNG identically; OK
    return out_png, out_pdf

# -----------------------------
# Panel 6A
# -----------------------------
def make_fig6A(df2: pd.DataFrame, df4: pd.DataFrame, outdir: Path):
    # Merge covariates with ecotype labels
    m = df2.merge(df4[["Patient_ID", "ecotype_label"]], on="Patient_ID", how="left")

    cov_cols = [
        "frac_T_given_known_z",
        "frac_B_given_known_z",
        "frac_myeloid_given_known_z",
        "frac_NK_given_known_z",
        "frac_stromal_given_known_z",
        "frac_unknown_z",
    ]
    cov_cols = [c for c in cov_cols if c in m.columns]

    eco_order = ["E1", "E2", "E3", "E4"]
    eco_order = [e for e in eco_order if e in set(m["ecotype_label"].dropna())]

    counts = m["ecotype_label"].value_counts().reindex(eco_order)
    means = m.groupby("ecotype_label")[cov_cols].mean().reindex(eco_order)

    fig = plt.figure(figsize=(10.5, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2])

    # One “6A” label for the whole figure (not repeated)
    fig.text(-0.02, 0.98, "6A", ha="left", va="top", fontsize=16, fontweight="bold")

    # Left: ecotype composition
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(counts.index, counts.values)
    ax1.set_title("Ecotype composition")   # <- removed "6A"
    ax1.set_xlabel("Ecotype")
    ax1.set_ylabel("Patients (n)")
    total = counts.sum()
    for i, y in enumerate(counts.values):
        pct = 100.0 * y / total
        ax1.text(i, y + max(counts.values) * 0.01, f"{pct:.1f}%",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_ylim(0, max(counts.values) * 1.10)

    # Right: heatmap of mean z covariates
    ax2 = fig.add_subplot(gs[0, 1])
    data = means.values
    im = ax2.imshow(data, aspect="auto")
    ax2.set_title("Mean TME covariates (z-score)")  # <- removed "6A"

    xt = [c.replace("frac_", "").replace("_given_known_z", "").replace("_z", "") for c in cov_cols]
    ax2.set_xticks(np.arange(len(cov_cols)))
    ax2.set_xticklabels(xt, rotation=30, ha="right")

    ax2.set_yticks(np.arange(len(eco_order)))
    ax2.set_yticklabels(eco_order)

    # numeric annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax2.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Mean z")

    out_png = outdir / "Fig6A_cohort_summary.png"
    out_pdf = outdir / "Fig6A_cohort_summary.pdf"
    save_both(fig, out_png, out_pdf)
    return out_png, out_pdf


# -----------------------------
# Panel 6C
# -----------------------------
def make_fig6C(df3: pd.DataFrame, df4: pd.DataFrame, outdir: Path):
    df4c = df4.copy()
    df4c["ecotype_label"] = df4c["ecotype_label"].astype(str)

    eco_order = ["E1", "E2", "E3", "E4"]
    eco_order = [e for e in eco_order if e in set(df4c["ecotype_label"].dropna())]

    df3i = df3.set_index("ecotype") if "ecotype" in df3.columns else df3.set_index("ecotype_label")

    def violin_with_ecotype_interval(ax, col_patient, col_ecotype_lo, col_ecotype_hi,
                                     ylab, title):
        vals = [df4c.loc[df4c["ecotype_label"] == e, col_patient].values for e in eco_order]
        ax.violinplot(vals, showmeans=False, showmedians=True, showextrema=False)

        rng = np.random.default_rng(0)
        for i, e in enumerate(eco_order, start=1):
            v = df4c.loc[df4c["ecotype_label"] == e, col_patient].values
            if len(v) == 0:
                continue
            jitter = (rng.random(len(v)) - 0.5) * 0.15
            ax.scatter(np.full(len(v), i) + jitter, v, s=18, alpha=0.6)

            if e in df3i.index and (col_ecotype_lo in df3i.columns) and (col_ecotype_hi in df3i.columns):
                lo = float(df3i.loc[e, col_ecotype_lo])
                hi = float(df3i.loc[e, col_ecotype_hi])
                ax.plot([i, i], [lo, hi], linewidth=2)
                ax.scatter([i], [(lo + hi) / 2.0], s=55, zorder=3)

        ax.set_xticks(range(1, len(eco_order) + 1))
        ax.set_xticklabels(eco_order)
        ax.set_ylabel(ylab)
        ax.set_title(title)

    # Wider + shorter -> looks larger inside the composite
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.30)

    # One “6C” label for the whole figure (not repeated)
    fig.text(0.06, 0.90, "6C", ha="left", va="top", fontsize=24, fontweight="bold")

    ax_mu = fig.add_subplot(gs[0, 0])
    violin_with_ecotype_interval(
        ax_mu,
        col_patient="mu_mean",
        col_ecotype_lo="mu_p03",
        col_ecotype_hi="mu_p97",
        ylab="μ (posterior mean per patient)",
        title="Drift/optimum by ecotype",
    )

    ax_th = fig.add_subplot(gs[0, 1])
    violin_with_ecotype_interval(
        ax_th,
        col_patient="log10_theta_mean",
        col_ecotype_lo="log10theta_p03",
        col_ecotype_hi="log10theta_p97",
        ylab=r"log$_{10}\,\theta$ (posterior mean per patient)",
        title="Stabilizing selection by ecotype",
    )

    out_png = outdir / "Fig6C_posteriors_by_ecotype.png"
    out_pdf = outdir / "Fig6C_posteriors_by_ecotype.pdf"
    save_both(fig, out_png, out_pdf)
    return out_png, out_pdf


# -----------------------------
# Panel 6D
# -----------------------------
def make_fig6D(
    xlsx_path: Path,
    full_nc_path: Path,
    df2: pd.DataFrame,
    df4: pd.DataFrame,
    outdir: Path,
    exemplars: list[tuple[str, str]] | None = None,
):
    """
    Example trajectories with posterior predictive uncertainty.
    Uses the same ordering as the OU calibration model:
      - pat_index_map from df2 Patient_ID order
      - sort by pat_index, series, t
      - transitions are those with dt>0, aligned to y_obs in full.nc
    """
    # netcdf groups
    y_obs_vec, ypp = load_netcdf_groups(full_nc_path, var="y_obs")

    # patient order map must match the model run
    patient_ids = df2["Patient_ID"].tolist()
    pat_index_map = {pid: i for i, pid in enumerate(patient_ids)}

    # ecotype labels
    pat_to_eco = dict(zip(df4["Patient_ID"], df4["ecotype_label"]))

    raw = pd.read_excel(xlsx_path, sheet_name="Series")
    raw["pat_index"] = raw["Patient_ID"].map(pat_index_map)
    raw = raw.dropna(subset=["pat_index"]).copy()
    raw["pat_index"] = raw["pat_index"].astype(int)
    raw = raw.sort_values(["pat_index", "series", "t"]).reset_index(drop=True)

    # transitions (aligned to y_obs order)
    raw["y_prev"] = raw.groupby(["pat_index", "series"])["value"].shift(1)
    raw["dt"] = raw.groupby(["pat_index", "series"])["t"].diff()
    trans = raw.dropna(subset=["y_prev", "dt"]).copy()
    trans = trans[trans["dt"] > 0].reset_index(drop=True)
    trans = trans.rename(columns={"value": "y", "t": "time"})
    trans["obs_index"] = np.arange(len(trans))

    # verify alignment with netcdf observed vector
    if len(trans) != len(y_obs_vec):
        raise ValueError(f"Mismatch: transitions={len(trans)} vs y_obs={len(y_obs_vec)}. "
                         "This usually means Patient_ID ordering differs from the model run.")
    maxdiff = float(np.max(np.abs(trans["y"].values.astype(float) - y_obs_vec)))
    if maxdiff > 1e-9:
        raise ValueError("Transition ordering differs from netCDF y_obs. "
                         "Check Patient_ID ordering and sorting logic.")

    trans["ecotype_label"] = trans["Patient_ID"].map(pat_to_eco)

    # choose exemplars if not provided: top n_trans for series "n" first, else overall
    if exemplars is None:
        counts = (
            trans.groupby(["Patient_ID", "series"])["obs_index"]
            .count()
            .reset_index(name="n_trans")
            .sort_values("n_trans", ascending=False)
        )
        exemplars = []
        for _, r in counts.iterrows():
            pid, s = r["Patient_ID"], r["series"]
            if (pid, s) not in exemplars:
                exemplars.append((pid, s))
            if len(exemplars) >= 4:
                break

    exemplars = exemplars[:4]

    def summarize_ppc(obs_indices):
        draws = ypp[:, obs_indices]  # (samples, n)
        med = np.quantile(draws, 0.5, axis=0)
        lo = np.quantile(draws, 0.025, axis=0)
        hi = np.quantile(draws, 0.975, axis=0)
        return med, lo, hi

    def plot_patient_series(ax, pid, series):
        df_obs = raw[(raw["Patient_ID"] == pid) & (raw["series"] == series)].copy()
        df_obs = df_obs.sort_values("t")
        t_all = df_obs["t"].values.astype(float)
        y_all = df_obs["value"].values.astype(float)

        df_tr = trans[(trans["Patient_ID"] == pid) & (trans["series"] == series)].copy()
        df_tr = df_tr.sort_values("time")
        t_pred = df_tr["time"].values.astype(float)
        obs_idx = df_tr["obs_index"].values.astype(int)

        eco = pat_to_eco.get(pid, "NA")
        med, lo, hi = summarize_ppc(obs_idx)

        ax.fill_between(t_pred, lo, hi, alpha=0.25)
        ax.plot(t_pred, med, linewidth=2, label="PPC median")
        ax.plot(t_all, y_all, marker="o", linewidth=1.5, label="Observed")

        ax.set_title(f"{pid} ({eco}), series {series} (n={len(df_tr)})")
        ax.set_xlabel("t")
        ax.set_ylabel("Trait (ScPCA summary)")

    # ---- IMPORTANT: figure creation is OUTSIDE plot_patient_series ----
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.35)

    # One “6D” label for the whole figure (not repeated)
    fig.subplots_adjust(top=0.90)  # smaller top -> plots move up
    fig.text(0.05, 0.97, "6D", ha="left", va="top", fontsize=20, fontweight="bold")

    for i, (pid, series) in enumerate(exemplars):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        plot_patient_series(ax, pid, series)
        if i == 0:
            ax.legend(frameon=False, fontsize=9, loc="best")
        else:
            ax.legend().remove()

    # No "6D" here (label is handled by fig.text)
    fig.suptitle("Example patient trajectories with posterior predictive uncertainty")

    out_png = outdir / "Fig6D_example_trajectories.png"
    out_pdf = outdir / "Fig6D_example_trajectories.pdf"
    save_both(fig, out_png, out_pdf)
    return out_png, out_pdf


# -----------------------------
# Panel 6E
# -----------------------------
def make_fig6E(df2: pd.DataFrame, df4: pd.DataFrame, outdir: Path):
    cov_cols = [
        "frac_T_given_known_z",
        "frac_B_given_known_z",
        "frac_myeloid_given_known_z",
        "frac_NK_given_known_z",
        "frac_stromal_given_known_z",
        "frac_unknown_z",
    ]
    cov_cols = [c for c in cov_cols if c in df2.columns]

    m = df2.merge(df4[["Patient_ID", "log10_theta_mean"]], on="Patient_ID", how="inner").copy()
    m = m.dropna(subset=cov_cols + ["log10_theta_mean"]).reset_index(drop=True)

    X = m[cov_cols].values.astype(float)
    y_theta = m["log10_theta_mean"].values.astype(float)
    stromal = m["frac_stromal_given_known_z"].values.astype(float)

    Ks = [3, 4, 5, 6]
    summaries = []

    for k in Ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=0)
        lab = km.fit_predict(X)

        # order clusters by mean stromal z
        order = sorted(range(k), key=lambda c: float(np.nanmean(stromal[lab == c])))
        metric_vals = [float(np.nanmean(stromal[lab == c])) for c in order]
        remap = {old: new for new, old in enumerate(order, start=1)}
        lab_ord = np.array([remap[x] for x in lab], dtype=int)

        for c in range(1, k + 1):
            vals = y_theta[lab_ord == c]
            mu, lo, hi, n = mean_ci_normal(vals)
            summaries.append(
                {
                    "k": k,
                    "cluster": f"C{c}",
                    "n": n,
                    "theta_mean": mu,
                    "theta_lo": lo,
                    "theta_hi": hi,
                    "mean_stromal_z": metric_vals[c - 1],
                }
            )

    sum_df = pd.DataFrame(summaries)
    (outdir / "Fig6E_k_sensitivity_summary.csv").write_text(sum_df.to_csv(index=False))

    fig = plt.figure(figsize=(11, 8.8))
    gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.80)
    
    for i, k in enumerate(Ks):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        sub = sum_df[sum_df["k"] == k].copy()
        x_pos = np.arange(1, len(sub) + 1)
    
        means = sub["theta_mean"].values
        yerr = np.vstack([means - sub["theta_lo"].values, sub["theta_hi"].values - means])
    
        ax.errorbar(x_pos, means, yerr=yerr, fmt="o", capsize=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sub["cluster"].tolist())
        ax.set_title(f"Alternative k-means (k={k})")
        ax.set_ylabel(r"log$_{10}\,\theta$ (patient posterior mean)")
        ax.set_xlabel("Clusters ordered by mean stromal z")
    
        # Rotated annotations
        for xp, n, mv in zip(x_pos, sub["n"].values, sub["mean_stromal_z"].values):
            ax.text(
                xp, -0.28,
                f"n={int(n)}, stromal_z={mv:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="right", va="top",
                rotation=45, rotation_mode="anchor",
                fontsize=8,
            )
    
        ax.margins(y=0.15)
    
    fig.suptitle("Sensitivity to ecotype resolution (k=3–6): log10 θ by cluster", y=0.98)
    #fig.tight_layout(rect=[0, 0.10, 1, 0.96])

    out_png = outdir / "Fig6E_k_sensitivity.png"
    out_pdf = outdir / "Fig6E_k_sensitivity.pdf"
    save_both(fig, out_png, out_pdf)
    return out_png, out_pdf


# -----------------------------
# Composite stitcher
# -----------------------------
def make_composite(outdir: Path):
    """
    Composite (A, B, C, D, E) where:
      A = cohort summary
      B = schematic
      C = posteriors by ecotype
      D = trajectories
      E = k-sensitivity
    """
    a = outdir / "Fig6A_cohort_summary.png"
    b = outdir / "Fig6B_schematic.png"
    c = outdir / "Fig6C_posteriors_by_ecotype.png"
    d = outdir / "Fig6D_example_trajectories.png"
    e = outdir / "Fig6E_k_sensitivity.png"

    for p in [a, b, c, d, e]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run panel generation first.")

    A = plt.imread(a)
    B = plt.imread(b)
    C = plt.imread(c)
    D = plt.imread(d)
    E = plt.imread(e)

    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.0, 1.1, 1.6, 1.4],
        hspace=0.22, wspace=0.12
    )

    axA = fig.add_subplot(gs[0, 0]); axA.imshow(A); axA.axis("off")
    axB = fig.add_subplot(gs[0, 1]); axB.imshow(B); axB.axis("off")

    axC = fig.add_subplot(gs[1, :]); axC.imshow(C); axC.axis("off")
    axD = fig.add_subplot(gs[2, :]); axD.imshow(D); axD.axis("off")
    axE = fig.add_subplot(gs[3, :]); axE.imshow(E); axE.axis("off")

    out_png = outdir / "Fig6_composite.png"
    out_pdf = outdir / "Fig6_composite.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote {out_png} and {out_pdf}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Project root containing inputs")
    ap.add_argument("--outdir", type=str, default="Figure_6", help="Output directory")
    ap.add_argument("--composite", action="store_true", help="Also stitch a composite Fig6")
    ap.add_argument("--fig6b", type=str, default="", help="Path to Fig6B.png (optional)")
    ap.add_argument("--exemplars", type=str, default="",
                    help='Optional exemplars like "P63:n,P17:n,P28:n,P100:n"')
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Inputs
    st_dir = root / "Supplementary" / "STables"
    
    df2 = pd.read_csv(st_dir / "STable2_tumor_TME_covariate_matrix.csv")
    df3 = pd.read_csv(st_dir / "STable3_posterior_by_ecotype.csv")
    df4 = pd.read_csv(st_dir / "STable4_posterior_by_patient.csv")
    xlsx = root / "kmt2a_longitudinal_clean.xlsx"
    full_nc = root / "results" / "full.nc"

    # Parse exemplars if provided
    exemplars = None
    if args.exemplars.strip():
        exemplars = []
        for tok in args.exemplars.split(","):
            pid, ser = tok.split(":")
            exemplars.append((pid.strip(), ser.strip()))

    # Generate panels
    make_fig6A(df2, df4, outdir)
    
    # 6B schematic (swap: schematic is now B)
    schematic_path = (root / "Figure_6" / "Fig6B.png").resolve()
    if not schematic_path.exists():
        raise FileNotFoundError(f"Cannot find schematic: {schematic_path}")
    #schematic_path = (Path(args.fig6b).expanduser().resolve()
                      #if args.fig6b.strip() else (root / "Fig6B.png"))
    make_fig6B_schematic(schematic_path, outdir)
    
    # 6D trajectories (swap: trajectories are now D)
    make_fig6D(xlsx, full_nc, df2, df4, outdir, exemplars=exemplars)
    
    make_fig6C(df3, df4, outdir)
    make_fig6E(df2, df4, outdir)
    
    # Composite (optional)
    if args.composite:
        fig6b_path = (Path(args.fig6b).expanduser().resolve()
                      if args.fig6b.strip() else (root / "Fig6B.png"))
        make_composite(outdir, fig6d_path=fig6b_path)  # rename arg if your function uses fig6b_path


if __name__ == "__main__":
    main()
