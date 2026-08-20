from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ---------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------

CONTEXT_COLORS = {
    "TNK-dominant": "#4C78A8",
    "B/Myeloid-shifted": "#F28E2B",
}

MODEL_LABELS = {
    "M_0": "OU only",
    "M_B": "OU + B",
    "M_M": "OU + Myeloid",
    "M_E": "OU + B + Myeloid",
    "M_C": "OU + context",
    "M_EC": "OU + ecology + context",
    "M_S": "OU + shuffled context",
    "X_0": "OU only",
    "X_B": "OU + B",
    "X_E": "OU + B + Myeloid",
    "X_C": "OU + context",
    "N_0": "Previous state + Δt",
    "N_B": "+ B",
    "N_E": "+ B + Myeloid",
    "N_C": "+ context",
    "N_EC": "+ ecology + context",
}

MAIN_MODELS = ["M_0", "M_B", "M_E", "M_C", "M_EC", "M_S"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require(path))


def save_figure(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[SAVED] {stem}")


def panel_label(ax, label):
    ax.text(
        -0.14, 1.10, label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )


def ternary_xy(tnk, b, myeloid):
    """
    Triangle vertices:
      TNK = left
      B = right
      Myeloid = top
    Fractions sum to 1.
    """
    tnk = np.asarray(tnk, float)
    b = np.asarray(b, float)
    my = np.asarray(myeloid, float)
    x = b + 0.5 * my
    y = (np.sqrt(3) / 2.0) * my
    return x, y


def draw_ternary_frame(ax):
    h = np.sqrt(3) / 2
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], linewidth=1.2)
    ax.text(-0.03, -0.04, "TNK", ha="right", va="top", fontweight="bold")
    ax.text(1.03, -0.04, "B", ha="left", va="top", fontweight="bold")
    ax.text(0.5, h + 0.045, "Myeloid", ha="center", va="bottom", fontweight="bold")
    for frac in [0.25, 0.5, 0.75]:
        # Light internal guide lines.
        ax.plot([frac, 0.5 + frac/2], [0, h*(1-frac)], linewidth=0.45, alpha=0.25)
        ax.plot([1-frac, 0.5 - frac/2], [0, h*(1-frac)], linewidth=0.45, alpha=0.25)
        ax.plot([frac/2, 1-frac/2], [h*frac, h*frac], linewidth=0.45, alpha=0.25)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, h + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")


def horizontal_forest(ax, df, ylabels, mean_col="mean",
                      lo_col="hdi_2.5%", hi_col="hdi_97.5%"):
    y = np.arange(len(df))
    means = df[mean_col].to_numpy(float)
    lo = df[lo_col].to_numpy(float)
    hi = df[hi_col].to_numpy(float)
    ax.errorbar(
        means, y,
        xerr=np.vstack([means-lo, hi-means]),
        fmt="o", capsize=3, linewidth=1.2
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()


def model_order_subset(df, order):
    x = df.set_index("model").reindex([m for m in order if m in set(df["model"])])
    return x.reset_index()


# ---------------------------------------------------------------------
# Load canonical data
# ---------------------------------------------------------------------

def load_all(root: Path):
    data = {}

    # Wave 1
    w1s = root / "revision_wave1_context_stability"
    w1b = root / "revision_wave1_id_ball"
    data["primary_comp"] = read_csv(w1s / "primary_composition_participant.csv")
    data["raw_scan"] = read_csv(w1s / "k_scan_raw.csv")
    data["clr_scan"] = read_csv(w1s / "k_scan_clr.csv")
    data["concordance"] = read_csv(w1s / "raw_clr_concordance.csv")
    data["boot_raw"] = read_csv(w1s / "bootstrap_cluster_stability_raw.csv")
    data["ball_raw_scan"] = read_csv(w1b / "B_ALL_k_scan_raw.csv")
    data["ball_boot_raw"] = read_csv(w1b / "B_ALL_bootstrap_raw.csv")
    data["ball_comp"] = read_csv(w1b / "B_ALL_primary_composition.csv")
    data["ball_assign_k2"] = read_csv(w1b / "B_ALL_cluster_assignments_raw_k2.csv")
    data["flow"] = read_csv(w1b / "participant_flow_summary.csv")
    data["ball_project"] = read_csv(w1b / "B_ALL_raw_k2_project_crosstab.csv")

    # Wave 2
    w2i = root / "revision_wave2_inputs"
    w2m = root / "revision_wave2_models"
    w2l = root / "revision_wave2_lopo"
    data["x_participant"] = read_csv(w2i / "wave2_x_participant_summary.csv")
    data["x_sampling"] = read_csv(w2i / "wave2_x_sampling_summary.csv")
    data["coef"] = read_csv(w2m / "inspection" / "wave2_coefficient_posterior_summary.csv")
    data["full_loo"] = read_csv(w2m / "diagnostics" / "wave2_ablation_loo_summary.csv")
    data["pareto"] = read_csv(w2m / "inspection" / "wave2_pareto_gt_07.csv")
    data["lopo_summary"] = read_csv(w2l / "wave2_lopo_model_summary.csv")
    data["lopo_pair"] = read_csv(w2l / "wave2_lopo_pairwise_differences.csv")
    data["lopo_part"] = read_csv(w2l / "wave2_lopo_participant_results.csv")

    # Secondary Wave 2
    w2s = root / "revision_wave2_secondary_models" / "diagnostics"
    data["secondary_loo"] = read_csv(w2s / "wave2_secondary_loo_summary.csv")
    data["secondary_coef"] = read_csv(w2s / "wave2_secondary_coefficient_summary.csv")

    w2tx = root / "revision_wave2_lopo_transformed_x_eps005"
    data["tx_lopo_summary"] = read_csv(w2tx / "wave2_lopo_eps005_model_summary.csv")
    data["tx_lopo_pair"] = read_csv(w2tx / "wave2_lopo_eps005_pairwise_differences.csv")

    # Wave 3
    w3p = root / "revision_wave3_prior_sensitivity"
    data["prior_summary"] = read_csv(w3p / "wave3_prior_sensitivity_summary.csv")
    data["prior_coef"] = read_csv(w3p / "wave3_prior_coefficient_summary.csv")

    w3r = root / "revision_wave3_parameter_recovery"
    data["recovery"] = read_csv(w3r / "recovery_summary.csv")

    w3c = root / "revision_wave3_ppc"
    data["ppc"] = read_csv(w3c / "wave3_ppc_model_summary.csv")
    data["ppc_context"] = read_csv(w3c / "wave3_ppc_by_context.csv")

    return data


# ---------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------

def make_figure2(data, outdir: Path):
    comp = data["primary_comp"].copy()
    ball = data["ball_comp"].copy()
    assign = data["ball_assign_k2"][["participant_id", "cluster"]].copy()
    ball = ball.merge(assign, on="participant_id", how="left", validate="one_to_one")
    ball["context"] = ball["cluster"].map({
        "C1": "TNK-dominant",
        "C2": "B/Myeloid-shifted",
    })

    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # A: ternary by project
    draw_ternary_frame(axA)
    proj_markers = {"SCPCP000008": "o", "SCPCP000022": "^"}
    for proj, g in comp.groupby("project"):
        x, y = ternary_xy(g["comp_TNK"], g["comp_B"], g["comp_Myeloid"])
        axA.scatter(x, y, s=34, alpha=0.75, marker=proj_markers.get(proj, "o"),
                    label=f"{proj} (n={len(g)})", edgecolor="white", linewidth=0.4)
    axA.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    axA.set_title("Harmonized TNK/B/Myeloid composition by project")
    panel_label(axA, "A")

    # B: ternary by B-ALL k2 context
    draw_ternary_frame(axB)
    for ctx, g in ball.groupby("context"):
        x, y = ternary_xy(g["comp_TNK"], g["comp_B"], g["comp_Myeloid"])
        axB.scatter(
            x, y, s=36, alpha=0.80,
            color=CONTEXT_COLORS[ctx],
            label=f"{ctx} (n={len(g)})",
            edgecolor="white", linewidth=0.4
        )
    axB.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    axB.set_title("B-ALL coarse immune-compositional contexts")
    panel_label(axB, "B")

    # C: mean composition bars
    means = (
        ball.groupby("context")[["comp_TNK", "comp_B", "comp_Myeloid"]]
        .mean()
        .reindex(["TNK-dominant", "B/Myeloid-shifted"])
    )
    x = np.arange(len(means))
    bottom = np.zeros(len(means))
    for col, lab in [("comp_TNK","TNK"),("comp_B","B"),("comp_Myeloid","Myeloid")]:
        vals = means[col].to_numpy()
        axC.bar(x, vals, bottom=bottom, label=lab, width=0.65, edgecolor="white")
        bottom += vals
    axC.set_xticks(x)
    axC.set_xticklabels(["TNK-dominant", "B/Myeloid-shifted"])
    axC.set_ylabel("Mean compositional fraction")
    axC.set_ylim(0, 1)
    axC.legend(frameon=False, ncol=3, loc="upper center")
    axC.set_title("Mean immune composition of the two candidate contexts")
    panel_label(axC, "C")

    # D: project proportions within B-ALL k2
    ct = pd.crosstab(ball["project"], ball["context"], normalize="index")
    ct = ct.reindex(columns=["TNK-dominant", "B/Myeloid-shifted"]).fillna(0)
    xx = np.arange(len(ct))
    bottom = np.zeros(len(ct))
    for ctx in ct.columns:
        vals = ct[ctx].to_numpy()
        axD.bar(xx, vals, bottom=bottom, color=CONTEXT_COLORS[ctx],
                label=ctx, width=0.62, edgecolor="white")
        bottom += vals
    axD.set_xticks(xx)
    axD.set_xticklabels(ct.index)
    axD.set_ylim(0, 1)
    axD.set_ylabel("Proportion of B-ALL participants")
    axD.legend(frameon=False, loc="upper center", ncol=2)
    axD.set_title("Nearly identical context proportions across projects")
    panel_label(axD, "D")

    fig.suptitle(
        "Figure 2. Harmonized immune-compositional landscape across ScPCA projects",
        fontsize=15, y=0.995
    )
    save_figure(fig, outdir / "Figure2_harmonized_immune_compositional_landscape")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------

def make_figure3(data, outdir: Path):
    raw = data["raw_scan"]
    clr = data["clr_scan"]
    conc = data["concordance"]
    boot = data["boot_raw"]
    ballboot = data["ball_boot_raw"]
    ballscan = data["ball_raw_scan"]

    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)
    axA = fig.add_subplot(gs[0,0])
    axB = fig.add_subplot(gs[0,1])
    axC = fig.add_subplot(gs[1,0])
    axD = fig.add_subplot(gs[1,1])

    # A raw k scan
    axA.plot(raw["k"], raw["silhouette"], marker="o", label="Silhouette")
    axA.plot(raw["k"], raw["project_cramers_v"], marker="s", label="Project Cramér's V")
    axA.plot(raw["k"], raw["diagnosis_cramers_v"], marker="^", label="Diagnosis Cramér's V")
    axA.axvline(2, linestyle="--", linewidth=1)
    axA.set_xticks(raw["k"])
    axA.set_xlabel("Number of clusters (k)")
    axA.set_ylabel("Metric")
    axA.set_title("Raw-composition resolution scan")
    axA.legend(frameon=False)
    panel_label(axA, "A")

    # B bootstrap k2 full vs B-ALL
    def grab_k2(df, cluster):
        z = df[(df["k"]==2) & (df["cluster"]==cluster)]
        if len(z) == 0:
            return np.nan, np.nan, np.nan
        r=z.iloc[0]
        return r["mean"], r["q05"], r["q95"]

    labels=["Global agreement","C1 Jaccard","C2 Jaccard"]
    clusters=["ALL","C1","C2"]
    y=np.arange(3)
    for j,(name,df) in enumerate([("All diagnoses",boot),("B-ALL only",ballboot)]):
        means=[]; lows=[]; highs=[]
        for c in clusters:
            m,l,h=grab_k2(df,c)
            means.append(m); lows.append(l); highs.append(h)
        offset=(-0.10 if j==0 else 0.10)
        axB.errorbar(means, y+offset,
                     xerr=np.vstack([np.array(means)-np.array(lows),
                                     np.array(highs)-np.array(means)]),
                     fmt="o", capsize=3, label=name)
    axB.set_yticks(y); axB.set_yticklabels(labels)
    axB.invert_yaxis()
    axB.set_xlim(0,1.03)
    axB.set_xlabel("Bootstrap stability")
    axB.set_title("k=2 resampling stability")
    axB.legend(frameon=False)
    panel_label(axB, "B")

    # C B-ALL raw scan project dependence
    axC.plot(ballscan["k"], ballscan["silhouette"], marker="o", label="Silhouette")
    axC.plot(ballscan["k"], ballscan["project_cramers_v"], marker="s", label="Project Cramér's V")
    axC.axvline(2, linestyle="--", linewidth=1)
    axC.set_xticks(ballscan["k"])
    axC.set_xlabel("Number of clusters (k)")
    axC.set_ylabel("Metric")
    axC.set_title("Within-B-ALL sensitivity")
    axC.legend(frameon=False)
    panel_label(axC, "C")

    # D raw vs CLR
    axD.plot(conc["k"], conc["raw_vs_clr_ARI"], marker="o")
    axD.axhline(0, linestyle="--", linewidth=1)
    axD.set_xticks(conc["k"])
    axD.set_xlabel("Number of clusters (k)")
    axD.set_ylabel("Adjusted Rand index")
    axD.set_title("Discrete assignments depend on compositional representation")
    panel_label(axD, "D")

    fig.suptitle(
        "Figure 3. Stability and sensitivity of candidate ecological contexts",
        fontsize=15, y=0.995
    )
    save_figure(fig, outdir / "Figure3_context_stability_and_sensitivity")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------

def make_figure4(data, outdir: Path):
    coef=data["coef"]
    lopo=data["lopo_summary"]
    lopop=data["lopo_part"]

    fig=plt.figure(figsize=(14.5,10.5))
    gs=fig.add_gridspec(2,2,hspace=0.40,wspace=0.34)
    axA=fig.add_subplot(gs[0,0])
    axB=fig.add_subplot(gs[0,1])
    axC=fig.add_subplot(gs[1,0])
    axD=fig.add_subplot(gs[1,1])

    # A schematic
    axA.axis("off")
    boxes=[
        (0.05,0.58,0.23,0.20,"Continuous\necology\nB, Myeloid"),
        (0.39,0.58,0.23,0.20,"OU parameters\nμ and θ"),
        (0.73,0.58,0.23,0.20,"Longitudinal x\ntransition\nlikelihood"),
    ]
    for x,y,w,h,txt in boxes:
        patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02",
                             linewidth=1.2,transform=axA.transAxes,fill=False)
        axA.add_patch(patch)
        axA.text(x+w/2,y+h/2,txt,ha="center",va="center",
                 transform=axA.transAxes,fontsize=11)
    for x1,x2 in [(0.28,0.39),(0.62,0.73)]:
        axA.add_patch(FancyArrowPatch((x1,0.68),(x2,0.68),
                                     arrowstyle="->",mutation_scale=12,
                                     transform=axA.transAxes))
    axA.text(0.50,0.30,
             "Primary endpoint: x only\n34 participants • 207 transitions\n"
             "Participant-level validation",
             ha="center",va="center",transform=axA.transAxes,fontsize=11)
    axA.set_title("Revised longitudinal modeling architecture")
    panel_label(axA,"A")

    # B forest coefficients
    subset=coef[
        ((coef["model"]=="M_B") & (coef["predictor"]=="B_z")) |
        ((coef["model"]=="M_E") & (coef["predictor"].isin(["B_z","Myeloid_z"]))) |
        ((coef["model"]=="M_C") & (coef["predictor"]=="Context_z"))
    ].copy()
    subset["label"]=subset.apply(
        lambda r: f'{r["model"]}: {r["parameter"].replace("beta_","")} {r["predictor"].replace("_z","")}',
        axis=1
    )
    subset=subset.sort_values(["model","parameter","predictor"])
    horizontal_forest(axB,subset,subset["label"].tolist())
    axB.set_xlabel("Posterior coefficient (95% HDI)")
    axB.set_title("Ecological associations with OU parameters")
    panel_label(axB,"B")

    # C LOPO delta ELPD relative to OU-only
    s=model_order_subset(lopo,MAIN_MODELS)
    baseline=float(s.loc[s["model"].eq("M_0"),"total_elpd"].iloc[0])
    s["delta_elpd_vs_M0"]=s["total_elpd"]-baseline
    yy=np.arange(len(s))
    axC.barh(yy,s["delta_elpd_vs_M0"])
    axC.axvline(0,linestyle="--",linewidth=1)
    axC.set_yticks(yy)
    axC.set_yticklabels([MODEL_LABELS.get(m,m) for m in s["model"]])
    axC.invert_yaxis()
    axC.set_xlabel("Δ participant-held-out ELPD vs OU only")
    axC.set_title("Participant-level predictive validation")
    panel_label(axC,"C")

    # D per-patient MB-M0 deltas
    pivot=lopop.pivot(index="Patient_ID",columns="model",values="heldout_elpd")
    if {"M_B","M_0"}.issubset(pivot.columns):
        delta=(pivot["M_B"]-pivot["M_0"]).sort_values()
        xx=np.arange(len(delta))
        axD.bar(xx,delta.values)
        axD.axhline(0,linestyle="--",linewidth=1)
        axD.set_xticks([])
        axD.set_ylabel("ΔELPD (M_B − M_0)")
        axD.set_xlabel("Held-out participants (ordered)")
        axD.set_title("B-model advantage is modest and participant-heterogeneous")
    panel_label(axD,"D")

    fig.suptitle(
        "Figure 4. Continuous ecological composition and revised OU dynamics",
        fontsize=15,y=0.995
    )
    save_figure(fig,outdir/"Figure4_revised_OU_continuous_ecology")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------

def make_figure5(data, outdir: Path):
    tx=data["tx_lopo_summary"]
    sec=data["secondary_coef"]
    prior=data["prior_coef"]
    ppc=data["ppc"]

    fig=plt.figure(figsize=(14.5,10.5))
    gs=fig.add_gridspec(2,2,hspace=0.40,wspace=0.34)
    axA=fig.add_subplot(gs[0,0])
    axB=fig.add_subplot(gs[0,1])
    axC=fig.add_subplot(gs[1,0])
    axD=fig.add_subplot(gs[1,1])

    # A transformed x LOPO: delta relative to transformed OU-only
    txo=model_order_subset(tx,["X_B","X_E","X_0","X_C"])
    baseline=float(txo.loc[txo["model"].eq("X_0"),"total_elpd"].iloc[0])
    txo["delta_elpd_vs_X0"]=txo["total_elpd"]-baseline
    y=np.arange(len(txo))
    axA.barh(y,txo["delta_elpd_vs_X0"])
    axA.axvline(0,linestyle="--",linewidth=1)
    axA.set_yticks(y)
    axA.set_yticklabels([MODEL_LABELS.get(m,m) for m in txo["model"]])
    axA.invert_yaxis()
    axA.set_xlabel("Δ participant-held-out ELPD vs transformed OU only")
    axA.set_title("Logit-transformed x sensitivity (ε = 0.05)")
    panel_label(axA,"A")

    # B ordinal n coefficients
    ncoef=sec[(sec["family"]=="ordinal_n") & (sec["parameter"]=="beta")].copy()
    ncoef=ncoef[ncoef["predictor"].isin(["B_z","Myeloid_z","Context_z"])].copy()
    ncoef["label"]=ncoef.apply(
        lambda r:f'{r["model"]}: {r["predictor"].replace("_z","")}',axis=1
    )
    horizontal_forest(axB,ncoef,ncoef["label"].tolist())
    axB.set_xlabel("Ordered-logistic coefficient (95% HDI)")
    axB.set_title("Secondary ordinal n endpoint")
    panel_label(axB,"B")

    # C prior sensitivity: B->mu and context->mu
    ps=prior[
        ((prior["model"]=="M_B")&(prior["parameter"]=="beta_mu")&(prior["predictor"]=="B_z")) |
        ((prior["model"]=="M_C")&(prior["parameter"]=="beta_mu")&(prior["predictor"]=="Context_z"))
    ].copy()
    regimes=["narrow","reference","wide"]
    for target,label in [(("M_B","B_z"),"B → μ"),(("M_C","Context_z"),"Context → μ")]:
        sub=ps[(ps["model"]==target[0])&(ps["predictor"]==target[1])].copy()
        sub["prior_regime"]=pd.Categorical(sub["prior_regime"],categories=regimes,ordered=True)
        sub=sub.sort_values("prior_regime")
        axC.plot(regimes,sub["mean"],marker="o",label=label)
        axC.fill_between(
            np.arange(len(regimes)),
            sub["hdi_2.5%"].to_numpy(),
            sub["hdi_97.5%"].to_numpy(),
            alpha=0.15
        )
    axC.axhline(0,linestyle="--",linewidth=1)
    axC.set_ylabel("Posterior coefficient")
    axC.set_xlabel("Prior regime")
    axC.set_title("Key ecological conclusions are prior-robust")
    axC.legend(frameon=False)
    panel_label(axC,"C")

    # D PPC calibration
    pp=model_order_subset(ppc,["M_0","M_B","M_E","M_C"])
    x=np.arange(len(pp))
    width=.22
    axD.bar(x-width,pp["coverage_50"],width,label="50% interval")
    axD.bar(x,pp["coverage_80"],width,label="80% interval")
    axD.bar(x+width,pp["coverage_95"],width,label="95% interval")
    axD.axhline(.50,linestyle=":",linewidth=.8)
    axD.axhline(.80,linestyle=":",linewidth=.8)
    axD.axhline(.95,linestyle=":",linewidth=.8)
    axD.set_xticks(x)
    axD.set_xticklabels([MODEL_LABELS.get(m,m) for m in pp["model"]],rotation=20,ha="right")
    axD.set_ylim(0,1.02)
    axD.set_ylabel("Observed posterior-predictive coverage")
    axD.set_title("Posterior-predictive calibration")
    axD.legend(frameon=False,ncol=3,loc="upper center")
    panel_label(axD,"D")

    fig.suptitle(
        "Figure 5. Robustness, calibration, and secondary endpoint analyses",
        fontsize=15,y=0.995
    )
    save_figure(fig,outdir/"Figure5_robustness_calibration_secondary_endpoints")
    plt.close(fig)


# ---------------------------------------------------------------------
# Supplementary figures
# ---------------------------------------------------------------------
     
def make_supp_s1(data,outdir):
    flow=data["flow"]; ps=data["x_participant"]
    fig,axes=plt.subplots(1,2,figsize=(13,5.2))
    ax=axes[0]
    y=np.arange(len(flow))
    ax.barh(y,flow["n"])
    ax.set_yticks(y); ax.set_yticklabels(flow["stage"])
    ax.invert_yaxis(); ax.set_xlabel("Participants")
    ax.set_title("Participant flow")
    panel_label(ax,"A")

    ax=axes[1]
    ax.scatter(ps["n_unique_times"],ps["followup_years"],s=35,alpha=.8)
    ax.set_xlabel("Unique x time points")
    ax.set_ylabel("Follow-up (years)")
    ax.set_title("Longitudinal sampling structure")
    panel_label(ax,"B")
    fig.suptitle(
        "Supplementary Figure S1. Cohort flow and longitudinal support",
        fontsize=14,
        y=1.04
    )
    
    fig.subplots_adjust(top=0.82)
    save_figure(fig,outdir/"SuppFigS1_participant_flow_sampling")
    plt.close(fig)


def make_supp_s2(data,outdir):
    raw=data["raw_scan"]; clr=data["clr_scan"]
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.2))
    fig.subplots_adjust(wspace=0.42)
    for ax,df,title in [(axes[0],raw,"Raw composition"),(axes[1],clr,"CLR representation")]:
        ax.plot(df["k"],df["silhouette"],marker="o",label="Silhouette")
        ax2=ax.twinx()
        ax2.plot(df["k"],df["min_cluster_n"],marker="s",label="Minimum cluster n")
        ax.set_xlabel("k"); ax.set_ylabel("Silhouette")
        ax2.set_ylabel("Minimum cluster size")
        ax.set_title(title)
    panel_label(axes[0],"A"); panel_label(axes[1],"B")
    fig.suptitle(
        "Supplementary Figure S2. Cluster-resolution sensitivity",
        fontsize=14,
        y=1.04
    )
             
    fig.subplots_adjust(top=0.82)
    save_figure(fig,outdir/"SuppFigS2_raw_clr_resolution_sensitivity")
    plt.close(fig)


def make_supp_s3(data,outdir):
    df=data["prior_coef"].copy()
    targets=df[
        ((df["model"]=="M_B")&(df["predictor"]=="B_z")) |
        ((df["model"]=="M_E")&(df["predictor"].isin(["B_z","Myeloid_z"]))) |
        ((df["model"]=="M_C")&(df["predictor"]=="Context_z"))
    ].copy()
    labels=[]
    for _,r in targets.iterrows():
        labels.append(f'{r["model"]} {r["parameter"].replace("beta_","")} {r["predictor"].replace("_z","")} [{r["prior_regime"]}]')
    fig,ax=plt.subplots(figsize=(10,8))
    horizontal_forest(ax,targets,labels)
    ax.set_xlabel("Posterior coefficient (95% HDI)")
    ax.set_title(
        "Supplementary Figure S3. Prior sensitivity of ecological coefficients",
        fontsize=14,
        y=1.04
    )
             
    fig.subplots_adjust(top=0.82)
    save_figure(fig,outdir/"SuppFigS3_prior_sensitivity_coefficients")
    plt.close(fig)


def make_supp_s4(data,outdir):
    rec=data["recovery"].copy()
    me=rec[(rec["model"]=="M_E") & (rec["scenario"]=="moderate_directional")].copy()
    params=[
        "beta_mu_B_z","beta_mu_Myeloid_z",
        "beta_theta_B_z","beta_theta_Myeloid_z"
    ]
    me=me[me["parameter"].isin(params)].copy()
    x=np.arange(len(me))
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.errorbar(
        me["true_value"],me["mean_estimate"],
        yerr=me["rmse"],fmt="o",capsize=3
    )
    lim=[
        min(me["true_value"].min(),me["mean_estimate"].min())-.08,
        max(me["true_value"].max(),me["mean_estimate"].max())+.08
    ]
    ax.plot(lim,lim,linestyle="--",linewidth=1)
    for _,r in me.iterrows():
        ax.text(r["true_value"],r["mean_estimate"],
                " "+r["parameter"].replace("beta_",""),va="center",fontsize=9)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True parameter")
    ax.set_ylabel("Mean recovered estimate (error bars = RMSE)")
    ax.set_title(
        "Supplementary Figure S4. Simulation-based parameter recovery",
        fontsize=14,
        y=1.04
    )
             
    fig.subplots_adjust(top=0.82)        
    save_figure(fig,outdir/"SuppFigS4_parameter_recovery")
    plt.close(fig)


def make_supp_s5(data,outdir):
    df=data["ppc_context"].copy()
    models=["M_0","M_B","M_E","M_C"]
    contexts=["TNK-dominant","B/Myeloid-shifted"]
    fig,ax=plt.subplots(figsize=(10,5.5))
    width=.35; x=np.arange(len(models))
    for j,ctx in enumerate(contexts):
        vals=[]
        for m in models:
            z=df[(df["model"]==m)&(df["ecological_context_k2"]==ctx)]
            vals.append(float(z["coverage_95"].iloc[0]) if len(z) else np.nan)
        ax.bar(x+(j-.5)*width,vals,width,label=ctx,color=CONTEXT_COLORS[ctx])
    ax.axhline(.95,linestyle="--",linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(0,1.02)
    ax.set_ylabel("95% posterior-predictive coverage")
    ax.legend(frameon=False)
    ax.set_title(
        "Supplementary Figure S5. Calibration by ecological context",
        fontsize=14,
        y=1.04
    )
             
    fig.subplots_adjust(top=0.82)
    save_figure(fig,outdir/"SuppFigS5_ppc_by_context")
    plt.close(fig)


def make_supp_s6(data,outdir):
    pareto=data["pareto"].copy()
    sec=data["secondary_coef"].copy()
    fig,axes=plt.subplots(1,2,figsize=(13,5.2))

    # influential transitions
    top=pareto.sort_values("pareto_k",ascending=False).head(12)
    labels=[f'{r["Patient_ID"]}: {r["x_prev"]:.2f}→{r["x"]:.2f}, Δt={r["dt"]:.3f}' for _,r in top.iterrows()]
    axes[0].barh(np.arange(len(top)),top["pareto_k"])
    axes[0].set_yticks(np.arange(len(top))); axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].axvline(.7,linestyle="--",linewidth=1)
    axes[0].set_xlabel("Pareto k")
    axes[0].set_title("Influential transition audit")
    panel_label(axes[0],"A")

    # transformed x coefficient stability across epsilon
    bx=sec[
        (sec["family"]=="transformed_x")&
        (sec["model"].str.startswith("X_B"))&
        (sec["parameter"]=="beta_mu")&
        (sec["predictor"]=="B_z")
    ].copy().sort_values("epsilon")
    axes[1].errorbar(
        bx["epsilon"],bx["mean"],
        yerr=np.vstack([bx["mean"]-bx["hdi_2.5%"],bx["hdi_97.5%"]-bx["mean"]]),
        fmt="o-",capsize=3
    )
    axes[1].axhline(0,linestyle="--",linewidth=1)
    axes[1].set_xlabel("Clipping ε")
    axes[1].set_ylabel("B → μ coefficient")
    axes[1].set_title("Transformed-x robustness")
    panel_label(axes[1],"B")

    fig.suptitle(
        "Supplementary Figure S6. Influential observations and transformed-x sensitivity",
        fontsize=14,
        y=1.04
    )
             
    fig.subplots_adjust(top=0.82)
    save_figure(fig,outdir/"SuppFigS6_influential_transitions_transformed_x")
    plt.close(fig)


# ---------------------------------------------------------------------
# Source-table exports
# ---------------------------------------------------------------------

def export_source_tables(data, outdir: Path):
    outdir.mkdir(parents=True,exist_ok=True)
    mapping = {
        "Figure2_primary_composition.csv": data["primary_comp"],
        "Figure2_BALL_composition.csv": data["ball_comp"],
        "Figure3_raw_k_scan.csv": data["raw_scan"],
        "Figure3_bootstrap_raw.csv": data["boot_raw"],
        "Figure3_BALL_bootstrap_raw.csv": data["ball_boot_raw"],
        "Figure3_raw_clr_concordance.csv": data["concordance"],
        "Figure4_coefficients.csv": data["coef"],
        "Figure4_LOPO_summary.csv": data["lopo_summary"],
        "Figure4_LOPO_participants.csv": data["lopo_part"],
        "Figure5_transformed_x_LOPO.csv": data["tx_lopo_summary"],
        "Figure5_secondary_coefficients.csv": data["secondary_coef"],
        "Figure5_prior_coefficients.csv": data["prior_coef"],
        "Figure5_PPC_summary.csv": data["ppc"],
        "Supp_parameter_recovery.csv": data["recovery"],
    }
    for fn,df in mapping.items():
        df.to_csv(outdir/fn,index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/TME_OU_Branching")
    args=ap.parse_args()

    root=Path(args.root)
    outroot=root/"revision_final_figures"
    mainout=outroot/"main"
    suppout=outroot/"supplementary"
    srcout=outroot/"figure_source_tables"
    mainout.mkdir(parents=True,exist_ok=True)
    suppout.mkdir(parents=True,exist_ok=True)

    data=load_all(root)

    make_figure2(data,mainout)
    make_figure3(data,mainout)
    make_figure4(data,mainout)
    make_figure5(data,mainout)

    make_supp_s1(data,suppout)
    make_supp_s2(data,suppout)
    make_supp_s3(data,suppout)
    make_supp_s4(data,suppout)
    make_supp_s5(data,suppout)
    make_supp_s6(data,suppout)

    export_source_tables(data,srcout)

    print("\n[OK] All Waves 1-3 revision figures regenerated.")
    print("Main figures:",mainout)
    print("Supplementary figures:",suppout)
    print("Source tables:",srcout)

if __name__=="__main__":
    main()
