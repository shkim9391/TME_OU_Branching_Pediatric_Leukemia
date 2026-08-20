from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import arviz as az

MODELS = ["M_B", "M_M", "M_E", "M_C", "M_EC", "M_S"]

def flatten_var(idata, varname):
    da = idata.posterior[varname]
    if "predictor" in da.dims:
        out = {}
        for pred in da.coords["predictor"].values.tolist():
            vals = da.sel(predictor=pred).values.reshape(-1)
            out[str(pred)] = vals
        return out
    return {varname: da.values.reshape(-1)}

def summarize_draws(model, parameter, predictor, vals):
    vals = np.asarray(vals, dtype=float)
    hdi = az.hdi(vals, hdi_prob=0.95)
    return {
        "model": model,
        "parameter": parameter,
        "predictor": predictor,
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)),
        "median": float(np.median(vals)),
        "hdi_2.5%": float(hdi[0]),
        "hdi_97.5%": float(hdi[1]),
        "p_gt_0": float(np.mean(vals > 0)),
        "p_lt_0": float(np.mean(vals < 0)),
        "p_abs_gt_0.1": float(np.mean(np.abs(vals) > 0.1)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/TME_OU_Branching")
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()

    root = Path(args.root)
    idata_dir = root / "revision_wave2_models" / "idata"
    diag_dir = root / "revision_wave2_models" / "diagnostics"
    outdir = root / "revision_wave2_models" / "inspection"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODELS:
        path = idata_dir / f"{model}_ou_x_idata_ppc.nc"
        if not path.exists():
            print(f"[WARN] Missing {path}; skipping")
            continue
        idata = az.from_netcdf(path)
        for parameter in ["beta_mu", "beta_theta"]:
            if parameter not in idata.posterior:
                continue
            for pred, vals in flatten_var(idata, parameter).items():
                rows.append(summarize_draws(model, parameter, pred, vals))

    coef = pd.DataFrame(rows)
    coef.to_csv(outdir / "wave2_coefficient_posterior_summary.csv", index=False)

    # Compact directional probability table.
    prob = coef[
        ["model","parameter","predictor","mean","hdi_2.5%","hdi_97.5%","p_gt_0","p_lt_0"]
    ].copy()
    prob["dominant_direction_probability"] = prob[["p_gt_0","p_lt_0"]].max(axis=1)
    prob["direction"] = np.where(prob["p_gt_0"] >= prob["p_lt_0"], "positive", "negative")
    prob.to_csv(outdir / "wave2_coefficient_probability_summary.csv", index=False)

    pareto_path = diag_dir / "wave2_ablation_pareto_k.csv"
    if not pareto_path.exists():
        raise FileNotFoundError(pareto_path)

    pareto = pd.read_csv(pareto_path)
    pareto = pareto.sort_values(["pareto_k"], ascending=False).reset_index(drop=True)

    top = (
        pareto.groupby("model", group_keys=False)
        .head(args.top_n)
        .sort_values(["model","pareto_k"], ascending=[True,False])
    )
    top.to_csv(outdir / "wave2_top_pareto_transitions.csv", index=False)

    gt07 = pareto.loc[pareto["pareto_k"] > 0.7].copy()
    gt07.to_csv(outdir / "wave2_pareto_gt_07.csv", index=False)

    by_patient = (
        pareto.groupby(["model","Patient_ID","participant_id"], dropna=False)
        .agg(
            max_pareto_k=("pareto_k","max"),
            mean_pareto_k=("pareto_k","mean"),
            n_transitions=("pareto_k","size"),
            n_gt_05=("pareto_k", lambda x: int(np.sum(x > 0.5))),
            n_gt_07=("pareto_k", lambda x: int(np.sum(x > 0.7))),
            n_gt_10=("pareto_k", lambda x: int(np.sum(x > 1.0))),
        )
        .reset_index()
        .sort_values(["model","max_pareto_k"], ascending=[True,False])
    )
    by_patient.to_csv(outdir / "wave2_pareto_by_patient.csv", index=False)

    print("\n[OK] Coefficient + Pareto inspection complete.")
    print("Outputs:", outdir)

    if len(coef):
        print("\nCoefficient posterior summary:")
        print(coef.to_string(index=False))

    print("\nTransitions with Pareto-k > 0.7:")
    if len(gt07):
        show = [c for c in [
            "model","Patient_ID","participant_id","ecological_context_k2",
            "time","x_prev","x","dt","pareto_k"
        ] if c in gt07.columns]
        print(gt07[show].to_string(index=False))
    else:
        print("None")

if __name__ == "__main__":
    main()
