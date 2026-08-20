from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.special import logsumexp

MODEL_SPECS = {
    "M_0": [],
    "M_B": ["B_z"],
    "M_M": ["Myeloid_z"],
    "M_E": ["B_z","Myeloid_z"],
    "M_C": ["Context_z"],
    "M_EC": ["B_z","Myeloid_z","Context_z"],
    "M_S": ["ShuffledContext_z"],
}

def ou_mean_var_np(y_prev, mu, theta, sigma, dt):
    decay = np.exp(-theta * dt)
    mean = mu + (y_prev - mu) * decay
    var = (sigma**2)/(2.0*theta) * (1.0 - np.exp(-2.0*theta*dt))
    return mean, np.maximum(var, 1e-12)

def ou_mean_var_pt(y_prev, mu, theta, sigma, dt):
    decay = pt.exp(-theta * dt)
    mean = mu + (y_prev - mu)*decay
    var = (sigma**2)/(2.0*theta) * (1.0 - pt.exp(-2.0*theta*dt))
    return mean, pt.clip(var, 1e-12, np.inf)

def normal_logpdf(y, mu, var):
    return -0.5*(np.log(2*np.pi*var) + (y-mu)**2/var)

def safe_scale(train_vals, test_vals, name):
    train_vals = np.asarray(train_vals, float)
    test_vals = np.asarray(test_vals, float)
    m = float(np.mean(train_vals))
    s = float(np.std(train_vals, ddof=0))
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"{name}: zero/non-finite training SD")
    return (train_vals-m)/s, (test_vals-m)/s, m, s

def build_global_shuffle(design, seed):
    d = design.sort_values("Patient_ID").copy()
    rng = np.random.default_rng(seed)
    vals = d["Context_numeric"].to_numpy(float)
    d["ShuffledContext_numeric"] = rng.permutation(vals)
    return dict(zip(d["Patient_ID"], d["ShuffledContext_numeric"]))

def participant_design(trans):
    needed = [
        "Patient_ID","participant_id","project","diagnosis",
        "cluster","TNK","B","Myeloid"
    ]
    d = trans[needed].drop_duplicates("Patient_ID").sort_values("Patient_ID").reset_index(drop=True)
    d["Context_numeric"] = d["cluster"].map({"C1":0.0,"C2":1.0})
    if d["Context_numeric"].isna().any():
        raise ValueError("Unexpected context label")
    return d

def prepare_fold_design(full_design, heldout_id, shuffle_map):
    tr = full_design.loc[full_design["Patient_ID"] != heldout_id].copy()
    te = full_design.loc[full_design["Patient_ID"] == heldout_id].copy()
    if len(te) != 1:
        raise ValueError(f"Held-out design row count !=1 for {heldout_id}")

    # Continuous ecology
    tr["B_z"], te_B, bmean, bsd = safe_scale(tr["B"], te["B"], "B")
    te["B_z"] = te_B

    tr["Myeloid_z"], te_M, mmean, msd = safe_scale(tr["Myeloid"], te["Myeloid"], "Myeloid")
    te["Myeloid_z"] = te_M

    # True context scaling from TRAINING distribution only.
    tr["Context_z"], te_C, cmean, csd = safe_scale(
        tr["Context_numeric"], te["Context_numeric"], "Context"
    )
    te["Context_z"] = te_C

    # Fixed shuffled mapping, but fold-specific scaling.
    tr["ShuffledContext_numeric"] = tr["Patient_ID"].map(shuffle_map).astype(float)
    te["ShuffledContext_numeric"] = te["Patient_ID"].map(shuffle_map).astype(float)
    tr["ShuffledContext_z"], te_S, smean, ssd = safe_scale(
        tr["ShuffledContext_numeric"], te["ShuffledContext_numeric"], "ShuffledContext"
    )
    te["ShuffledContext_z"] = te_S

    scales = {
        "B_mean":bmean, "B_sd":bsd,
        "Myeloid_mean":mmean, "Myeloid_sd":msd,
        "Context_mean":cmean, "Context_sd":csd,
        "ShuffledContext_mean":smean, "ShuffledContext_sd":ssd,
    }
    return tr.reset_index(drop=True), te.reset_index(drop=True), scales

def build_training_model(model_name, predictors, train_trans, train_design):
    pids = train_design["Patient_ID"].tolist()
    pmap = {p:i for i,p in enumerate(pids)}
    pat_idx = train_trans["Patient_ID"].map(pmap).to_numpy("int64")
    y = train_trans["x"].to_numpy(float)
    yp = train_trans["x_prev"].to_numpy(float)
    dt = train_trans["dt"].to_numpy(float)

    coords={"patient":pids,"obs":np.arange(len(train_trans))}
    X=None
    if predictors:
        X=train_design[predictors].to_numpy(float)
        coords["predictor"]=predictors

    with pm.Model(coords=coords) as model:
        mu_0=pm.Normal("mu_0",0,1)
        theta_0=pm.Normal("theta_0",0,1)
        sigma_proc=pm.HalfNormal("sigma_proc",1)

        if predictors:
            beta_mu=pm.Normal("beta_mu",0,0.35,dims="predictor")
            beta_theta=pm.Normal("beta_theta",0,0.35,dims="predictor")
            mu_pat=pm.Deterministic("mu_pat",mu_0+pt.dot(X,beta_mu),dims="patient")
            log_theta_pat=pm.Deterministic("log_theta_pat",theta_0+pt.dot(X,beta_theta),dims="patient")
        else:
            mu_pat=pm.Deterministic("mu_pat",pt.repeat(mu_0,len(pids)),dims="patient")
            log_theta_pat=pm.Deterministic("log_theta_pat",pt.repeat(theta_0,len(pids)),dims="patient")

        theta_pat=pm.Deterministic("theta_pat",pt.exp(log_theta_pat)+1e-6,dims="patient")
        mean,var=ou_mean_var_pt(yp,mu_pat[pat_idx],theta_pat[pat_idx],sigma_proc,dt)
        pm.Normal("y_obs",mean,pt.sqrt(var),observed=y,dims="obs")
    return model

def heldout_lpd(idata, model_name, predictors, test_design, test_trans):
    mu0 = idata.posterior["mu_0"].values.reshape(-1)
    th0 = idata.posterior["theta_0"].values.reshape(-1)
    sig = idata.posterior["sigma_proc"].values.reshape(-1)
    ndraw = len(mu0)

    if predictors:
        bm = idata.posterior["beta_mu"].values.reshape(-1, len(predictors))
        bt = idata.posterior["beta_theta"].values.reshape(-1, len(predictors))
        x = test_design[predictors].iloc[0].to_numpy(float)
        mu = mu0 + bm @ x
        theta = np.exp(th0 + bt @ x) + 1e-6
    else:
        mu = mu0
        theta = np.exp(th0) + 1e-6

    pointwise = []
    for _, r in test_trans.iterrows():
        mean,var = ou_mean_var_np(
            float(r["x_prev"]), mu, theta, sig, float(r["dt"])
        )
        ll = normal_logpdf(float(r["x"]), mean, var)
        lpd = float(logsumexp(ll) - np.log(ndraw))
        pointwise.append(lpd)

    return np.asarray(pointwise), float(np.sum(pointwise))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/TME_OU_Branching")
    ap.add_argument("--input",default="revision_wave2_inputs/wave2_x_transition_table.csv")
    ap.add_argument("--models",nargs="+",default=["M_0","M_B","M_E","M_C","M_EC","M_S"],
                    choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--draws",type=int,default=1000)
    ap.add_argument("--tune",type=int,default=1500)
    ap.add_argument("--chains",type=int,default=2)
    ap.add_argument("--cores",type=int,default=2)
    ap.add_argument("--target-accept",type=float,default=0.99)
    ap.add_argument("--seed",type=int,default=20260819)
    ap.add_argument("--shuffle-seed",type=int,default=20260820)
    ap.add_argument("--only-patients",nargs="*",default=None)
    args=ap.parse_args()

    root=Path(args.root)
    inp=root/args.input
    out=root/"revision_wave2_lopo"
    fold_dir=out/"fold_results"
    fold_dir.mkdir(parents=True,exist_ok=True)

    trans=pd.read_csv(inp)
    design=participant_design(trans)
    shuffle_map=build_global_shuffle(design,args.shuffle_seed)
    pids=design["Patient_ID"].tolist()
    if args.only_patients:
        pids=[p for p in pids if p in set(args.only_patients)]

    config={
        "models":args.models,"n_folds":len(pids),"draws":args.draws,"tune":args.tune,
        "chains":args.chains,"cores":args.cores,"target_accept":args.target_accept,
        "seed":args.seed,"shuffle_seed":args.shuffle_seed,
        "participant_level_scaling":"training fold only",
    }
    out.mkdir(parents=True,exist_ok=True)
    with open(out/"wave2_lopo_run_config.json","w") as f:
        json.dump(config,f,indent=2)

    results=[]
    diags=[]

    for mi,model_name in enumerate(args.models):
        predictors=MODEL_SPECS[model_name]
        print("\n"+"#"*80)
        print(f"[MODEL] {model_name} predictors={predictors or 'None'}")
        print("#"*80)

        for fi,pid in enumerate(pids):
            print(f"[FOLD {fi+1}/{len(pids)}] Hold out {pid}")

            train_trans=trans.loc[trans["Patient_ID"]!=pid].copy()
            test_trans=trans.loc[trans["Patient_ID"]==pid].copy()
            train_design,test_design,scales=prepare_fold_design(design,pid,shuffle_map)

            model=build_training_model(
                model_name,predictors,train_trans,train_design
            )

            fold_seed=args.seed + mi*10000 + fi
            with model:
                idata=pm.sample(
                    draws=args.draws,tune=args.tune,chains=args.chains,cores=args.cores,
                    target_accept=args.target_accept,random_seed=fold_seed,
                    progressbar=False,return_inferencedata=True,
                )

            # MCMC diagnostics
            div=int(idata.sample_stats["diverging"].sum().values)
            vars_=["mu_0","theta_0","sigma_proc"]
            if predictors: vars_ += ["beta_mu","beta_theta"]
            summ=az.summary(idata,var_names=vars_,round_to=None)
            max_rhat=float(np.nanmax(summ["r_hat"])) if "r_hat" in summ else np.nan
            min_ess=float(np.nanmin(summ["ess_bulk"])) if "ess_bulk" in summ else np.nan

            pointwise,total=heldout_lpd(
                idata,model_name,predictors,test_design,test_trans
            )

            row={
                "model":model_name,
                "Patient_ID":pid,
                "participant_id":test_design.iloc[0]["participant_id"],
                "ecological_context_k2":test_trans.iloc[0]["ecological_context_k2"],
                "n_test_transitions":len(test_trans),
                "heldout_elpd":total,
                "mean_lpd_per_transition":float(np.mean(pointwise)),
                "divergences":div,
                "max_rhat":max_rhat,
                "min_ess_bulk":min_ess,
                **scales,
            }
            results.append(row)
            diags.append({
                "model":model_name,"Patient_ID":pid,"divergences":div,
                "max_rhat":max_rhat,"min_ess_bulk":min_ess
            })

            pd.DataFrame([row]).to_csv(
                fold_dir/f"{model_name}_{pid}_fold_summary.csv",index=False
            )

    res=pd.DataFrame(results)
    res.to_csv(out/"wave2_lopo_participant_results.csv",index=False)
    pd.DataFrame(diags).to_csv(out/"wave2_lopo_diagnostics.csv",index=False)

    # Model summaries at participant unit.
    summary=(
        res.groupby("model")
        .agg(
            n_participants=("Patient_ID","nunique"),
            total_elpd=("heldout_elpd","sum"),
            mean_participant_elpd=("heldout_elpd","mean"),
            se_participant_elpd=("heldout_elpd",lambda x: float(np.std(x,ddof=1)/np.sqrt(len(x)))),
            mean_lpd_per_transition=("mean_lpd_per_transition","mean"),
            total_test_transitions=("n_test_transitions","sum"),
            n_folds_with_divergence=("divergences",lambda x:int(np.sum(np.asarray(x)>0))),
            max_fold_rhat=("max_rhat","max"),
            min_fold_ess_bulk=("min_ess_bulk","min"),
        )
        .reset_index()
        .sort_values("total_elpd",ascending=False)
    )
    summary.to_csv(out/"wave2_lopo_model_summary.csv",index=False)

    # Paired participant-level model differences.
    wide=res.pivot(index="Patient_ID",columns="model",values="heldout_elpd")
    pairs=[]
    models=list(wide.columns)
    for i,a in enumerate(models):
        for b in models[i+1:]:
            d=(wide[a]-wide[b]).dropna()
            pairs.append({
                "model_A":a,"model_B":b,
                "mean_delta_elpd_A_minus_B":float(d.mean()),
                "total_delta_elpd_A_minus_B":float(d.sum()),
                "se_delta":float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else np.nan,
                "n_participants":len(d),
                "n_A_better":int(np.sum(d>0)),
                "n_B_better":int(np.sum(d<0)),
                "n_ties":int(np.sum(d==0)),
            })
    pd.DataFrame(pairs).to_csv(out/"wave2_lopo_pairwise_differences.csv",index=False)

    print("\n[OK] True LOPO validation complete.")
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
