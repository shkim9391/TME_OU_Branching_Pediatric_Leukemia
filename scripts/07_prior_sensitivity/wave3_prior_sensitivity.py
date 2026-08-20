from __future__ import annotations
import argparse, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

MODEL_SPECS = {
    "M_0": [],
    "M_B": ["B_z"],
    "M_E": ["B_z", "Myeloid_z"],
    "M_C": ["Context_z"],
}

PRIOR_REGIMES = {
    "narrow": {
        "mu0_sd": 0.5,
        "theta0_sd": 0.5,
        "sigma_proc_sd": 0.5,
        "beta_mu_sd": 0.20,
        "beta_theta_sd": 0.20,
    },
    "reference": {
        "mu0_sd": 1.0,
        "theta0_sd": 1.0,
        "sigma_proc_sd": 1.0,
        "beta_mu_sd": 0.35,
        "beta_theta_sd": 0.35,
    },
    "wide": {
        "mu0_sd": 2.0,
        "theta0_sd": 1.5,
        "sigma_proc_sd": 2.0,
        "beta_mu_sd": 0.70,
        "beta_theta_sd": 0.70,
    },
}

def zscore(s):
    x = pd.to_numeric(s, errors="raise").to_numpy(float)
    m = float(x.mean()); sd = float(x.std(ddof=0))
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError("Invalid predictor SD")
    return (x-m)/sd, m, sd

def build_design(trans):
    part = trans[
        ["Patient_ID","participant_id","cluster","B","Myeloid"]
    ].drop_duplicates("Patient_ID").sort_values("Patient_ID").reset_index(drop=True)
    part["Context_numeric"] = part["cluster"].map({"C1":0.0,"C2":1.0})
    part["B_z"],_,_ = zscore(part["B"])
    part["Myeloid_z"],_,_ = zscore(part["Myeloid"])
    part["Context_z"],_,_ = zscore(part["Context_numeric"])
    return part

def ou_mv(yp, mu, theta, sig, dt):
    decay = pt.exp(-theta*dt)
    mean = mu + (yp-mu)*decay
    var = (sig**2)/(2*theta)*(1-pt.exp(-2*theta*dt))
    return mean, pt.clip(var,1e-12,np.inf)

def build_model(trans, design, predictors, pri):
    pids = design["Patient_ID"].tolist()
    pmap = {p:i for i,p in enumerate(pids)}
    idx = trans["Patient_ID"].map(pmap).to_numpy("int64")
    y = trans["x"].to_numpy(float)
    yp = trans["x_prev"].to_numpy(float)
    dt = trans["dt"].to_numpy(float)
    coords={"patient":pids,"obs":np.arange(len(trans))}
    X=None
    if predictors:
        X=design[predictors].to_numpy(float)
        coords["predictor"]=predictors
    with pm.Model(coords=coords) as model:
        mu0=pm.Normal("mu_0",0,pri["mu0_sd"])
        th0=pm.Normal("theta_0",0,pri["theta0_sd"])
        sig=pm.HalfNormal("sigma_proc",pri["sigma_proc_sd"])
        if predictors:
            bm=pm.Normal("beta_mu",0,pri["beta_mu_sd"],dims="predictor")
            bt=pm.Normal("beta_theta",0,pri["beta_theta_sd"],dims="predictor")
            mu=pm.Deterministic("mu_pat",mu0+pt.dot(X,bm),dims="patient")
            lth=pm.Deterministic("log_theta_pat",th0+pt.dot(X,bt),dims="patient")
        else:
            mu=pm.Deterministic("mu_pat",pt.repeat(mu0,len(pids)),dims="patient")
            lth=pm.Deterministic("log_theta_pat",pt.repeat(th0,len(pids)),dims="patient")
        th=pm.Deterministic("theta_pat",pt.exp(lth)+1e-6,dims="patient")
        mean,var=ou_mv(yp,mu[idx],th[idx],sig,dt)
        pm.Normal("y_obs",mean,pt.sqrt(var),observed=y,dims="obs")
    return model

def atomic_save(idata,path):
    tmp=path.with_name(path.stem+"_tmp.nc")
    if tmp.exists(): tmp.unlink()
    az.to_netcdf(idata,tmp,engine="netcdf4")
    if path.exists(): path.unlink()
    tmp.replace(path)

def coef_rows(model_name, regime, idata):
    rows=[]
    for pname in ["beta_mu","beta_theta"]:
        if pname not in idata.posterior: continue
        da=idata.posterior[pname]
        for pred in da.coords["predictor"].values.tolist():
            vals=da.sel(predictor=pred).values.reshape(-1)
            h=az.hdi(vals,hdi_prob=.95)
            rows.append({
                "model":model_name,"prior_regime":regime,
                "parameter":pname,"predictor":str(pred),
                "mean":float(vals.mean()),"sd":float(vals.std(ddof=1)),
                "hdi_2.5%":float(h[0]),"hdi_97.5%":float(h[1]),
                "p_gt_0":float(np.mean(vals>0)),"p_lt_0":float(np.mean(vals<0))
            })
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/TME_OU_Branching")
    ap.add_argument("--draws",type=int,default=1500)
    ap.add_argument("--tune",type=int,default=2000)
    ap.add_argument("--chains",type=int,default=4)
    ap.add_argument("--cores",type=int,default=4)
    ap.add_argument("--target-accept",type=float,default=.99)
    ap.add_argument("--seed",type=int,default=20260819)
    args=ap.parse_args()

    root=Path(args.root)
    inp=root/"revision_wave2_inputs"/"wave2_x_transition_table.csv"
    trans=pd.read_csv(inp)
    design=build_design(trans)

    out=root/"revision_wave3_prior_sensitivity"
    idir=out/"idata"; ddir=out/"diagnostics"
    idir.mkdir(parents=True,exist_ok=True); ddir.mkdir(parents=True,exist_ok=True)

    rows=[]; coefs=[]
    for ri,(regime,pri) in enumerate(PRIOR_REGIMES.items()):
        for mi,(mname,preds) in enumerate(MODEL_SPECS.items()):
            print(f"\n[PRIOR {regime}] {mname} predictors={preds or 'None'}")
            model=build_model(trans,design,preds,pri)
            with model:
                idata=pm.sample(
                    draws=args.draws,tune=args.tune,chains=args.chains,cores=args.cores,
                    target_accept=args.target_accept,
                    random_seed=args.seed+ri*100+mi,
                    idata_kwargs={"log_likelihood":True},
                    return_inferencedata=True,
                )
                idata=pm.sample_posterior_predictive(
                    idata,var_names=["y_obs"],
                    random_seed=args.seed+10000+ri*100+mi,
                    extend_inferencedata=True
                )
            atomic_save(idata,idir/f"{mname}_{regime}_idata_ppc.nc")
            vars_=["mu_0","theta_0","sigma_proc"] + (["beta_mu","beta_theta"] if preds else [])
            summ=az.summary(idata,var_names=vars_,round_to=None)
            loo=az.loo(idata,pointwise=True)
            pk=np.asarray(loo.pareto_k.values,float)
            rows.append({
                "model":mname,"prior_regime":regime,
                "divergences":int(idata.sample_stats["diverging"].sum().values),
                "max_rhat":float(np.nanmax(summ["r_hat"])),
                "min_ess_bulk":float(np.nanmin(summ["ess_bulk"])),
                "elpd_loo":float(loo.elpd_loo),"p_loo":float(loo.p_loo),
                "max_pareto_k":float(pk.max()),
            })
            coefs += coef_rows(mname,regime,idata)
            del idata,model
            gc.collect()

    pd.DataFrame(rows).to_csv(out/"wave3_prior_sensitivity_summary.csv",index=False)
    pd.DataFrame(coefs).to_csv(out/"wave3_prior_coefficient_summary.csv",index=False)
    with open(out/"wave3_prior_run_config.json","w") as f:
        json.dump({"prior_regimes":PRIOR_REGIMES},f,indent=2)
    print("\n[OK] Prior sensitivity complete.")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":
    main()
