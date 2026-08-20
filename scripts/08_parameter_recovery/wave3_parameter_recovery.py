from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

SCENARIOS = {
    "null": {
        "mu_0":0.25,"theta_0":0.0,"sigma_proc":0.30,
        "beta_mu_B":0.0,"beta_mu_Myeloid":0.0,
        "beta_theta_B":0.0,"beta_theta_Myeloid":0.0,
    },
    "moderate_directional": {
        "mu_0":0.25,"theta_0":0.0,"sigma_proc":0.30,
        "beta_mu_B":-0.07,"beta_mu_Myeloid":0.03,
        "beta_theta_B":0.20,"beta_theta_Myeloid":-0.20,
    }
}

def zscore(s):
    x=np.asarray(s,float); m=x.mean(); sd=x.std(ddof=0)
    return (x-m)/sd

def design_from_trans(trans):
    d=trans[["Patient_ID","B","Myeloid"]].drop_duplicates("Patient_ID").sort_values("Patient_ID").reset_index(drop=True)
    d["B_z"]=zscore(d["B"])
    d["Myeloid_z"]=zscore(d["Myeloid"])
    return d

def simulate_recursive(template, design, model_name, pars, rng):
    dmap=design.set_index("Patient_ID")
    out=[]
    for pid,g in template.sort_values(["Patient_ID","time"]).groupby("Patient_ID"):
        g=g.sort_values("time")
        if model_name=="M_0":
            mu=pars["mu_0"]
            theta=np.exp(pars["theta_0"])
        else:
            b=float(dmap.loc[pid,"B_z"]); m=float(dmap.loc[pid,"Myeloid_z"])
            mu=pars["mu_0"]+pars["beta_mu_B"]*b+pars["beta_mu_Myeloid"]*m
            theta=np.exp(pars["theta_0"]+pars["beta_theta_B"]*b+pars["beta_theta_Myeloid"]*m)
        yprev=float(g.iloc[0]["x_prev"])
        for _,r in g.iterrows():
            dt=float(r["dt"])
            decay=np.exp(-theta*dt)
            mean=mu+(yprev-mu)*decay
            var=(pars["sigma_proc"]**2)/(2*theta)*(1-np.exp(-2*theta*dt))
            y=float(rng.normal(mean,np.sqrt(max(var,1e-12))))
            rr=r.copy()
            rr["x_prev_sim"]=yprev; rr["x_sim"]=y
            out.append(rr)
            yprev=y
    return pd.DataFrame(out)

def ou_mv(yp,mu,theta,sig,dt):
    decay=pt.exp(-theta*dt)
    mean=mu+(yp-mu)*decay
    var=(sig**2)/(2*theta)*(1-pt.exp(-2*theta*dt))
    return mean,pt.clip(var,1e-12,np.inf)

def fit_model(sim,design,model_name,draws,tune,chains,cores,target_accept,seed):
    pids=design["Patient_ID"].tolist()
    pmap={p:i for i,p in enumerate(pids)}
    idx=sim["Patient_ID"].map(pmap).to_numpy("int64")
    yp=sim["x_prev_sim"].to_numpy(float); y=sim["x_sim"].to_numpy(float); dt=sim["dt"].to_numpy(float)
    coords={"patient":pids,"obs":np.arange(len(sim))}
    X=None
    if model_name=="M_E":
        X=design[["B_z","Myeloid_z"]].to_numpy(float)
        coords["predictor"]=["B_z","Myeloid_z"]
    with pm.Model(coords=coords) as model:
        mu0=pm.Normal("mu_0",0,1)
        th0=pm.Normal("theta_0",0,1)
        sig=pm.HalfNormal("sigma_proc",1)
        if model_name=="M_E":
            bm=pm.Normal("beta_mu",0,.35,dims="predictor")
            bt=pm.Normal("beta_theta",0,.35,dims="predictor")
            mu=pm.Deterministic("mu_pat",mu0+pt.dot(X,bm),dims="patient")
            lth=pm.Deterministic("log_theta_pat",th0+pt.dot(X,bt),dims="patient")
        else:
            mu=pm.Deterministic("mu_pat",pt.repeat(mu0,len(pids)),dims="patient")
            lth=pm.Deterministic("log_theta_pat",pt.repeat(th0,len(pids)),dims="patient")
        th=pm.Deterministic("theta_pat",pt.exp(lth)+1e-6,dims="patient")
        mean,var=ou_mv(yp,mu[idx],th[idx],sig,dt)
        pm.Normal("y_obs",mean,pt.sqrt(var),observed=y,dims="obs")
        idata=pm.sample(
            draws=draws,tune=tune,chains=chains,cores=cores,target_accept=target_accept,
            random_seed=seed,progressbar=False,return_inferencedata=True
        )
    return idata

def extract(idata,model_name):
    est={}
    for p in ["mu_0","theta_0","sigma_proc"]:
        vals=idata.posterior[p].values.reshape(-1)
        h=az.hdi(vals,hdi_prob=.95)
        est[p]=(float(vals.mean()),float(h[0]),float(h[1]))
    if model_name=="M_E":
        for pname in ["beta_mu","beta_theta"]:
            for pred in ["B_z","Myeloid_z"]:
                vals=idata.posterior[pname].sel(predictor=pred).values.reshape(-1)
                h=az.hdi(vals,hdi_prob=.95)
                est[f"{pname}_{pred}"]=(float(vals.mean()),float(h[0]),float(h[1]))
    return est

def true_value(key,pars):
    mp={
        "mu_0":"mu_0","theta_0":"theta_0","sigma_proc":"sigma_proc",
        "beta_mu_B_z":"beta_mu_B","beta_mu_Myeloid_z":"beta_mu_Myeloid",
        "beta_theta_B_z":"beta_theta_B","beta_theta_Myeloid_z":"beta_theta_Myeloid",
    }
    return pars[mp[key]]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/TME_OU_Branching")
    ap.add_argument("--replicates",type=int,default=30)
    ap.add_argument("--draws",type=int,default=800)
    ap.add_argument("--tune",type=int,default=1200)
    ap.add_argument("--chains",type=int,default=2)
    ap.add_argument("--cores",type=int,default=2)
    ap.add_argument("--target-accept",type=float,default=.97)
    ap.add_argument("--seed",type=int,default=20260819)
    args=ap.parse_args()

    root=Path(args.root)
    template=pd.read_csv(root/"revision_wave2_inputs"/"wave2_x_transition_table.csv")
    design=design_from_trans(template)
    out=root/"revision_wave3_parameter_recovery"
    out.mkdir(parents=True,exist_ok=True)

    rows=[]
    models=["M_0","M_E"]
    for si,(sname,pars) in enumerate(SCENARIOS.items()):
        for model_name in models:
            if model_name=="M_0" and sname=="moderate_directional":
                # still simulate M0 using only intercept parameters from this scenario
                pass
            for rep in range(args.replicates):
                seed=args.seed+si*100000+(0 if model_name=="M_0" else 50000)+rep
                rng=np.random.default_rng(seed)
                sim=simulate_recursive(template,design,model_name,pars,rng)
                idata=fit_model(sim,design,model_name,args.draws,args.tune,args.chains,args.cores,args.target_accept,seed)
                est=extract(idata,model_name)
                div=int(idata.sample_stats["diverging"].sum().values)
                for key,(mean,lo,hi) in est.items():
                    tv=true_value(key,pars)
                    rows.append({
                        "scenario":sname,"model":model_name,"replicate":rep,
                        "parameter":key,"true_value":tv,
                        "posterior_mean":mean,"hdi_low":lo,"hdi_high":hi,
                        "error":mean-tv,"squared_error":(mean-tv)**2,
                        "covered_95":int(lo<=tv<=hi),"divergences":div
                    })
                print(f"[RECOVERY] {sname} {model_name} rep {rep+1}/{args.replicates}")

    res=pd.DataFrame(rows)
    res.to_csv(out/"replicate_results.csv",index=False)
    summ=(res.groupby(["scenario","model","parameter"])
          .agg(
              true_value=("true_value","first"),
              mean_estimate=("posterior_mean","mean"),
              bias=("error","mean"),
              rmse=("squared_error",lambda x:float(np.sqrt(np.mean(x)))),
              coverage_95=("covered_95","mean"),
              mean_divergences=("divergences","mean"),
              n_replicates=("replicate","nunique"),
          ).reset_index())
    summ.to_csv(out/"recovery_summary.csv",index=False)
    with open(out/"run_config.json","w") as f:
        json.dump({"scenarios":SCENARIOS,"replicates":args.replicates},f,indent=2)
    print("\n[OK] Parameter recovery complete.")
    print(summ.to_string(index=False))

if __name__=="__main__":
    main()
