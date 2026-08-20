from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import arviz as az

MODELS=["M_0","M_B","M_E","M_C"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/TME_OU_Branching")
    args=ap.parse_args()
    root=Path(args.root)
    trans=pd.read_csv(root/"revision_wave2_inputs"/"wave2_x_transition_table.csv").reset_index(drop=True)
    out=root/"revision_wave3_ppc"
    out.mkdir(parents=True,exist_ok=True)

    all_rows=[]
    for m in MODELS:
        path=root/"revision_wave2_models"/"idata"/f"{m}_ou_x_idata_ppc.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        idata=az.from_netcdf(path)
        yrep=idata.posterior_predictive["y_obs"].stack(sample=("chain","draw")).transpose("obs","sample").values
        obs=trans["x"].to_numpy(float)

        med=np.median(yrep,axis=1)
        lo50=np.quantile(yrep,.25,axis=1); hi50=np.quantile(yrep,.75,axis=1)
        lo80=np.quantile(yrep,.10,axis=1); hi80=np.quantile(yrep,.90,axis=1)
        lo95=np.quantile(yrep,.025,axis=1); hi95=np.quantile(yrep,.975,axis=1)

        df=trans[[
            "Patient_ID","participant_id","ecological_context_k2",
            "time","x_prev","x","dt"
        ]].copy()
        df["model"]=m
        df["ppc_median"]=med
        df["residual"]=obs-med
        df["abs_error"]=np.abs(obs-med)
        df["sq_error"]=(obs-med)**2
        df["covered_50"]=(obs>=lo50)&(obs<=hi50)
        df["covered_80"]=(obs>=lo80)&(obs<=hi80)
        df["covered_95"]=(obs>=lo95)&(obs<=hi95)
        df["pit"]=(yrep<=obs[:,None]).mean(axis=1)
        all_rows.append(df)

    cal=pd.concat(all_rows,ignore_index=True)
    cal.to_csv(out/"wave3_ppc_transition_calibration.csv",index=False)

    summary=(cal.groupby("model")
             .agg(
                 n_transitions=("x","size"),
                 mae=("abs_error","mean"),
                 rmse=("sq_error",lambda x:float(np.sqrt(np.mean(x)))),
                 mean_residual=("residual","mean"),
                 coverage_50=("covered_50","mean"),
                 coverage_80=("covered_80","mean"),
                 coverage_95=("covered_95","mean"),
                 pit_mean=("pit","mean"),
                 pit_sd=("pit","std"),
             ).reset_index())
    summary.to_csv(out/"wave3_ppc_model_summary.csv",index=False)

    by_context=(cal.groupby(["model","ecological_context_k2"])
                .agg(
                    n_transitions=("x","size"),
                    mae=("abs_error","mean"),
                    rmse=("sq_error",lambda x:float(np.sqrt(np.mean(x)))),
                    coverage_95=("covered_95","mean"),
                    mean_residual=("residual","mean"),
                ).reset_index())
    by_context.to_csv(out/"wave3_ppc_by_context.csv",index=False)

    by_patient=(cal.groupby(["model","Patient_ID","participant_id"])
                .agg(
                    n_transitions=("x","size"),
                    mae=("abs_error","mean"),
                    rmse=("sq_error",lambda x:float(np.sqrt(np.mean(x)))),
                    coverage_95=("covered_95","mean"),
                    mean_residual=("residual","mean"),
                ).reset_index())
    by_patient.to_csv(out/"wave3_ppc_by_patient.csv",index=False)

    print("\n[OK] PPC calibration audit complete.")
    print(summary.to_string(index=False))
    print("\nBy context:")
    print(by_context.to_string(index=False))

if __name__=="__main__":
    main()
