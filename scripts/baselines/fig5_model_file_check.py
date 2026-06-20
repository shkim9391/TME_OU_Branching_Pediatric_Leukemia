from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az


ROOT = Path("/TME_OU_Branching")
RESULTS_DIR = ROOT / "results"
OUTDIR = ROOT / "Figure_5"
OUTDIR.mkdir(exist_ok=True, parents=True)

OUT_CSV = OUTDIR / "Figure5_model_file_inventory.csv"

OBS_VAR = "y_obs"

MODEL_FILES = {
    "context_ou": RESULTS_DIR / "ou_revised_ecological_context_idata_ppc.nc",
    "ou_only": RESULTS_DIR / "ou_only_revised_context_idata.nc",
    "shuffled_context": RESULTS_DIR / "shuffled_context_revised_idata.nc",
    "random_walk": RESULTS_DIR / "random_walk_revised_idata.nc",
    "static_context": RESULTS_DIR / "static_context_revised_idata.nc",
}


def to_numpy(x):
    try:
        return x.values
    except Exception:
        return np.asarray(x)


def summarize_model(model_key, path):
    row = {
        "model_key": model_key,
        "path": str(path),
        "exists": path.exists(),
        "load_ok": False,
        "has_observed_y": False,
        "n_obs": np.nan,
        "has_log_likelihood": False,
        "log_likelihood_vars": "",
        "has_posterior_predictive": False,
        "posterior_predictive_vars": "",
        "has_yobs_ppc": False,
        "has_yrep_ppc": False,
        "same_n_as_context_ou": False,
        "same_y_as_context_ou": False,
        "error": "",
    }

    if not path.exists():
        row["error"] = "file not found"
        return row, None

    try:
        idata = az.from_netcdf(path)
        row["load_ok"] = True
    except Exception as e:
        row["error"] = f"load failed: {e}"
        return row, None

    y = None

    if hasattr(idata, "observed_data") and OBS_VAR in idata.observed_data:
        y = to_numpy(idata.observed_data[OBS_VAR]).ravel()
        row["has_observed_y"] = True
        row["n_obs"] = len(y)
    else:
        row["error"] += " missing observed_data/y_obs;"

    if hasattr(idata, "log_likelihood"):
        ll_vars = list(idata.log_likelihood.data_vars)
        row["has_log_likelihood"] = len(ll_vars) > 0
        row["log_likelihood_vars"] = ",".join(ll_vars)

    if hasattr(idata, "posterior_predictive"):
        pp_vars = list(idata.posterior_predictive.data_vars)
        row["has_posterior_predictive"] = len(pp_vars) > 0
        row["posterior_predictive_vars"] = ",".join(pp_vars)
        row["has_yobs_ppc"] = "y_obs" in pp_vars
        row["has_yrep_ppc"] = "y_rep" in pp_vars

    return row, y


def main():
    rows = []
    y_by_model = {}

    print("\nChecking revised Figure 5 model files...\n")

    for model_key, path in MODEL_FILES.items():
        row, y = summarize_model(model_key, path)
        rows.append(row)

        if y is not None:
            y_by_model[model_key] = y

    if "context_ou" not in y_by_model:
        print("ERROR: context_ou does not contain observed y_obs. Cannot compare observation sets.")
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(df.to_string(index=False))
        raise SystemExit(1)

    y_ref = y_by_model["context_ou"]

    for row in rows:
        model_key = row["model_key"]

        if model_key not in y_by_model:
            continue

        y = y_by_model[model_key]

        row["same_n_as_context_ou"] = len(y) == len(y_ref)
        row["same_y_as_context_ou"] = (
            len(y) == len(y_ref)
            and np.allclose(y, y_ref, equal_nan=True)
        )

    df = pd.DataFrame(rows)

    cols = [
        "model_key",
        "exists",
        "load_ok",
        "has_observed_y",
        "n_obs",
        "same_n_as_context_ou",
        "same_y_as_context_ou",
        "has_log_likelihood",
        "log_likelihood_vars",
        "has_posterior_predictive",
        "posterior_predictive_vars",
        "has_yobs_ppc",
        "has_yrep_ppc",
        "path",
        "error",
    ]

    df = df[cols]
    df.to_csv(OUT_CSV, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved inventory to:\n{OUT_CSV}")

    print("\nRequired for final Figure 5:")
    print("  exists                      = True")
    print("  load_ok                     = True")
    print("  has_observed_y              = True")
    print("  n_obs                       = 434")
    print("  same_y_as_context_ou        = True")
    print("  has_log_likelihood          = True")
    print("  has_posterior_predictive    = True")
    print("  has_yobs_ppc or has_yrep_ppc = True")

    failed = df[
        (~df["exists"])
        | (~df["load_ok"])
        | (~df["has_observed_y"])
        | (~df["same_y_as_context_ou"])
        | (~df["has_log_likelihood"])
        | (~df["has_posterior_predictive"])
        | (~(df["has_yobs_ppc"] | df["has_yrep_ppc"]))
    ]

    if len(failed) == 0:
        print("\nPASS: all model files are compatible for Figure 5.")
    else:
        print("\nWARNING: some model files are not ready for Figure 5:")
        print(failed[["model_key", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
