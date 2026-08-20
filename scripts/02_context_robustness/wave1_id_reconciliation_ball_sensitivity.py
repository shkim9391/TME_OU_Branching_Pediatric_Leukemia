import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency

PRIMARY = ["TNK", "B", "Myeloid"]
K_VALUES = [2, 3, 4, 5, 6]

BALL_NAMES = {
    "B-ALL",
    "B-cell acute lymphoblastic leukemia",
}

def clean_diag(x):
    s = "" if pd.isna(x) else str(x).strip()
    if s in BALL_NAMES:
        return "B-ALL"
    return s if s else "Unknown"

def cramers_v(table):
    arr = np.asarray(table, dtype=float)
    if arr.size == 0 or arr.sum() == 0:
        return np.nan
    chi2, _, _, _ = chi2_contingency(arr, correction=False)
    n = arr.sum()
    r, k = arr.shape
    d = min(r - 1, k - 1)
    return float(np.sqrt((chi2 / n) / d)) if d > 0 else np.nan

def clr_transform(x, pseudocount=1e-6):
    x = np.asarray(x, float)
    x = np.clip(x, 0, None) + pseudocount
    x /= x.sum(axis=1, keepdims=True)
    lx = np.log(x)
    return lx - lx.mean(axis=1, keepdims=True)

def stable_labels(raw_labels, biological_X):
    tmp = pd.DataFrame(biological_X, columns=PRIMARY)
    tmp["raw"] = raw_labels
    means = tmp.groupby("raw")[PRIMARY].mean()
    order = means.sort_values(PRIMARY, ascending=[False, False, False]).index.tolist()
    mp = {old: f"C{i+1}" for i, old in enumerate(order)}
    return np.array([mp[x] for x in raw_labels])

def bootstrap_stability(X, base_labels, k, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(X)
    base_same = base_labels[:, None] == base_labels[None, :]
    tri = np.triu_indices(n, 1)
    glob = []
    by_cluster = {c: [] for c in np.unique(base_labels)}

    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        km = KMeans(n_clusters=k, random_state=seed+b+1, n_init=30).fit(X[idx])
        pred = km.predict(X)
        pred_same = pred[:, None] == pred[None, :]
        glob.append(np.mean(base_same[tri] == pred_same[tri]))
        for c in by_cluster:
            base_set = set(np.where(base_labels == c)[0])
            best = 0.0
            for pc in np.unique(pred):
                ps = set(np.where(pred == pc)[0])
                u = len(base_set | ps)
                j = len(base_set & ps) / u if u else np.nan
                best = max(best, j)
            by_cluster[c].append(best)

    rows = [{
        "k": k, "cluster": "ALL", "metric": "global_pairwise_agreement",
        "mean": np.mean(glob), "median": np.median(glob),
        "q05": np.quantile(glob, .05), "q95": np.quantile(glob, .95),
        "n_boot_effective": len(glob)
    }]
    for c, vals in by_cluster.items():
        vals = np.asarray(vals)
        rows.append({
            "k": k, "cluster": c, "metric": "cluster_jaccard",
            "mean": np.mean(vals), "median": np.median(vals),
            "q05": np.quantile(vals, .05), "q95": np.quantile(vals, .95),
            "n_boot_effective": len(vals)
        })
    return pd.DataFrame(rows)

def scan_subset(df, outdir, n_boot, seed):
    comp_cols = [f"comp_{x}" for x in PRIMARY]
    Xraw = df[comp_cols].to_numpy(float)
    Xclr = StandardScaler().fit_transform(clr_transform(Xraw))

    raw_rows, clr_rows, conc = [], [], []
    brs, bcs = [], []

    for k in K_VALUES:
        kr = KMeans(n_clusters=k, random_state=seed, n_init=100).fit(Xraw)
        lr = stable_labels(kr.labels_, Xraw)
        kc = KMeans(n_clusters=k, random_state=seed, n_init=100).fit(Xclr)
        lc = stable_labels(kc.labels_, Xraw)

        for rep, X, lab, rows in [("raw", Xraw, lr, raw_rows), ("clr", Xclr, lc, clr_rows)]:
            counts = pd.Series(lab).value_counts().sort_index()
            proj = pd.crosstab(lab, df["project"])
            row = {
                "representation": rep, "k": k, "n": len(df),
                "min_cluster_n": int(counts.min()),
                "max_cluster_n": int(counts.max()),
                "cluster_size_cv": float(counts.std(ddof=0)/counts.mean()),
                "silhouette": float(silhouette_score(X, lab)),
                "project_cramers_v": cramers_v(proj.values),
            }
            for c, nn in counts.items():
                row[f"n_{c}"] = int(nn)
            rows.append(row)

        ar = df[["participant_id","Patient_ID","project","diagnosis"]].copy()
        ar["cluster"] = lr
        ar.to_csv(outdir/f"B_ALL_cluster_assignments_raw_k{k}.csv", index=False)
        ac = df[["participant_id","Patient_ID","project","diagnosis"]].copy()
        ac["cluster"] = lc
        ac.to_csv(outdir/f"B_ALL_cluster_assignments_clr_k{k}.csv", index=False)

        br = bootstrap_stability(Xraw, lr, k, n_boot, seed+1000*k)
        br["representation"]="raw"; brs.append(br)
        bc = bootstrap_stability(Xclr, lc, k, n_boot, seed+2000*k)
        bc["representation"]="clr"; bcs.append(bc)

        conc.append({"k": k, "raw_vs_clr_ARI": adjusted_rand_score(lr,lc)})

    raw = pd.DataFrame(raw_rows); clr = pd.DataFrame(clr_rows)
    concord = pd.DataFrame(conc)
    pd.concat(brs,ignore_index=True).to_csv(outdir/"B_ALL_bootstrap_raw.csv",index=False)
    pd.concat(bcs,ignore_index=True).to_csv(outdir/"B_ALL_bootstrap_clr.csv",index=False)
    raw.to_csv(outdir/"B_ALL_k_scan_raw.csv",index=False)
    clr.to_csv(outdir/"B_ALL_k_scan_clr.csv",index=False)
    concord.to_csv(outdir/"B_ALL_raw_clr_concordance.csv",index=False)
    return raw, clr, concord

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/seung-hwan.kim/Desktop/TME_OU_Branching")
    ap.add_argument("--common-dir", default="revision_common_mapper")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    args=ap.parse_args()

    root=Path(args.root)
    out=root/"revision_wave1_id_ball"
    out.mkdir(parents=True,exist_ok=True)

    part_path=root/args.common_dir/"scpcp_combined_participant_TME_features_common_mapper.csv"
    mapping_path=root/"patient_id_mapping.csv"
    if not part_path.exists(): raise FileNotFoundError(part_path)
    if not mapping_path.exists(): raise FileNotFoundError(mapping_path)

    df=pd.read_csv(part_path)
    if "Patient_ID" in df.columns and "participant_id" not in df.columns:
        df=df.rename(columns={"Patient_ID":"participant_id"})
    df["participant_id"]=df["participant_id"].astype(str).str.strip()
    df["diagnosis"]=df["diagnosis"].map(clean_diag)

    mp=pd.read_csv(mapping_path)
    mp["participant_id"]=mp["participant_id"].astype(str).str.strip()
    mp["Patient_ID"]=mp["Patient_ID"].astype(str).str.strip()
    if mp["participant_id"].duplicated().any() or mp["Patient_ID"].duplicated().any():
        raise ValueError("Mapping file contains duplicate IDs.")

    df=df.merge(mp,on="participant_id",how="left",validate="one_to_one")

    raw_cols=[f"frac_{x}" for x in PRIMARY]
    df["primary_total_raw"]=df[raw_cols].sum(axis=1)
    df["eligible_primary"]=df["primary_total_raw"]>0
    df["in_previous_100"]=df["Patient_ID"].notna()

    for x in PRIMARY:
        df[f"comp_{x}"]=np.where(
            df["eligible_primary"],
            df[f"frac_{x}"]/df["primary_total_raw"],
            np.nan
        )

    # full reconciliation
    cols=["participant_id","Patient_ID","project","diagnosis","subdiagnosis",
          "primary_total_raw","eligible_primary","in_previous_100"]
    df[cols].to_csv(out/"cohort_reconciliation_fixed.csv",index=False)

    excluded=df.loc[~df["eligible_primary"], cols].copy()
    excluded.to_csv(out/"primary_simplex_excluded_participants.csv",index=False)

    primary=df.loc[df["eligible_primary"]].copy()
    primary.to_csv(out/"primary_composition_with_legacy_ids.csv",index=False)

    # flow summary
    flow=pd.DataFrame([
        {"stage":"Regenerated common-mapper participants","n":df["participant_id"].nunique()},
        {"stage":"Mapped to previous P1-P100 cohort","n":df["in_previous_100"].sum()},
        {"stage":"Eligible TNK/B/Myeloid primary simplex","n":primary["participant_id"].nunique()},
        {"stage":"Eligible primary simplex AND previous P1-P100","n":(primary["in_previous_100"]).sum()},
        {"stage":"Eligible B-ALL primary simplex","n":(primary["diagnosis"]=="B-ALL").sum()},
        {"stage":"Eligible B-ALL primary simplex AND previous P1-P100","n":((primary["diagnosis"]=="B-ALL") & primary["in_previous_100"]).sum()},
    ])
    flow.to_csv(out/"participant_flow_summary.csv",index=False)

    # B-ALL sensitivity: all eligible B-ALL participants.
    ball=primary.loc[primary["diagnosis"]=="B-ALL"].copy()
    ball.to_csv(out/"B_ALL_primary_composition.csv",index=False)
    raw,clr,conc=scan_subset(ball,out,args.n_boot,args.seed)

    # Composition summary by B-ALL raw k2.
    ass=pd.read_csv(out/"B_ALL_cluster_assignments_raw_k2.csv")
    ball2=ball.merge(ass[["participant_id","cluster"]],on="participant_id",how="left")
    means=ball2.groupby("cluster")[[f"comp_{x}" for x in PRIMARY]].agg(["mean","median","std"])
    means.to_csv(out/"B_ALL_raw_k2_composition_summary.csv")

    # Project x k2 within B-ALL
    pd.crosstab(ball2["cluster"],ball2["project"]).to_csv(out/"B_ALL_raw_k2_project_crosstab.csv")

    print("\\n[OK] ID reconciliation + within-B-ALL sensitivity complete.")
    print("Outputs:",out)
    print("\\nParticipant flow:")
    print(flow.to_string(index=False))
    print("\\nExcluded from primary simplex:")
    print(excluded[["participant_id","Patient_ID","project","diagnosis"]].to_string(index=False))
    print("\\nB-ALL raw scan:")
    print(raw.to_string(index=False))
    print("\\nB-ALL CLR scan:")
    print(clr.to_string(index=False))
    print("\\nB-ALL raw-vs-CLR concordance:")
    print(conc.to_string(index=False))

if __name__=="__main__":
    main()
