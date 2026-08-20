import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency


PRIMARY = ["TNK", "B", "Myeloid"]
K_VALUES = [2, 3, 4, 5, 6]


def cramers_v(table):
    arr = np.asarray(table, dtype=float)
    if arr.size == 0 or arr.sum() == 0:
        return np.nan
    chi2, _, _, _ = chi2_contingency(arr, correction=False)
    n = arr.sum()
    r, k = arr.shape
    denom = min(k - 1, r - 1)
    if denom <= 0 or n <= 0:
        return np.nan
    return float(np.sqrt((chi2 / n) / denom))


def clr_transform(comp, pseudocount=1e-6):
    x = np.asarray(comp, dtype=float)
    x = np.clip(x, 0, None)
    x = x + pseudocount
    x = x / x.sum(axis=1, keepdims=True)
    logx = np.log(x)
    return logx - logx.mean(axis=1, keepdims=True)


def stable_label_order(labels, X):
    """
    Convert arbitrary KMeans labels to stable labels C1..Ck.
    Ordering rule: descending mean TNK, then descending B, then descending Myeloid.
    """
    tmp = pd.DataFrame(X, columns=PRIMARY)
    tmp["raw"] = labels
    means = tmp.groupby("raw")[PRIMARY].mean()
    order = means.sort_values(PRIMARY, ascending=[False, False, False]).index.tolist()
    mapping = {old: f"C{i+1}" for i, old in enumerate(order)}
    return np.array([mapping[x] for x in labels]), mapping


def fit_kmeans(X, k, seed=123):
    km = KMeans(n_clusters=k, random_state=seed, n_init=100)
    raw = km.fit_predict(X)
    return km, raw


def bootstrap_stability(X, base_labels, k, n_boot=500, seed=12345):
    """
    Participant bootstrap stability using co-clustering agreement.

    For each bootstrap sample:
      - resample participants with replacement
      - fit KMeans to unique represented participants' rows
      - predict cluster labels for all original participants using bootstrap centroids
      - compare pairwise same-cluster matrices with the base solution

    Returns global agreement and cluster-specific Jaccard-like co-membership stability.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    base_same = base_labels[:, None] == base_labels[None, :]

    global_scores = []
    cluster_scores = {c: [] for c in np.unique(base_labels)}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        try:
            km = KMeans(n_clusters=k, random_state=seed + b + 1, n_init=30).fit(Xb)
            pred = km.predict(X)
        except Exception:
            continue

        pred_same = pred[:, None] == pred[None, :]
        tri = np.triu_indices(n, 1)
        global_scores.append(np.mean(base_same[tri] == pred_same[tri]))

        for c in cluster_scores:
            members = np.where(base_labels == c)[0]
            if len(members) < 2:
                cluster_scores[c].append(np.nan)
                continue
            base_set = set(members.tolist())
            # bootstrap-predicted cluster with maximum overlap with base cluster
            best = 0.0
            for pc in np.unique(pred):
                pred_set = set(np.where(pred == pc)[0].tolist())
                inter = len(base_set & pred_set)
                union = len(base_set | pred_set)
                jac = inter / union if union else np.nan
                if jac > best:
                    best = jac
            cluster_scores[c].append(best)

    rows = [{
        "k": k,
        "metric": "global_pairwise_agreement",
        "cluster": "ALL",
        "mean": float(np.nanmean(global_scores)),
        "median": float(np.nanmedian(global_scores)),
        "q05": float(np.nanquantile(global_scores, 0.05)),
        "q95": float(np.nanquantile(global_scores, 0.95)),
        "n_boot_effective": int(np.sum(np.isfinite(global_scores))),
    }]

    for c, vals in cluster_scores.items():
        vals = np.asarray(vals, dtype=float)
        rows.append({
            "k": k,
            "metric": "cluster_jaccard",
            "cluster": str(c),
            "mean": float(np.nanmean(vals)),
            "median": float(np.nanmedian(vals)),
            "q05": float(np.nanquantile(vals, 0.05)),
            "q95": float(np.nanquantile(vals, 0.95)),
            "n_boot_effective": int(np.sum(np.isfinite(vals))),
        })
    return pd.DataFrame(rows)


def summarize_solution(df, labels, X, k, representation):
    out = {}
    out["representation"] = representation
    out["k"] = k
    out["n"] = len(df)
    counts = pd.Series(labels).value_counts().sort_index()
    out["min_cluster_n"] = int(counts.min())
    out["max_cluster_n"] = int(counts.max())
    out["cluster_size_cv"] = float(counts.std(ddof=0) / counts.mean()) if counts.mean() else np.nan
    out["silhouette"] = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else np.nan

    proj_tab = pd.crosstab(labels, df["project"])
    out["project_cramers_v"] = cramers_v(proj_tab.values)

    if "diagnosis" in df.columns:
        diag_tab = pd.crosstab(labels, df["diagnosis"].fillna("Unknown"))
        out["diagnosis_cramers_v"] = cramers_v(diag_tab.values)
    else:
        out["diagnosis_cramers_v"] = np.nan

    for c, n in counts.items():
        out[f"n_{c}"] = int(n)
    return out


def make_k_scan_plot(scan_df, outfile, title):
    ks = scan_df["k"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ks, scan_df["silhouette"], marker="o", label="Silhouette")
    ax.plot(ks, scan_df["project_cramers_v"], marker="o", label="Project Cramér's V")
    ax.plot(ks, scan_df["diagnosis_cramers_v"], marker="o", label="Diagnosis Cramér's V")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Score")
    ax.set_xticks(ks)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/Users/seung-hwan.kim/Desktop/TME_OU_Branching",
        help="Project root"
    )
    ap.add_argument(
        "--common-dir",
        default="revision_common_mapper",
        help="Directory containing regenerated common-mapper outputs"
    )
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    root = Path(args.root)
    common_dir = root / args.common_dir
    outdir = root / "revision_wave1_context_stability"
    outdir.mkdir(parents=True, exist_ok=True)

    in_part = common_dir / "scpcp_combined_participant_TME_features_common_mapper.csv"
    if not in_part.exists():
        raise FileNotFoundError(in_part)

    df = pd.read_csv(in_part)
    if "participant_id" not in df.columns:
        raise ValueError("Expected participant_id in regenerated participant table.")

    # harmonize ID name
    df["participant_id"] = df["participant_id"].astype(str).str.strip()

    # --------------------------------------------------
    # Cohort reconciliation against previous 100-person table
    # --------------------------------------------------
    prev_ids = set()
    cov_path = root / "covariate_matrix.csv"
    if cov_path.exists():
        prev = pd.read_csv(cov_path)
        pid_col = "Patient_ID" if "Patient_ID" in prev.columns else (
            "participant_id" if "participant_id" in prev.columns else None
        )
        if pid_col:
            prev_ids = set(prev[pid_col].astype(str).str.strip())

    reconciliation = df[["participant_id", "project"]].copy()
    reconciliation["in_previous_covariate_matrix"] = reconciliation["participant_id"].isin(prev_ids)
    reconciliation.to_csv(outdir / "cohort_reconciliation.csv", index=False)

    # --------------------------------------------------
    # Build primary TNK/B/Myeloid simplex
    # --------------------------------------------------
    raw_cols = [f"frac_{c}" for c in PRIMARY]
    missing = [c for c in raw_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing primary fraction columns: {missing}")

    primary = df.copy()
    denom = primary[raw_cols].sum(axis=1)
    primary["primary_total_raw"] = denom
    primary = primary.loc[denom > 0].copy()

    for c in PRIMARY:
        primary[f"comp_{c}"] = primary[f"frac_{c}"] / primary["primary_total_raw"]

    comp_cols = [f"comp_{c}" for c in PRIMARY]

    # attach diagnosis from previous master/covariate table if not present
    if "diagnosis" not in primary.columns or primary["diagnosis"].isna().all():
        master_path = root / "patient_master_table.csv"
        if master_path.exists():
            master = pd.read_csv(master_path)
            pid_col = "Patient_ID" if "Patient_ID" in master.columns else "participant_id"
            keep = [pid_col] + [c for c in ["diagnosis", "subdiagnosis"] if c in master.columns]
            meta = master[keep].drop_duplicates(pid_col).copy()
            meta[pid_col] = meta[pid_col].astype(str).str.strip()
            meta = meta.rename(columns={pid_col: "participant_id"})
            primary = primary.drop(columns=[c for c in ["diagnosis", "subdiagnosis"] if c in primary.columns],
                                   errors="ignore")
            primary = primary.merge(meta, on="participant_id", how="left")

    primary.to_csv(outdir / "primary_composition_participant.csv", index=False)

    X_raw = primary[comp_cols].to_numpy(dtype=float)
    X_clr = clr_transform(X_raw, pseudocount=1e-6)

    # standardize CLR coordinates for Euclidean KMeans; raw simplex remains raw proportions
    X_clr_scaled = StandardScaler().fit_transform(X_clr)

    scan_raw = []
    scan_clr = []
    concordance = []
    boot_raw_all = []
    boot_clr_all = []

    for k in K_VALUES:
        # RAW
        kmr, lr0 = fit_kmeans(X_raw, k, seed=args.seed)
        lr, mapr = stable_label_order(lr0, X_raw)
        sr = summarize_solution(primary, lr, X_raw, k, "raw")
        scan_raw.append(sr)

        adf = primary[["participant_id", "project"] + ([ "diagnosis" ] if "diagnosis" in primary.columns else [])].copy()
        adf["cluster"] = lr
        adf.to_csv(outdir / f"cluster_assignments_raw_k{k}.csv", index=False)

        br = bootstrap_stability(X_raw, lr, k, n_boot=args.n_boot, seed=args.seed + 1000*k)
        br["representation"] = "raw"
        boot_raw_all.append(br)

        # CLR
        kmc, lc0 = fit_kmeans(X_clr_scaled, k, seed=args.seed)
        # stable ordering uses the original biological composition means
        lc, mapc = stable_label_order(lc0, X_raw)
        sc = summarize_solution(primary, lc, X_clr_scaled, k, "clr")
        scan_clr.append(sc)

        adf2 = primary[["participant_id", "project"] + ([ "diagnosis" ] if "diagnosis" in primary.columns else [])].copy()
        adf2["cluster"] = lc
        adf2.to_csv(outdir / f"cluster_assignments_clr_k{k}.csv", index=False)

        bc = bootstrap_stability(X_clr_scaled, lc, k, n_boot=args.n_boot, seed=args.seed + 2000*k)
        bc["representation"] = "clr"
        boot_clr_all.append(bc)

        concordance.append({
            "k": k,
            "raw_vs_clr_ARI": adjusted_rand_score(lr, lc),
        })

    scan_raw = pd.DataFrame(scan_raw)
    scan_clr = pd.DataFrame(scan_clr)
    concordance = pd.DataFrame(concordance)
    boot_raw = pd.concat(boot_raw_all, ignore_index=True)
    boot_clr = pd.concat(boot_clr_all, ignore_index=True)

    scan_raw.to_csv(outdir / "k_scan_raw.csv", index=False)
    scan_clr.to_csv(outdir / "k_scan_clr.csv", index=False)
    concordance.to_csv(outdir / "raw_clr_concordance.csv", index=False)
    boot_raw.to_csv(outdir / "bootstrap_cluster_stability_raw.csv", index=False)
    boot_clr.to_csv(outdir / "bootstrap_cluster_stability_clr.csv", index=False)

    make_k_scan_plot(
        scan_raw,
        outdir / "Figure_wave1_k_scan_raw.png",
        "Wave 1 ecological-context scan: raw TNK/B/Myeloid composition"
    )
    make_k_scan_plot(
        scan_clr,
        outdir / "Figure_wave1_k_scan_clr.png",
        "Wave 1 ecological-context scan: CLR TNK/B/Myeloid composition"
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(concordance["k"], concordance["raw_vs_clr_ARI"], marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Adjusted Rand index")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(concordance["k"])
    ax.set_title("Raw versus CLR cluster concordance")
    fig.tight_layout()
    fig.savefig(outdir / "Figure_wave1_raw_vs_clr_ari.png", dpi=300)
    plt.close(fig)

    summary = {
        "n_regenerated_participants": int(df["participant_id"].nunique()),
        "n_primary_participants": int(primary["participant_id"].nunique()),
        "n_in_previous_covariate_matrix": int(reconciliation["in_previous_covariate_matrix"].sum()),
        "n_new_vs_previous": int((~reconciliation["in_previous_covariate_matrix"]).sum()),
        "primary_components": PRIMARY,
        "k_values": K_VALUES,
        "n_boot": args.n_boot,
    }
    with open(outdir / "wave1_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] Wave 1 ecological-context stability analysis complete.")
    print("Outputs:", outdir)
    print("\nCohort summary:")
    print(json.dumps(summary, indent=2))
    print("\nRaw k scan:")
    print(scan_raw.to_string(index=False))
    print("\nCLR k scan:")
    print(scan_clr.to_string(index=False))
    print("\nRaw vs CLR concordance:")
    print(concordance.to_string(index=False))


if __name__ == "__main__":
    main()
