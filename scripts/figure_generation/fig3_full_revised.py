import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    import umap
except ImportError as e:
    raise ImportError(
        "The 'umap-learn' package is required for this script.\n"
        "Install it with: pip install umap-learn"
    ) from e


# =============================================================================
# Paths
# =============================================================================

input_path = "/TME_OU_Branching/covariate_matrix.csv"
master_path = "/TME_OU_Branching/patient_master_table.csv"
outdir = "/TME_OU_Branching/Figure_3"
os.makedirs(outdir, exist_ok=True)

out_png = os.path.join(outdir, "Figure3_candidate_ecological_contexts.png")
out_tif = os.path.join(outdir, "Figure3_candidate_ecological_contexts.tiff")
out_master = os.path.join(outdir, "ecological_context_master_table.csv")
out_assignments = os.path.join(outdir, "patient_ecological_context_assignments.csv")
out_color_key = os.path.join(outdir, "ecological_context_color_key.json")


# =============================================================================
# Optional manual override
# =============================================================================
# First run with MANUAL_RAW_TO_CONTEXT = None.
# Inspect ecological_context_master_table.csv.
# If you need a specific raw KMeans cluster to become E1/E2/E3/E4, set:
#
# MANUAL_RAW_TO_CONTEXT = {
#     "0": "E1",
#     "1": "E2",
#     "2": "E3",
#     "3": "E4",
# }
#
# Raw cluster labels are strings in the saved master table.
#
MANUAL_RAW_TO_CONTEXT = None


# =============================================================================
# Feature definitions
# =============================================================================

z_feature_map = {
    "frac_T_given_known_z": "T_z",
    "frac_B_given_known_z": "B_z",
    "frac_myeloid_given_known_z": "Myeloid_z",
    "frac_NK_given_known_z": "NK_z",
    "frac_stromal_given_known_z": "Stromal_z",
    "frac_unknown_z": "Unknown_z",
}

z_tme_features = ["T_z", "B_z", "Myeloid_z", "NK_z", "Stromal_z", "Unknown_z"]

feature_labels = {
    "T_z": "T",
    "B_z": "B",
    "Myeloid_z": "Myeloid",
    "NK_z": "NK",
    "Stromal_z": "Stromal",
    "Unknown_z": "Unknown",
}

diagnosis_label_map = {
    "Acute myeloid leukemia": "AML",
    "B-cell acute lymphoblastic leukemia": "B-ALL",
    "Early T-cell precursor T-cell acute lymphoblastic leukemia": "ETP-ALL",
    "Mixed phenotype acute leukemia": "MPAL",
    "T-cell acute lymphoblastic leukemia": "T-ALL",
}


# =============================================================================
# Colors
# =============================================================================

# Keep these colors fixed in every downstream figure.
context_color_map = {
    "E1": "#4C78A8",  # blue
    "E2": "#59A14F",  # green
    "E3": "#F28E2B",  # orange
    "E4": "#B279A2",  # purple
}

diagnosis_color_map = {
    "AML": "#1f77b4",
    "B-ALL": "#ff7f0e",
    "ETP-ALL": "#2ca02c",
    "MPAL": "#d62728",
    "T-ALL": "#9467bd",
    "Unknown": "#7f7f7f",
}


# =============================================================================
# Helpers
# =============================================================================

def first_non_null(series):
    """Return the first non-null value in a Series, otherwise 'Unknown'."""
    x = series.dropna()
    if len(x) == 0:
        return "Unknown"
    return x.iloc[0]


def clean_diagnosis(x):
    """Map verbose diagnosis names to short labels."""
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip()
    return diagnosis_label_map.get(x, x)


def signature_label(row, is_dominant=False):
    """
    Build a conservative biological label from positive z-score features.
    Labels are descriptive, not definitive subtype claims.
    """
    if is_dominant:
        mean_abs = np.mean(np.abs(row[z_tme_features].values))
        if mean_abs < 0.75:
            return "typical/low-deviation"
        return "dominant context"

    # Feature groups used only for descriptive labels.
    group_scores = {
        "T/NK": row["T_z"] + row["NK_z"],
        "B-lineage": row["B_z"],
        "myeloid": row["Myeloid_z"],
        "stromal/unknown": row["Stromal_z"] + row["Unknown_z"],
        "unknown/other": row["Unknown_z"],
    }

    # Pick the best group, but avoid over-labeling weak clusters.
    best_group = max(group_scores, key=group_scores.get)
    best_score = group_scores[best_group]

    if best_score < 0.5:
        return "weakly differentiated"

    # Add one secondary feature if it is clearly positive.
    single_features = row[z_tme_features].sort_values(ascending=False)
    positive_features = [
        feature_labels[f]
        for f, val in single_features.items()
        if val >= 0.75
    ]

    if len(positive_features) >= 2:
        return "/".join(positive_features[:2]) + "-enriched"
    if len(positive_features) == 1:
        return positive_features[0] + "-enriched"

    return best_group + "-enriched"


def assign_stable_context_ids(df_tme, raw_col, features):
    """
    Convert arbitrary KMeans raw labels into stable E1-E4 labels.

    Rule:
      - E1 = largest cluster, interpreted as dominant/low-deviation context.
      - E2-E4 = remaining clusters ordered by dominant positive feature priority.
    """
    counts = df_tme.groupby(raw_col).size()
    means = df_tme.groupby(raw_col)[features].mean()

    raw_labels = list(means.index.astype(str))
    counts.index = counts.index.astype(str)
    means.index = means.index.astype(str)

    if MANUAL_RAW_TO_CONTEXT is not None:
        raw_to_context = {str(k): str(v) for k, v in MANUAL_RAW_TO_CONTEXT.items()}
        required = {"E1", "E2", "E3", "E4"}
        observed = set(raw_to_context.values())
        if observed != required:
            raise ValueError(
                f"MANUAL_RAW_TO_CONTEXT must map to exactly {required}; got {observed}"
            )
        return raw_to_context

    # E1 is the largest cluster.
    largest_raw = counts.idxmax()

    # Remaining clusters are sorted by their most positive feature.
    priority = {
        "T_z": 0,
        "NK_z": 1,
        "Myeloid_z": 2,
        "Stromal_z": 3,
        "Unknown_z": 4,
        "B_z": 5,
    }

    remaining = [r for r in raw_labels if r != largest_raw]

    def sort_key(raw):
        row = means.loc[raw, features]
        top_feature = row.idxmax()
        return (priority.get(top_feature, 99), -counts.loc[raw])

    remaining_sorted = sorted(remaining, key=sort_key)

    raw_to_context = {largest_raw: "E1"}
    for raw, eid in zip(remaining_sorted, ["E2", "E3", "E4"]):
        raw_to_context[raw] = eid

    return raw_to_context


# =============================================================================
# Load data and metadata
# =============================================================================

df = pd.read_csv(input_path)

if "Patient_ID" not in df.columns:
    raise ValueError("covariate_matrix.csv must contain a Patient_ID column.")

# Merge metadata only for columns not already present.
needed_meta_cols = [
    "Patient_ID",
    "diagnosis",
    "subdiagnosis",
    "frac_T_given_known",
    "frac_B_given_known",
    "frac_myeloid_given_known",
    "frac_NK_given_known",
    "frac_stromal_given_known",
    "frac_unknown",
]

if os.path.exists(master_path):
    meta_all = pd.read_csv(master_path)
    merge_cols = ["Patient_ID"] + [
        c for c in needed_meta_cols
        if c != "Patient_ID" and c in meta_all.columns and c not in df.columns
    ]
    if len(merge_cols) > 1:
        df = df.merge(meta_all[merge_cols], on="Patient_ID", how="left")
else:
    print(f"Warning: master table not found: {master_path}")

# Rename standardized TME features.
for old, new in z_feature_map.items():
    if old in df.columns and new not in df.columns:
        df = df.rename(columns={old: new})

missing = [c for c in z_tme_features if c not in df.columns]
if missing:
    raise ValueError(f"Missing standardized TME feature columns: {missing}")

# Diagnosis metadata.
if "diagnosis" not in df.columns:
    df["diagnosis"] = "Unknown"
if "subdiagnosis" not in df.columns:
    df["subdiagnosis"] = "Unknown"

# Participant-level table.
# This prevents multiple sample/timepoint rows from influencing clustering twice.
agg_features = df.groupby("Patient_ID", as_index=True)[z_tme_features].mean()
agg_meta = df.groupby("Patient_ID", as_index=True)[["diagnosis", "subdiagnosis"]].agg(first_non_null)

df_tme = agg_features.join(agg_meta).reset_index()
df_tme = df_tme.dropna(subset=z_tme_features).copy()

df_tme["diagnosis"] = df_tme["diagnosis"].map(clean_diagnosis)
df_tme["subdiagnosis"] = df_tme["subdiagnosis"].fillna("Unknown").astype(str).str.strip()

X = df_tme[z_tme_features].values
n_participants = len(df_tme)

print(f"Retained participant-level TME profiles: n = {n_participants}")


# =============================================================================
# KMeans ecological-context inference
# =============================================================================

n_contexts = 4
kmeans = KMeans(n_clusters=n_contexts, random_state=123, n_init=50)
raw_labels = kmeans.fit_predict(X).astype(str)

df_tme["raw_kmeans_cluster"] = raw_labels

raw_to_context = assign_stable_context_ids(
    df_tme=df_tme,
    raw_col="raw_kmeans_cluster",
    features=z_tme_features,
)

df_tme["ecological_context"] = df_tme["raw_kmeans_cluster"].map(raw_to_context)

context_order = ["E1", "E2", "E3", "E4"]
df_tme["ecological_context"] = pd.Categorical(
    df_tme["ecological_context"],
    categories=context_order,
    ordered=True,
)

# Mean profiles and counts.
context_means = (
    df_tme.groupby("ecological_context", observed=False)[z_tme_features]
    .mean()
    .reindex(context_order)
)

context_counts = (
    df_tme.groupby("ecological_context", observed=False)
    .size()
    .reindex(context_order)
    .fillna(0)
    .astype(int)
)

# Build labels.
context_signature = {}
for eid in context_order:
    row = context_means.loc[eid]
    is_dominant = (eid == "E1")
    context_signature[eid] = signature_label(row, is_dominant=is_dominant)

context_display = {
    eid: f"{eid}: {context_signature[eid]} (n={context_counts.loc[eid]})"
    for eid in context_order
}

# Save master table.
master_rows = []
for raw, eid in raw_to_context.items():
    row = {
        "raw_kmeans_cluster": raw,
        "ecological_context": eid,
        "context_label": context_signature[eid],
        "n_participants": int(context_counts.loc[eid]),
        "color": context_color_map[eid],
    }
    for feat in z_tme_features:
        row[f"mean_{feat}"] = float(context_means.loc[eid, feat])
    master_rows.append(row)

master_df = pd.DataFrame(master_rows).sort_values("ecological_context")
master_df.to_csv(out_master, index=False)

df_tme.to_csv(out_assignments, index=False)

with open(out_color_key, "w") as f:
    json.dump(context_color_map, f, indent=2)

print("\nRaw cluster -> stable ecological context mapping:")
print(pd.DataFrame(master_rows).sort_values("ecological_context"))

print(f"\nSaved master table: {out_master}")
print(f"Saved assignments: {out_assignments}")
print(f"Saved color key: {out_color_key}")


# =============================================================================
# PCA and UMAP
# =============================================================================

pca = PCA(n_components=2, random_state=123)
X_pca = pca.fit_transform(X)
expl_var = pca.explained_variance_ratio_ * 100

n_neighbors = max(2, min(10, n_participants - 1))
umap_model = umap.UMAP(
    n_neighbors=10,
    min_dist=0.3,
    n_components=2,
    metric="euclidean",
    init="spectral",
    random_state=123,
    transform_seed=123,
    n_jobs=1,
)
X_umap = umap_model.fit_transform(X)

df_tme["PC1"] = X_pca[:, 0]
df_tme["PC2"] = X_pca[:, 1]
df_tme["UMAP1"] = X_umap[:, 0]
df_tme["UMAP2"] = X_umap[:, 1]


# =============================================================================
# Diagnosis/context cross-tab
# =============================================================================

diag_order = [d for d in ["B-ALL", "T-ALL", "ETP-ALL", "AML", "MPAL", "Unknown"]
              if d in set(df_tme["diagnosis"])]

# Add any nonstandard labels at the end.
for d in sorted(set(df_tme["diagnosis"])):
    if d not in diag_order:
        diag_order.append(d)

diag_counts = pd.crosstab(df_tme["ecological_context"], df_tme["diagnosis"])
diag_counts = diag_counts.reindex(index=context_order, columns=diag_order, fill_value=0)
diag_prop = diag_counts.div(diag_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


# =============================================================================
# Plot
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

fig = plt.figure(figsize=(15.5, 11.0))
gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[1.0, 1.08],
    width_ratios=[1.0, 1.05],
    wspace=0.44,
    hspace=0.48,
)

ax_pca = fig.add_subplot(gs[0, 0])
ax_umap = fig.add_subplot(gs[0, 1])
ax_heat = fig.add_subplot(gs[1, 0])
ax_diag = fig.add_subplot(gs[1, 1])

fig.subplots_adjust(left=0.07, right=0.88, top=0.94, bottom=0.08)


# -----------------------------------------------------------------------------
# Panel A: PCA by diagnosis
# -----------------------------------------------------------------------------

for d in diag_order:
    idx = df_tme["diagnosis"].eq(d).values
    if idx.sum() == 0:
        continue
    ax_pca.scatter(
        df_tme.loc[idx, "PC1"],
        df_tme.loc[idx, "PC2"],
        s=34,
        alpha=0.85,
        label=f"{d} (n={idx.sum()})",
        color=diagnosis_color_map.get(d, "#7f7f7f"),
        edgecolor="white",
        linewidth=0.5,
    )

ax_pca.set_xlabel(f"PC1 ({expl_var[0]:.1f}% var.)")
ax_pca.set_ylabel(f"PC2 ({expl_var[1]:.1f}% var.)")
ax_pca.set_title("PCA of standardized TME features by diagnosis")

legA = ax_pca.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1.00),
    frameon=False,
    markerscale=1.3,
    handletextpad=0.5,
    labelspacing=0.35,
)
plt.setp(legA.get_texts(), fontsize=9.5)

ax_pca.text(
    -0.16, 1.14, "A",
    transform=ax_pca.transAxes,
    fontsize=24,
    fontweight="bold",
    va="top",
)


# -----------------------------------------------------------------------------
# Panel B: UMAP by candidate ecological context
# -----------------------------------------------------------------------------

for eid in context_order:
    idx = df_tme["ecological_context"].eq(eid).values
    if idx.sum() == 0:
        continue
    ax_umap.scatter(
        df_tme.loc[idx, "UMAP1"],
        df_tme.loc[idx, "UMAP2"],
        s=34,
        alpha=0.88,
        label=context_display[eid],
        color=context_color_map[eid],
        edgecolor="white",
        linewidth=0.5,
    )

ax_umap.set_xlabel("UMAP 1")
ax_umap.set_ylabel("UMAP 2")
ax_umap.set_title("UMAP of TME features by candidate ecological context")

legB = ax_umap.legend(
    bbox_to_anchor=(1.01, 1.00),
    loc="upper left",
    frameon=False,
    markerscale=1.25,
    handletextpad=0.5,
    labelspacing=0.4,
)
plt.setp(legB.get_texts(), fontsize=9.5)

ax_umap.text(
    -0.15, 1.14, "B",
    transform=ax_umap.transAxes,
    fontsize=24,
    fontweight="bold",
    va="top",
)


# -----------------------------------------------------------------------------
# Panel C: Mean standardized TME features by context, heatmap
# -----------------------------------------------------------------------------

heat_data = context_means.loc[context_order, z_tme_features].values
max_abs = np.nanmax(np.abs(heat_data))
max_abs = max(max_abs, 1.0)

im = ax_heat.imshow(
    heat_data,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-max_abs,
    vmax=max_abs,
)

ax_heat.set_xticks(np.arange(len(z_tme_features)))
ax_heat.set_xticklabels([feature_labels[f] for f in z_tme_features], rotation=35, ha="right")

row_labels = [
    f"{eid}\n(n={context_counts.loc[eid]})"
    for eid in context_order
]
ax_heat.set_yticks(np.arange(len(context_order)))
ax_heat.set_yticklabels(row_labels)

ax_heat.set_title("Mean standardized TME features by ecological context")

# Add cell values.
for i in range(len(context_order)):
    for j in range(len(z_tme_features)):
        val = heat_data[i, j]
        text_color = "white" if abs(val) > 0.6 * max_abs else "black"
        ax_heat.text(
            j, i, f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
        )

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
cbar.set_label("Mean z-score")

ax_heat.text(
    -0.16, 1.14, "C",
    transform=ax_heat.transAxes,
    fontsize=24,
    fontweight="bold",
    va="top",
)


# -----------------------------------------------------------------------------
# Panel D: Diagnosis composition by ecological context
# -----------------------------------------------------------------------------

bottom = np.zeros(len(context_order))
x = np.arange(len(context_order))

for d in diag_order:
    vals = diag_prop[d].values
    ax_diag.bar(
        x,
        vals,
        bottom=bottom,
        label=d,
        color=diagnosis_color_map.get(d, "#7f7f7f"),
        edgecolor="white",
        linewidth=0.6,
    )
    bottom += vals

ax_diag.set_ylim(0, 1.0)
ax_diag.set_xticks(x)
ax_diag.set_xticklabels([f"{eid}\n(n={context_counts.loc[eid]})" for eid in context_order])
ax_diag.set_ylabel("Proportion of participants")
ax_diag.set_title("Diagnosis composition within ecological contexts", pad=10)

legD = ax_diag.legend(
    title="Diagnosis",
    bbox_to_anchor=(1.01, 1.00),
    loc="upper left",
    frameon=False,
    handletextpad=0.5,
    labelspacing=0.4,
)
plt.setp(legD.get_texts(), fontsize=9.5)
plt.setp(legD.get_title(), fontsize=10.5)

ax_diag.text(
    -0.15, 1.14, "D",
    transform=ax_diag.transAxes,
    fontsize=24,
    fontweight="bold",
    va="top",
)


# =============================================================================
# Save
# =============================================================================

fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_tif, dpi=600, bbox_inches="tight")

print(f"\nSaved Figure 3 PNG:  {out_png}")
print(f"Saved Figure 3 TIFF: {out_tif}")

plt.show()
