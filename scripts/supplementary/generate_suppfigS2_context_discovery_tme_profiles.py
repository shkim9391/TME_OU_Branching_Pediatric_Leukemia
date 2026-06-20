from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# User settings
# ============================================================

BASE_DIR = Path("/TME_OU_Branching")

PATIENT_FILE = BASE_DIR / "patient_master_table.csv"
CONTEXT_FILE = BASE_DIR / "Figure_3" / "patient_ecological_context_assignments.csv"

OUT_DIR = BASE_DIR / "BMC_Bioinformatics" / "Supplementary_Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = "SuppFigS2_context_discovery_tme_profiles"

CONTEXT_ORDER = ["E1", "E2", "E3", "E4"]

CONTEXT_COLORS = {
    "E1": "#4C78A8",
    "E2": "#59A14F",
    "E3": "#F28E2B",
    "E4": "#B279A2",
}

TME_FEATURES = [
    "T_z",
    "B_z",
    "Myeloid_z",
    "NK_z",
    "Stromal_z",
    "Unknown_z",
]

TME_FEATURE_LABELS = {
    "T_z": "T",
    "B_z": "B",
    "Myeloid_z": "Myeloid",
    "NK_z": "NK",
    "Stromal_z": "Stromal",
    "Unknown_z": "Unknown",
}


# ============================================================
# Helper functions
# ============================================================

def require_file(path):
    """Raise a clear error if a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


def pick_col(df, candidates, required=True):
    """Pick the first available column from candidate names."""
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def clean_context(value):
    """Standardize ecological-context labels."""
    if pd.isna(value):
        return "Unknown"

    label = str(value).strip()
    numeric_to_context = {"0": "E1", "1": "E2", "2": "E3", "3": "E4"}

    if label in numeric_to_context:
        return numeric_to_context[label]

    if label.upper() in CONTEXT_ORDER:
        return label.upper()

    return label


def add_panel_label(ax, label):
    ax.text(
        -0.10,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Load input files
# ============================================================

patient_file = require_file(PATIENT_FILE)
context_file = require_file(CONTEXT_FILE)

patient_df = pd.read_csv(patient_file)
context_df = pd.read_csv(context_file)

print("\nUsing patient file:")
print(patient_file)

print("\nUsing context file:")
print(context_file)

print("\nPatient table columns:")
print(patient_df.columns.tolist())

print("\nContext table columns:")
print(context_df.columns.tolist())


# ============================================================
# Validate columns and prepare analysis table
# ============================================================

patient_id_col = pick_col(
    context_df,
    ["Patient_ID", "patient_id", "participant_id", "Participant_ID", "sample_id"],
)

context_col = pick_col(
    context_df,
    [
        "ecological_context",
        "candidate_ecological_context",
        "context_label",
        "context",
        "immune_ecotype",
        "ecotype",
        "ecotype_label",
    ],
)

missing_features = [col for col in TME_FEATURES if col not in context_df.columns]
if missing_features:
    raise KeyError(
        "The canonical Figure_3 context file is missing expected TME feature columns:\n"
        f"{missing_features}\n"
        f"Available columns: {list(context_df.columns)}"
    )

merged = context_df[[patient_id_col, context_col] + TME_FEATURES].copy()
merged = merged.rename(
    columns={
        patient_id_col: "Patient_ID",
        context_col: "ecological_context",
    }
)

merged["Patient_ID"] = merged["Patient_ID"].astype(str)
merged["ecological_context"] = merged["ecological_context"].apply(clean_context)

for col in TME_FEATURES:
    merged[col] = pd.to_numeric(merged[col], errors="coerce")

merged = merged.drop_duplicates(subset=["Patient_ID"]).copy()
merged = merged[merged["ecological_context"].isin(CONTEXT_ORDER)].copy()
merged = merged.dropna(subset=TME_FEATURES).copy()

merged["context_order"] = merged["ecological_context"].map(
    {context: idx for idx, context in enumerate(CONTEXT_ORDER)}
)
merged = merged.sort_values(["context_order", "Patient_ID"]).reset_index(drop=True)

print("\nMerged feature/context summary:")
print(merged[["Patient_ID", "ecological_context"] + TME_FEATURES].head())

print("\nContext counts:")
print(merged["ecological_context"].value_counts().reindex(CONTEXT_ORDER, fill_value=0))


# ============================================================
# Matrix preparation
# ============================================================

X = merged[TME_FEATURES].to_numpy(dtype=float)

# Values are already standardized, but this keeps PCA numerically stable
# if a future input table uses slightly different scaling.
X_scaled = StandardScaler().fit_transform(X)

feature_labels = [TME_FEATURE_LABELS[col] for col in TME_FEATURES]

pca = PCA(n_components=2)
pc_scores = pca.fit_transform(X_scaled)
merged["PC1"] = pc_scores[:, 0]
merged["PC2"] = pc_scores[:, 1]

pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100

prototype_df = (
    merged.groupby("ecological_context")[TME_FEATURES]
    .mean()
    .reindex(CONTEXT_ORDER)
)

prototype_export = prototype_df.copy()
prototype_export.index.name = "ecological_context"
prototype_export.to_csv(OUT_DIR / "SuppFigS2_context_prototype_table.csv")

distance_rows = []
for _, row in merged.iterrows():
    context = row["ecological_context"]
    patient_vector = row[TME_FEATURES].to_numpy(dtype=float)
    prototype_vector = prototype_df.loc[context, TME_FEATURES].to_numpy(dtype=float)
    distance = float(np.linalg.norm(patient_vector - prototype_vector))

    distance_rows.append(
        {
            "Patient_ID": row["Patient_ID"],
            "ecological_context": context,
            "distance_to_context_prototype": distance,
        }
    )

distance_df = pd.DataFrame(distance_rows)
distance_df.to_csv(
    OUT_DIR / "SuppFigS2_patient_to_prototype_distances.csv",
    index=False,
)


# ============================================================
# Plot
# ============================================================

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 13,
    }
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), constrained_layout=False)
axA, axB = axes[0, 0], axes[0, 1]
axC, axD = axes[1, 0], axes[1, 1]


# ------------------------------------------------------------
# Panel A: Heatmap ordered by context
# ------------------------------------------------------------

heatmap_data = merged[TME_FEATURES].to_numpy(dtype=float)
vmax = np.nanpercentile(np.abs(heatmap_data), 95)
if vmax <= 0 or np.isnan(vmax):
    vmax = 1.0

im = axA.imshow(
    heatmap_data,
    aspect="auto",
    interpolation="nearest",
    cmap="coolwarm",
    vmin=-vmax,
    vmax=vmax,
)

axA.set_title("TME feature matrix ordered by context")
axA.set_xlabel("TME feature")
axA.set_ylabel("Participants")
axA.set_xticks(np.arange(len(feature_labels)))
axA.set_xticklabels(feature_labels, rotation=35, ha="right")
axA.set_yticks([])

context_counts = (
    merged["ecological_context"]
    .value_counts()
    .reindex(CONTEXT_ORDER, fill_value=0)
)

start = 0
for context in CONTEXT_ORDER:
    count = int(context_counts.loc[context])
    if count == 0:
        continue
    axA.axhline(start - 0.5, color="black", linewidth=0.6)
    start += count

axA.axhline(len(merged) - 0.5, color="black", linewidth=0.6)

cbar = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
cbar.set_label("Standardized value")
add_panel_label(axA, "A")


# ------------------------------------------------------------
# Panel B: PCA projection
# ------------------------------------------------------------

for context in CONTEXT_ORDER:
    sub = merged[merged["ecological_context"] == context]
    if sub.empty:
        continue

    axB.scatter(
        sub["PC1"],
        sub["PC2"],
        s=45,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
        color=CONTEXT_COLORS.get(context, "gray"),
        label=f"{context} (n={len(sub)})",
    )

axB.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
axB.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
axB.set_title("PCA of patient-level TME features")
axB.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)")
axB.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)")
axB.legend(frameon=False, loc="best")
remove_spines(axB)
add_panel_label(axB, "B")


# ------------------------------------------------------------
# Panel C: Context prototype profiles
# ------------------------------------------------------------

x = np.arange(len(feature_labels))
width = 0.18

for idx, context in enumerate(CONTEXT_ORDER):
    values = prototype_df.loc[context, TME_FEATURES].to_numpy(dtype=float)
    if np.all(np.isnan(values)):
        continue

    offset = (idx - 1.5) * width
    axC.bar(
        x + offset,
        values,
        width=width,
        color=CONTEXT_COLORS.get(context, "gray"),
        edgecolor="black",
        linewidth=0.4,
        label=context,
    )

axC.axhline(0, color="black", linewidth=0.8)
axC.set_title("Context prototype TME profiles")
axC.set_ylabel("Mean standardized feature value")
axC.set_xticks(x)
axC.set_xticklabels(feature_labels, rotation=35, ha="right")
axC.legend(title="Context", frameon=False, ncol=2)
remove_spines(axC)
add_panel_label(axC, "C")


# ------------------------------------------------------------
# Panel D: Patient-to-prototype distance
# ------------------------------------------------------------

rng = np.random.default_rng(123)
ymax = 0.0

for idx, context in enumerate(CONTEXT_ORDER, start=1):
    vals = distance_df.loc[
        distance_df["ecological_context"] == context,
        "distance_to_context_prototype",
    ].to_numpy()

    if len(vals) == 0:
        continue

    jitter = rng.normal(0, 0.045, size=len(vals))
    ymax = max(ymax, float(np.max(vals)))

    axD.scatter(
        np.full(len(vals), idx) + jitter,
        vals,
        s=35,
        color=CONTEXT_COLORS.get(context, "gray"),
        edgecolor="black",
        linewidth=0.4,
        alpha=0.80,
        zorder=3,
    )

    median_val = float(np.median(vals))
    axD.plot(
        [idx - 0.18, idx + 0.18],
        [median_val, median_val],
        color="black",
        linewidth=1.4,
        zorder=4,
    )

    axD.text(
        idx,
        float(np.max(vals)) + 0.08,
        f"n={len(vals)}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

axD.set_xticks(np.arange(1, len(CONTEXT_ORDER) + 1))
axD.set_xticklabels(CONTEXT_ORDER)
axD.set_title("Distance to assigned context prototype")
axD.set_xlabel("Candidate ecological context")
axD.set_ylabel("Euclidean distance")
axD.set_ylim(bottom=0, top=ymax + 0.35 if ymax > 0 else 1)
remove_spines(axD)
add_panel_label(axD, "D")


# ============================================================
# Final formatting and save
# ============================================================

fig.suptitle(
    "Supplementary Figure S2. Data-driven discovery of candidate TME ecological contexts",
    y=0.98,
    fontweight="bold",
)

fig.tight_layout(rect=[0, 0, 1, 0.95])

png_path = OUT_DIR / f"{OUT_PREFIX}.png"
pdf_path = OUT_DIR / f"{OUT_PREFIX}.pdf"
svg_path = OUT_DIR / f"{OUT_PREFIX}.svg"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")

plt.close(fig)

print("\nSaved files:")
print(png_path)
print(pdf_path)
print(svg_path)

print("\nSaved supplementary tables:")
print(OUT_DIR / "SuppFigS2_context_prototype_table.csv")
print(OUT_DIR / "SuppFigS2_patient_to_prototype_distances.csv")
