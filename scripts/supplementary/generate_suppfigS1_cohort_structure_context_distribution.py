from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

BASE_DIR = Path(
    "/TME_OU_Branching"
)

POSSIBLE_INPUT_DIRS = [
    BASE_DIR,
    BASE_DIR / "data",
    BASE_DIR / "results",
    BASE_DIR / "Figure_3",
    BASE_DIR / "BMC_Bioinformatics",
    Path.cwd(),
]

OUT_DIR = BASE_DIR / "BMC_Bioinformatics" / "Supplementary_Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = "SuppFigS1_cohort_structure_context_distribution"

CONTEXT_ORDER = ["E1", "E2", "E3", "E4"]

CONTEXT_COLORS = {
    "E1": "#4C78A8",
    "E2": "#59A14F",
    "E3": "#F28E2B",
    "E4": "#B279A2",
}

DIAGNOSIS_ORDER = ["B-ALL", "T-ALL", "ETP-ALL", "MPAL", "AML"]


# ============================================================
# Helper functions
# ============================================================

def find_file(possible_names):
    """Find the first matching file across possible input directories."""
    for input_dir in POSSIBLE_INPUT_DIRS:
        for name in possible_names:
            candidate = input_dir / name
            if candidate.exists():
                print(f"Found: {candidate}")
                return candidate
    raise FileNotFoundError(
        "Could not find any of these files:\n"
        + "\n".join(possible_names)
        + "\nSearched in:\n"
        + "\n".join(str(x) for x in POSSIBLE_INPUT_DIRS)
    )


def pick_col(df, candidates, required=True):
    """Pick the first available column from candidate names."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def clean_diagnosis(x):
    """Standardize diagnosis labels for plotting."""
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip()

    replacements = {
        "B-cell acute lymphoblastic leukemia": "B-ALL",
        "B-cell acute lymphoblastic leukaemia": "B-ALL",
        "B-ALL": "B-ALL",
        "BALL": "B-ALL",
        "B_ALL": "B-ALL",
        "B ALL": "B-ALL",

        "T-cell acute lymphoblastic leukemia": "T-ALL",
        "T-cell acute lymphoblastic leukaemia": "T-ALL",
        "T-ALL": "T-ALL",
        "TALL": "T-ALL",
        "T_ALL": "T-ALL",
        "T ALL": "T-ALL",

        "Early T-cell precursor T-cell acute lymphoblastic leukemia": "ETP-ALL",
        "Early T-cell precursor T-cell acute lymphoblastic leukaemia": "ETP-ALL",
        "ETP-ALL": "ETP-ALL",
        "ETP_ALL": "ETP-ALL",
        "ETP ALL": "ETP-ALL",

        "Mixed phenotype acute leukemia": "MPAL",
        "Mixed phenotype acute leukaemia": "MPAL",
        "MPAL": "MPAL",

        "Acute myeloid leukemia": "AML",
        "Acute myeloid leukaemia": "AML",
        "AML": "AML",
    }

    return replacements.get(s, s)


def clean_context(x):
    """Standardize ecological-context labels."""
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip()

    numeric_to_context = {
        "0": "E1",
        "1": "E2",
        "2": "E3",
        "3": "E4",
    }

    if s in numeric_to_context:
        return numeric_to_context[s]

    if s.upper() in ["E1", "E2", "E3", "E4"]:
        return s.upper()

    return s


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

BASE_DIR = Path("/Users/seung-hwan.kim/Desktop/TME_OU_Branching")

patient_file = BASE_DIR / "patient_master_table.csv"
context_file = BASE_DIR / "Figure_3" / "patient_ecological_context_assignments.csv"
longitudinal_file = BASE_DIR / "results" / "longitudinal_support_revised_context.csv"

for f in [patient_file, context_file, longitudinal_file]:
    if not f.exists():
        raise FileNotFoundError(f"Required input file not found: {f}")

patient_df = pd.read_csv(patient_file)
context_df = pd.read_csv(context_file)
long_df = pd.read_csv(longitudinal_file)

print("\nUsing patient file:")
print(patient_file)

print("\nUsing context file:")
print(context_file)

print("\nUsing longitudinal file:")
print(longitudinal_file)

print("\nPatient table columns:")
print(patient_df.columns.tolist())

print("\nContext table columns:")
print(context_df.columns.tolist())

print("\nLongitudinal table columns:")
print(long_df.columns.tolist())


# ============================================================
# Identify columns
# ============================================================

patient_id_col = pick_col(
    patient_df,
    ["Patient_ID", "patient_id", "participant_id", "Participant_ID", "sample_id"],
)

diagnosis_col = pick_col(
    patient_df,
    ["diagnosis", "Diagnosis", "disease", "Disease", "diagnosis_simple"],
)

context_patient_id_col = pick_col(
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

long_patient_id_col = pick_col(
    long_df,
    ["Patient_ID", "patient_id", "participant_id", "Participant_ID", "sample_id"],
)


# ============================================================
# Merge patient metadata and context assignments
# ============================================================

patient_df = patient_df.copy()
context_df = context_df.copy()
long_df = long_df.copy()

patient_df[patient_id_col] = patient_df[patient_id_col].astype(str)
context_df[context_patient_id_col] = context_df[context_patient_id_col].astype(str)
long_df[long_patient_id_col] = long_df[long_patient_id_col].astype(str)

patient_df["diagnosis_clean"] = patient_df[diagnosis_col].apply(clean_diagnosis)
context_df["context_clean"] = context_df[context_col].apply(clean_context)

merged = patient_df.merge(
    context_df[[context_patient_id_col, "context_clean"]],
    left_on=patient_id_col,
    right_on=context_patient_id_col,
    how="left",
)

merged["context_clean"] = merged["context_clean"].fillna("Unknown")

# Keep one row per patient
merged = merged.drop_duplicates(subset=[patient_id_col]).copy()

# Longitudinal observation counts
if "n_longitudinal_rows" in long_df.columns:
    obs_counts = long_df[[long_patient_id_col, "n_longitudinal_rows"]].copy()
    obs_counts = obs_counts.rename(columns={"n_longitudinal_rows": "n_observations"})
else:
    obs_counts = (
        long_df.groupby(long_patient_id_col)
        .size()
        .reset_index(name="n_observations")
    )

merged = merged.merge(
    obs_counts,
    left_on=patient_id_col,
    right_on=long_patient_id_col,
    how="left",
)

merged["n_observations"] = merged["n_observations"].fillna(0).astype(int)

print("\nMerged cohort summary:")
print(merged[[patient_id_col, "diagnosis_clean", "context_clean", "n_observations"]].head())
print(f"\nNumber of unique patients: {merged[patient_id_col].nunique()}")


# ============================================================
# Prepare summaries
# ============================================================

# Diagnosis counts
diagnosis_counts = merged["diagnosis_clean"].value_counts()

diagnosis_order_present = [
    d for d in DIAGNOSIS_ORDER if d in diagnosis_counts.index
]
diagnosis_extra = [
    d for d in diagnosis_counts.index if d not in diagnosis_order_present
]
diagnosis_plot_order = diagnosis_order_present + diagnosis_extra

diagnosis_counts = diagnosis_counts.reindex(diagnosis_plot_order)

# Context counts
context_counts = merged["context_clean"].value_counts()

context_order_present = [
    c for c in CONTEXT_ORDER if c in context_counts.index
]
context_extra = [
    c for c in context_counts.index if c not in context_order_present
]
context_plot_order = context_order_present + context_extra

context_counts = context_counts.reindex(context_plot_order)

# Diagnosis-by-context table
ct = pd.crosstab(
    merged["diagnosis_clean"],
    merged["context_clean"],
)

ct = ct.reindex(index=diagnosis_plot_order, fill_value=0)
ct = ct.reindex(columns=context_plot_order, fill_value=0)

# Convert to row proportions for composition
ct_prop = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

# Observation-count distribution
obs_dist = merged["n_observations"].value_counts().sort_index()


# ============================================================
# Plot
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 13,
})

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8.8),
    constrained_layout=False,
)

axA, axB = axes[0, 0], axes[0, 1]
axC, axD = axes[1, 0], axes[1, 1]


# -------------------------
# Panel A: Diagnosis counts
# -------------------------
xA = np.arange(len(diagnosis_counts))
axA.bar(
    xA,
    diagnosis_counts.values,
    edgecolor="black",
    linewidth=0.6,
)
axA.set_xticks(xA)
axA.set_xticklabels(diagnosis_counts.index, rotation=30, ha="right")
axA.set_ylabel("Number of participants")
axA.set_title("Participants by diagnosis")
remove_spines(axA)
add_panel_label(axA, "A")

for i, v in enumerate(diagnosis_counts.values):
    axA.text(i, v + max(diagnosis_counts.values) * 0.02, str(int(v)),
             ha="center", va="bottom", fontsize=8)


# -------------------------
# Panel B: Context counts
# -------------------------
xB = np.arange(len(context_counts))
bar_colors_B = [
    CONTEXT_COLORS.get(c, "lightgray") for c in context_counts.index
]
axB.bar(
    xB,
    context_counts.values,
    color=bar_colors_B,
    edgecolor="black",
    linewidth=0.6,
)
axB.set_xticks(xB)
axB.set_xticklabels(context_counts.index)
axB.set_ylabel("Number of participants")
axB.set_title("Participants by candidate ecological context")
remove_spines(axB)
add_panel_label(axB, "B")

for i, v in enumerate(context_counts.values):
    axB.text(i, v + max(context_counts.values) * 0.02, str(int(v)),
             ha="center", va="bottom", fontsize=8)


# -------------------------
# Panel C: Diagnosis-by-context composition
# -------------------------
bottom = np.zeros(len(ct_prop.index))

for c in ct_prop.columns:
    values = ct_prop[c].values
    axC.barh(
        np.arange(len(ct_prop.index)),
        values,
        left=bottom,
        color=CONTEXT_COLORS.get(c, "lightgray"),
        edgecolor="white",
        linewidth=0.5,
        label=c,
    )
    bottom += values

axC.set_yticks(np.arange(len(ct_prop.index)))
axC.set_yticklabels(ct_prop.index)
axC.set_xlim(0, 1)
axC.set_xlabel("Proportion within diagnosis")
axC.set_title("Diagnosis-by-context composition")
axC.legend(
    title="Context",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
)
remove_spines(axC)
add_panel_label(axC, "C")


# -------------------------
# Panel D: Longitudinal observations per patient
# -------------------------
xD = np.arange(len(obs_dist))
axD.bar(
    xD,
    obs_dist.values,
    edgecolor="black",
    linewidth=0.6,
)
axD.set_xticks(xD)
axD.set_xticklabels(obs_dist.index.astype(str))
axD.set_xlabel("Longitudinal rows per participant")
axD.set_ylabel("Number of participants")
axD.set_title("Longitudinal support per participant")
remove_spines(axD)
add_panel_label(axD, "D")

for i, v in enumerate(obs_dist.values):
    axD.text(i, v + max(obs_dist.values) * 0.02, str(int(v)),
             ha="center", va="bottom", fontsize=8)


# ============================================================
# Final formatting
# ============================================================

fig.suptitle(
    "Supplementary Figure S1. Cohort structure and distribution of candidate ecological contexts",
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


# ============================================================
# Save summary table for reproducibility
# ============================================================

summary_path = OUT_DIR / "SuppFigS1_cohort_summary_table.csv"

summary_rows = []

for diagnosis, n in diagnosis_counts.items():
    summary_rows.append({
        "summary_type": "diagnosis_count",
        "category": diagnosis,
        "n": int(n),
    })

for context, n in context_counts.items():
    summary_rows.append({
        "summary_type": "context_count",
        "category": context,
        "n": int(n),
    })

for n_obs, n_patients in obs_dist.items():
    summary_rows.append({
        "summary_type": "longitudinal_observation_count",
        "category": int(n_obs),
        "n": int(n_patients),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(summary_path, index=False)

print(summary_path)
