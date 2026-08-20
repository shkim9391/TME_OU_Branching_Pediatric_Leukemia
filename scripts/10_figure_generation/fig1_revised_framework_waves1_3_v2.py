from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl


# ============================================================
# Paths
# ============================================================

ROOT = Path("/TME_OU_Branching")
OUTDIR = ROOT / "Figure_1"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "Figure1_revised_analysis_framework_v2"


# ============================================================
# Palette
# ============================================================
# Restrained manuscript-friendly palette.
COLORS = {
    "data": "#DCEAF7",          # pale blue
    "process": "#E6F2E6",       # pale green
    "ecology": "#FFF0D9",       # pale orange
    "model": "#EAE3F4",         # pale purple
    "validation": "#FDE5E5",    # pale red
    "neutral": "#F2F2F2",       # light gray
    "accent_blue": "#4C78A8",
    "accent_green": "#59A14F",
    "accent_orange": "#F28E2B",
    "accent_purple": "#8E6BBE",
    "accent_red": "#E15759",
    "text": "#222222",
    "edge": "#333333",
}


# ============================================================
# Global style
# ============================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 13,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# ============================================================
# Drawing helpers
# ============================================================

def add_box(
    ax,
    xy,
    width,
    height,
    text,
    facecolor,
    edgecolor=COLORS["edge"],
    fontsize=10.5,
    fontweight="normal",
    lw=1.25,
    rounding=0.025,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=lw,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=COLORS["text"],
        wrap=True,
    )
    return patch


def add_arrow(ax, start, end, color=COLORS["edge"], lw=1.4):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        shrinkA=3,
        shrinkB=3,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def panel_label(ax, label):
    ax.text(
        -0.04,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=23,
        fontweight="bold",
        va="top",
        ha="left",
    )


def setup_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def footer(ax, text, y=0.04, fontsize=9.8, weight="normal"):
    ax.text(
        0.50,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["text"],
    )


# ============================================================
# Figure layout
# ============================================================

fig = plt.figure(figsize=(15.8, 10.2))
gs = fig.add_gridspec(
    2,
    2,
    left=0.055,
    right=0.975,
    top=0.90,
    bottom=0.06,
    wspace=0.11,
    hspace=0.23,
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

for ax in (axA, axB, axC, axD):
    setup_panel(ax)

fig.suptitle(
    "Figure 1. Analysis framework for ecological composition and longitudinal OU dynamics",
    fontsize=18,
    y=0.997,
)


# ============================================================
# Panel A — Harmonized single-cell representation
# ============================================================

panel_label(axA, "A")
axA.set_title("Harmonized single-cell ecological representation", pad=14)

# Inputs stacked
add_box(
    axA, (0.04, 0.68), 0.24, 0.16,
    "SCPCP000008\nsingle-cell data",
    COLORS["data"], fontsize=10.7, fontweight="bold"
)
add_box(
    axA, (0.04, 0.35), 0.24, 0.16,
    "SCPCP000022\nsingle-cell data",
    COLORS["data"], fontsize=10.7, fontweight="bold"
)

# Common mapper
add_box(
    axA, (0.37, 0.48), 0.26, 0.24,
    "Common cell-type\nmapping and\nannotation audit",
    COLORS["process"], fontsize=11, fontweight="bold"
)

# Final simplex
add_box(
    axA, (0.72, 0.48), 0.23, 0.24,
    "Primary immune\nsimplex\n\nTNK | B | Myeloid",
    COLORS["ecology"], fontsize=10.8, fontweight="bold"
)

add_arrow(axA, (0.28, 0.75), (0.37, 0.62), COLORS["accent_blue"])
add_arrow(axA, (0.28, 0.43), (0.37, 0.55), COLORS["accent_blue"])
add_arrow(axA, (0.63, 0.61), (0.72, 0.61), COLORS["accent_green"])

footer(
    axA,
    "113 regenerated participants  →  104 compositionally eligible\n"
    "TNK + B + Myeloid renormalized to sum to 1",
    y=0.12,
    fontsize=11.0,
)


# ============================================================
# Panel B — Ecological structure and sensitivity
# ============================================================

panel_label(axB, "B")
axB.set_title("Ecological structure, stability, and compositional sensitivity", pad=14)

# Representations
add_box(
    axB, (0.03, 0.68), 0.21, 0.15,
    "Raw composition",
    COLORS["ecology"], fontsize=10.8, fontweight="bold"
)
add_box(
    axB, (0.03, 0.34), 0.21, 0.15,
    "CLR sensitivity",
    COLORS["neutral"], fontsize=10.8, fontweight="bold"
)

# Shared assessment
add_box(
    axB, (0.34, 0.44), 0.27, 0.32,
    "Context assessment\n\n"
    "k = 2–6\n"
    "Cluster separation\n"
    "Project/diagnosis\nassociation\n"
    "Bootstrap stability",
    COLORS["process"], fontsize=9.8, fontweight="bold"
)

# Output context
add_box(
    axB, (0.70, 0.68), 0.26, 0.18,
    "Coarse k=2 contexts",
    COLORS["ecology"], fontsize=10.8, fontweight="bold"
)
add_box(
    axB, (0.70, 0.28), 0.26, 0.19,
    "Within-B-ALL sensitivity\n\n"
    "73 TNK-dominant\n"
    "26 B/Myeloid-shifted",
    COLORS["data"], fontsize=10.0
)

add_arrow(axB, (0.24, 0.75), (0.34, 0.63), COLORS["accent_orange"])
add_arrow(axB, (0.24, 0.415), (0.34, 0.56), COLORS["edge"])
add_arrow(axB, (0.61, 0.61), (0.70, 0.77), COLORS["accent_green"])
add_arrow(axB, (0.83, 0.68), (0.83, 0.47), COLORS["accent_blue"])

footer(
    axB,
    "Raw-versus-CLR ARI tests representation sensitivity.\n"
    "Contexts are descriptive coarse-graining, not fixed leukemia ecotypes.",
    y=0.07,
    fontsize=11.0,
)


# ============================================================
# Panel C — Corrected longitudinal model input
# ============================================================

panel_label(axC, "C")
axC.set_title("Corrected longitudinal x endpoint and model-input construction", pad=14)

add_box(
    axC, (0.03, 0.68), 0.23, 0.18,
    "ScPCA ecological\nparticipants",
    COLORS["data"], fontsize=10.6, fontweight="bold"
)
add_box(
    axC, (0.03, 0.29), 0.23, 0.18,
    "Longitudinal\nP1–P100 cohort",
    COLORS["data"], fontsize=10.6, fontweight="bold"
)

add_box(
    axC, (0.35, 0.43), 0.27, 0.27,
    "Curated ID crosswalk\nParticipant-flow audit",
    COLORS["process"], fontsize=10.8, fontweight="bold"
)

add_box(
    axC, (0.70, 0.43), 0.25, 0.27,
    "Primary endpoint: x\n\n"
    "Resolve same-time\nobservations\n"
    "Build x_prev, x, Δt",
    COLORS["validation"], fontsize=10.2, fontweight="bold"
)

add_arrow(axC, (0.26, 0.75), (0.35, 0.63), COLORS["accent_blue"])
add_arrow(axC, (0.26, 0.38), (0.35, 0.51), COLORS["accent_blue"])
add_arrow(axC, (0.62, 0.57), (0.70, 0.57), COLORS["accent_green"])

footer(
    axC,
    "Final primary cohort: 34 participants  |  241 x observations  \n|  207 positive-time transitions\n"
    "Irregular sampling retained explicitly for the OU transition likelihood",
    y=0.07,
    fontsize=11.0,
)


# ============================================================
# Panel D — OU inference and validation
# ============================================================

panel_label(axD, "D")
axD.set_title("OU inference, participant-level validation, and robustness", pad=14)

# Top row, left-to-right
add_box(
    axD, (0.02, 0.68), 0.23, 0.20,
    "Irregular-time OU\ntransition likelihood",
    COLORS["model"], fontsize=10.3, fontweight="bold"
)
add_box(
    axD, (0.33, 0.68), 0.28, 0.20,
    "Ecological modulation\nContinuous composition\nvs coarse context\nμ and log θ",
    COLORS["ecology"], fontsize=10.3, fontweight="bold"
)
add_box(
    axD, (0.69, 0.68), 0.27, 0.20,
    "Participant-level LOPO\nComplete model\nrefitting",
    COLORS["validation"], fontsize=10.3, fontweight="bold"
)

add_arrow(axD, (0.25, 0.775), (0.33, 0.775), COLORS["accent_purple"])
add_arrow(axD, (0.61, 0.775), (0.69, 0.775), COLORS["accent_red"])

# Bottom row
add_box(
    axD, (0.02, 0.28), 0.23, 0.30,
    "Matched OU ablations\n\n"
    "OU only\n"
    "Continuous ecology\n"
    "Coarse context\n"
    "Ecology + context\n"
    "Shuffled context",
    COLORS["neutral"], fontsize=9.4
)
add_box(
    axD, (0.33, 0.28), 0.28, 0.30,
    "Endpoint / prior robustness\n\n"
    "Logit-transformed x\n"
    "Ordinal n\n"
    "Prior sensitivity",
    COLORS["process"], fontsize=9.8
)
add_box(
    axD, (0.69, 0.28), 0.27, 0.30,
    "Model checking\n\n"
    "PPC calibration\n"
    "Influential transitions\n"
    "Parameter recovery",
    COLORS["data"], fontsize=9.8
)

# Vertical arrows only: no crossing
add_arrow(axD, (0.14, 0.68), (0.14, 0.58), COLORS["accent_purple"])
add_arrow(axD, (0.47, 0.68), (0.47, 0.58), COLORS["accent_green"])
add_arrow(axD, (0.83, 0.68), (0.83, 0.58), COLORS["accent_red"])

footer(
    axD,
    "Primary question: does continuous ecological composition\nadd longitudinal information\n"
    "beyond coarse ecological-context labels?",
    y=0.07,
    fontsize=11.0,
    weight="bold",
)


# ============================================================
# Save
# ============================================================

fig.savefig(f"{OUTBASE}.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUTBASE}.tiff", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUTBASE}.pdf", bbox_inches="tight")

print(f"[SAVED] {OUTBASE}.png")
print(f"[SAVED] {OUTBASE}.tiff")
print(f"[SAVED] {OUTBASE}.pdf")

plt.show()
