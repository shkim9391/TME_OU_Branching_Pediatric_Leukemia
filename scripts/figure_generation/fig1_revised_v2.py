import os
import warnings
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = "/TME_OU_Branching/Figure_1"

FIG_D_IN = os.path.join(BASE_DIR, "Fig1D_regenerated.png")

FIG_D_PNG = os.path.join(BASE_DIR, "Fig1D_labeled.png")
FIG_D_TIFF = os.path.join(BASE_DIR, "Fig1D_labeled_600dpi.tiff")

FIG1_PNG = os.path.join(BASE_DIR, "Figure1_revised_v2.png")
FIG1_TIFF = os.path.join(BASE_DIR, "Figure1_revised_v2_600dpi.tiff")


# =============================================================================
# Global style
# =============================================================================

FULL_FIGSIZE = (14.5, 9.2)
FULL_DPI = 600

PANEL_LABEL_SIZE = 28
PANEL_LABEL_WEIGHT = "bold"

BLUE = "#0072B2"
DARK_BLUE = "#005B8F"
LIGHT_BLUE = "#EAF4FB"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#7B3294"
GRAY = "#555555"
LIGHT_GRAY = "#F2F2F2"
DARK_GRAY = "#222222"

BG = (255, 255, 255)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# =============================================================================
# PIL helpers
# =============================================================================

def to_rgb_on_white(im: Image.Image) -> Image.Image:
    """Flatten alpha onto white background and return RGB image."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, BG + (255,))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return im.convert("RGB")


def whiten_edge_margin(im: Image.Image, margin: int = 4, thresh: int = 245) -> Image.Image:
    """Force near-edge non-white pixels to white to remove faint borders."""
    if margin <= 0:
        return im

    im = im.copy()
    w, h = im.size
    pix = im.load()

    def whiten_if_needed(x: int, y: int) -> None:
        r, g, b = pix[x, y]
        if (r < thresh) or (g < thresh) or (b < thresh):
            pix[x, y] = BG

    for y in range(0, min(margin, h)):
        for x in range(w):
            whiten_if_needed(x, y)

    for y in range(max(0, h - margin), h):
        for x in range(w):
            whiten_if_needed(x, y)

    for x in range(0, min(margin, w)):
        for y in range(h):
            whiten_if_needed(x, y)

    for x in range(max(0, w - margin), w):
        for y in range(h):
            whiten_if_needed(x, y)

    return im


def remove_existing_panel_label(im: Image.Image, width_frac: float = 0.18, height_frac: float = 0.16) -> Image.Image:
    """
    White out the upper-left corner of an imported panel to remove an existing
    panel label, e.g. a small pre-existing 'D'. This avoids double-labeling.
    """
    im = im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    x2 = int(w * width_frac)
    y2 = int(h * height_frac)
    draw.rectangle((0, 0, x2, y2), fill=BG)
    return im


def load_bold_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]

    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass

    return ImageFont.load_default()


def save_standalone_panel_d() -> None:
    """Create standalone labeled Figure 1D from the PPC input image."""
    if not os.path.exists(FIG_D_IN):
        raise FileNotFoundError(f"Cannot find Panel D input image: {FIG_D_IN}")

    img = Image.open(FIG_D_IN)
    img = to_rgb_on_white(img)
    img = whiten_edge_margin(img, margin=4, thresh=245)
    img = remove_existing_panel_label(img, width_frac=0.18, height_frac=0.16)

    draw = ImageDraw.Draw(img)
    font = load_bold_font(size=90)

    draw.text((25, 10), "D", fill="black", font=font)

    img.save(FIG_D_PNG)
    img.save(FIG_D_TIFF, dpi=(600, 600), compression="tiff_lzw")

    print("Saved standalone Panel D:")
    print(" PNG :", FIG_D_PNG)
    print(" TIFF:", FIG_D_TIFF)


# =============================================================================
# Matplotlib helpers
# =============================================================================

def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight=PANEL_LABEL_WEIGHT,
        va="top",
        ha="left",
        color="black",
        clip_on=False,
    )


def add_box(
    ax,
    xy: Tuple[float, float],
    width: float,
    height: float,
    text: str,
    fc: str = BLUE,
    ec: str = "none",
    text_color: str = "white",
    fontsize: int = 11,
    lw: float = 1.5,
    radius: float = 0.035,
    weight: str = "bold",
) -> FancyBboxPatch:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        transform=ax.transAxes,
        fc=fc,
        ec=ec,
        lw=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.05,
        clip_on=False,
    )
    return patch


def add_arrow(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = GRAY,
    lw: float = 1.8,
    mutation_scale: int = 16,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="->",
        mutation_scale=mutation_scale,
        lw=lw,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


# =============================================================================
# Panel A
# =============================================================================

def draw_panel_a(ax) -> None:
    ax.set_xlim(-0.55, 4.9)
    ax.set_ylim(-2.25, 2.35)
    ax.axis("off")
    add_panel_label(ax, "A", x=-0.07, y=1.09)

    # Axes
    ax.annotate(
        "",
        xy=(0, 2.08),
        xytext=(0, -1.72),
        arrowprops=dict(arrowstyle="-", lw=2.6, color="black"),
    )
    ax.annotate(
        "",
        xy=(4.55, -1.72),
        xytext=(0, -1.72),
        arrowprops=dict(arrowstyle="->", lw=2.2, color="black"),
    )

    ax.text(-0.22, 2.08, "Trait\n$Y(t)$", ha="right", va="top", fontsize=10)
    ax.text(4.65, -1.82, "time", ha="right", va="top", fontsize=10)

    # Local attractor
    mu = 0.0
    ax.hlines(mu, 0.0, 4.25, colors="black", linestyles=(0, (7, 5)), lw=2.0)
    ax.text(
        -0.18,
        mu,
        "$\\mu_i$\nlocal\nattractor",
        ha="right",
        va="center",
        fontsize=10,
    )

    # OU-like trajectories
    t = np.linspace(0, 4.0, 180)
    starts = [1.8, 1.1, 0.55, -1.35]
    thetas = [0.58, 0.85, 1.05, 0.80]
    colors = [ORANGE, GREEN, BLUE, PURPLE]

    end_offsets = [0.16, 0.04, -0.08, -0.18]

    for idx, (y0, theta, col, off) in enumerate(zip(starts, thetas, colors, end_offsets), start=1):
        y = mu + (y0 - mu) * np.exp(-theta * t)
        y += 0.04 * np.sin(2.1 * t + idx) * np.exp(-0.35 * t)

        ax.plot(t, y, lw=3, color=col, alpha=0.95)
        ax.scatter([t[-1]], [y[-1]], s=175, color=col, edgecolor="white", lw=1.1, zorder=5)
        ax.text(
            t[-1] + 0.14,
            y[-1] + off,
            f"$Y_{idx}(t)$",
            ha="left",
            va="center",
            fontsize=8.5,
            color=DARK_GRAY,
        )

    ax.annotate(
        "effective\nmean reversion\n$\\theta_i$",
        xy=(1.25, 0.45),
        xytext=(2.35, 1.38),
        ha="center",
        va="center",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", lw=1.6, color=DARK_GRAY),
    )

    # Shorter note, moved below axis and away from time label
    ax.text(
        0.05,
        -2.30,
        "OU-like dynamics summarize fluctuation\naround an inferred local attractor.",
        ha="left",
        va="bottom",
        fontsize=10.0,
        color=GRAY,
    )


# =============================================================================
# Panel B
# =============================================================================

def draw_panel_b(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "B", x=-0.02, y=1.09)

    # Main workflow boxes
    add_box(
        ax,
        (0.05, 0.76),
        0.34,
        0.15,
        "ScPCA single-cell\nannotations",
        fc=BLUE,
        fontsize=10.0,
    )
    add_box(
        ax,
        (0.60, 0.76),
        0.33,
        0.15,
        "sample/participant\nTME features",
        fc=BLUE,
        fontsize=10.0,
    )
    add_box(
        ax,
        (0.60, 0.49),
        0.33,
        0.15,
        "candidate ecological\ncontexts",
        fc=BLUE,
        fontsize=10.0,
    )
    add_box(
        ax,
        (0.05, 0.49),
        0.34,
        0.15,
        "ecological-context\nmodulated OU model",
        fc=BLUE,
        fontsize=10.0,
    )

    # Diagnosis-aware evaluation box moved below and kept separate
    add_box(
        ax,
        (0.36, 0.27),
        0.27,
        0.12,
        "diagnosis,\nsubdiagnosis,\nproject",
        fc=LIGHT_BLUE,
        ec=BLUE,
        text_color=DARK_BLUE,
        fontsize=9.6,
        lw=1.4,
        weight="normal",
    )

    # Bottom inference box with line break to prevent clipping
    add_box(
        ax,
        (0.08, 0.04),
        0.84,
        0.14,
        "Bayesian inference\nposteriors, PPC, calibration, model comparison",
        fc=BLUE,
        fontsize=10.0,
    )

    # Arrows
    add_arrow(ax, (0.39, 0.835), (0.60, 0.835))
    add_arrow(ax, (0.765, 0.76), (0.765, 0.64))
    add_arrow(ax, (0.60, 0.565), (0.39, 0.565))
    add_arrow(ax, (0.22, 0.49), (0.22, 0.18))
    add_arrow(ax, (0.765, 0.49), (0.765, 0.18))
    add_arrow(ax, (0.50, 0.27), (0.50, 0.18))

    # Parameter note placed between model and covariate box
    ax.text(
        0.50,
        0.435,
        "estimated parameters: $\\mu_i$, $\\theta_i$, shared $\\sigma_{proc}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=DARK_GRAY,
        clip_on=False,
    )


# =============================================================================
# Panel C
# =============================================================================

def draw_panel_c(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "C", x=-0.07, y=1.03)

    # Ecological context circle
    circle = Circle(
        (0.20, 0.64),
        radius=0.135,
        transform=ax.transAxes,
        fc=BLUE,
        ec="none",
    )
    ax.add_patch(circle)
    ax.text(
        0.20,
        0.64,
        "candidate\necological\ncontext",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=10.0,
        fontweight="bold",
        linespacing=1.05,
    )

    # Diagnosis-aware covariate box
    add_box(
        ax,
        (0.07, 0.23),
        0.27,
        0.13,
        "diagnosis-aware\ncovariates",
        fc=LIGHT_BLUE,
        ec=BLUE,
        text_color=DARK_BLUE,
        fontsize=10.0,
        lw=1.5,
        weight="normal",
    )

    # Main hierarchy box
    add_box(
        ax,
        (0.54, 0.43),
        0.38,
        0.25,
        "hierarchical priors\non $\\mu_i$ and $\\log \\theta_i$\n\npartial pooling",
        fc=BLUE,
        fontsize=10.0,
    )

    # Output parameter box
    add_box(
        ax,
        (0.55, 0.12),
        0.36,
        0.13,
        "$\\mu_i$: local attractor\n$\\theta_i$: effective mean reversion",
        fc=LIGHT_GRAY,
        ec=GRAY,
        text_color=DARK_GRAY,
        fontsize=9.3,
        lw=1.2,
        weight="normal",
    )

    # Arrows
    add_arrow(ax, (0.335, 0.64), (0.54, 0.57))
    add_arrow(ax, (0.34, 0.295), (0.54, 0.49))
    add_arrow(ax, (0.73, 0.43), (0.73, 0.25))

    # Move label above the arrow, not on top of it
    ax.text(
        0.43,
        0.67,
        "modulates",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.0,
        color=DARK_GRAY,
    )


# =============================================================================
# Panel D
# =============================================================================

def load_panel_d_image() -> Image.Image:
    if not os.path.exists(FIG_D_IN):
        raise FileNotFoundError(f"Cannot find Panel D input image: {FIG_D_IN}")

    img = Image.open(FIG_D_IN)
    img = to_rgb_on_white(img)
    img = whiten_edge_margin(img, margin=4, thresh=245)

    # Important: remove old small panel label from imported PPC image
    img = remove_existing_panel_label(img, width_frac=0.07, height_frac=0.07)

    return img


def draw_panel_d(ax) -> None:
    ax.axis("off")
    img = load_panel_d_image()
    ax.imshow(img)
    add_panel_label(ax, "D", x=-0.18, y=1.03)


# =============================================================================
# Full figure assembly
# =============================================================================

def make_full_figure_1() -> None:
    fig = plt.figure(figsize=FULL_FIGSIZE, facecolor="white")

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.02, 1.18],
        height_ratios=[1.0, 1.0],
        wspace=0.20,
        hspace=0.20,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)

    fig.savefig(
        FIG1_PNG,
        dpi=FULL_DPI,
        bbox_inches="tight",
        pad_inches=0.10,
    )
    plt.close(fig)

    img = Image.open(FIG1_PNG).convert("RGB")
    img.save(FIG1_TIFF, dpi=(FULL_DPI, FULL_DPI), compression="tiff_lzw")

    print("Saved full Figure 1:")
    print(" PNG :", FIG1_PNG)
    print(" TIFF:", FIG1_TIFF)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)

    save_standalone_panel_d()
    make_full_figure_1()

    print("\nDone.")
    print("Check that:")
    print("  1. Panel B bottom text is not clipped.")
    print("  2. Panel D has only one label.")
    print("  3. Panel A bottom note does not overlap the time axis.")
    print("  4. Panel C arrow text is not crowded.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
