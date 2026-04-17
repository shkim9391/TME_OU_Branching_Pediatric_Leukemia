# make_fig6_main_and_SI_composites.py

from __future__ import annotations

"""
Make a cleaner MAIN Figure 6 composite + two SI pages.

MAIN (Figure 6):
  Row 1: 6A (left) + 6B (right)
  Row 2: 6C (full width)
  Row 3: 6D (TOP ROW ONLY; 2 plots) (full width)

SI:
  Page 1: Annotated 6A (full) + 6B (full 4 exemplars)
  Page 2: 6E (full k-grid) (full page)

Expected PNG inputs in --indir (default: Figure6):
  - Fig6A_cohort_summary.png
  - Fig6B.png
  - Fig6C_posteriors_by_ecotype.png
  - Fig6D_example_trajectories.png
  - Fig6E_k_sensitivity.png

Optional (if you have separate annotated 6A):
  - Fig6A_cohort_summary_annotated.png  (set --a_si)

Outputs in --outdir (default: Figure6):
  - Fig6_main.png / Fig6_main.pdf
  - FigS6_page1.png / FigS6_page1.pdf
  - FigS6E_page2.png / FigS6E_page2.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_img(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return mpimg.imread(path)


def crop_frac(img: np.ndarray, left=0.0, right=1.0, top=0.0, bottom=1.0) -> np.ndarray:
    """
    Crop an image using fractional bounds.
    top/bottom are fractions from top of image (0..1).
    """
    h, w = img.shape[0], img.shape[1]
    x0 = int(round(w * left))
    x1 = int(round(w * right))
    y0 = int(round(h * top))
    y1 = int(round(h * bottom))
    return img[y0:y1, x0:x1]


def place(ax, img: np.ndarray):
    ax.imshow(img)
    ax.axis("off")


def save_fig(fig, out_png: Path, out_pdf: Path, dpi=300):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_main(A, B_full, C, D, outdir: Path, dpi: int,
              b_top=0.0, b_bottom=0.54):
    B = crop_frac(B_full, top=b_top, bottom=b_bottom)

    # crop a bit of whitespace from C and D to tighten the stack
    C = crop_frac(C, top=0.00, bottom=0.99)
    D = crop_frac(D, top=0.04, bottom=0.99)
    A = crop_frac(A, left=0.00, right=0.95, top=0.02, bottom=0.99)

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.00, 1.05, 1.10],
        hspace=0.06,   # tighter vertical spacing
        wspace=0.10
    )

    axA = fig.add_subplot(gs[0, 0]); place(axA, A)
    axD = fig.add_subplot(gs[0, 1]); place(axD, D)

    axC = fig.add_subplot(gs[1, :]); place(axC, C)
    axB = fig.add_subplot(gs[2, :]); place(axB, B)

    out_png = outdir / "Fig6_main.png"
    out_pdf = outdir / "Fig6_main.pdf"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_SI_page1(A_annot, B_full, outdir: Path, dpi: int):
    """
    SI Page 1:
      Row 1: Annotated 6A (full width)
      Row 2: Full 6B (4 exemplars) (full width)
    """
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.06)

    ax1 = fig.add_subplot(gs[0, 0]); place(ax1, A_annot)
    ax2 = fig.add_subplot(gs[1, 0]); place(ax2, B_full)

    save_fig(fig,
             outdir / "FigS6_page1.png",
             outdir / "FigS6_page1.pdf",
             dpi=dpi)


def make_SI_page2(E, outdir: Path, dpi: int):
    """
    SI Page 2:
      6E full grid (full page)
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    place(ax, E)

    save_fig(fig,
             outdir / "FigS6E_page2.png",
             outdir / "FigS6E_page2.pdf",
             dpi=dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="Figure_6", help="Folder containing panel PNGs")
    ap.add_argument("--outdir", type=str, default="Figure_6", help="Folder to write composites")
    ap.add_argument("--dpi", type=int, default=300)

    # Filenames (override if yours differ)
    ap.add_argument("--a_main", type=str, default="Fig6A_cohort_summary.png")
    ap.add_argument("--a_si", type=str, default="", help="Annotated 6A for SI (optional). If empty, uses --a_main.")
    ap.add_argument("--b", type=str, default="Fig6B_example_trajectories.png")
    ap.add_argument("--c", type=str, default="Fig6C_posteriors_by_ecotype.png")
    ap.add_argument("--d", type=str, default="Fig6D.png")
    ap.add_argument("--e", type=str, default="Fig6E_k_sensitivity.png")

    # Crop settings for MAIN B (top-row only)
    ap.add_argument("--b_crop_top", type=float, default=0.0, help="Fraction from top (0..1)")
    ap.add_argument("--b_crop_bottom", type=float, default=0.54, help="Fraction from top (0..1)")

    args = ap.parse_args()

    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    A = load_img(indir / args.a_main)
    B_full = load_img(indir / args.b)
    C = load_img(indir / args.c)
    D = load_img(indir / args.d)
    E = load_img(indir / args.e)

    # SI 6A (annotated) – fallback to main if not provided
    a_si_name = args.a_si.strip() if args.a_si.strip() else args.a_main
    A_si = load_img(indir / a_si_name)

    # MAIN composite
    make_main(
        A=A, B_full=B_full, C=C, D=D,
        outdir=outdir, dpi=args.dpi,
        b_top=args.b_crop_top, b_bottom=args.b_crop_bottom
    )

    # SI Page 1 (annotated A + full B)
    make_SI_page1(A_annot=A_si, B_full=B_full, outdir=outdir, dpi=args.dpi)

    # SI Page 2 (E only)
    make_SI_page2(E=E, outdir=outdir, dpi=args.dpi)

    print(f"[OK] Wrote:\n"
          f"  {outdir / 'Fig6_main.png'}\n"
          f"  {outdir / 'Fig6_main.pdf'}\n"
          f"  {outdir / 'FigS6_page1.png'}\n"
          f"  {outdir / 'FigS6_page1.pdf'}\n"
          f"  {outdir / 'FigS6E_page2.png'}\n"
          f"  {outdir / 'FigS6E_page2.pdf'}")


if __name__ == "__main__":
    main()
