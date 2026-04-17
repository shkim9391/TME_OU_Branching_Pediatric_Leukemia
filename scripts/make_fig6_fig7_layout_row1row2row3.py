#python3 make_fig6_fig7_layout_row1row2row3.py --indir figures --outdir figures --dpi 600

Composite Figures:
  - Figure 6 (A–D) laid out as:
      Row 1: 6A-left | 6A-right
      Row 2: 6B      | 6C
      Row 3: 6D spans both columns
  - Figure 7: sensitivity panel (E) only

Expected inputs (in --indir, default: figures):
  - Fig6A_cohort_summary.png
  - Fig6B_schematic.png
  - Fig6C_posteriors_by_ecotype.png
  - Fig6D_example_trajectories.png
  - Fig6E_k_sensitivity.png           (used to create Fig7)

Outputs (to --outdir, default: figures):
  - Fig6_composite.png
  - Fig6_composite.pdf
  - Fig7_k_sensitivity.png
  - Fig7_k_sensitivity.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def load_img(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required panel: {path}")
    return mpimg.imread(path)


def _to_float01(img: np.ndarray) -> np.ndarray:
    """Ensure image is float in [0,1] (supports uint8 or float input)."""
    arr = img
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        # If already 0..255 floats, scale down.
        if arr.max() > 1.5:
            arr = arr / 255.0
    return arr


def crop_white(img, thr=0.99, pad=8):
    """
    Crop white-ish borders from an RGB/RGBA image.
    thr: whiteness threshold (0..1). Higher = stricter 'white'.
    pad: pixels of padding kept around content.
    """
    arr = _to_float01(img)
    rgb = arr[..., :3]
    mask = np.any(rgb < thr, axis=-1)  # pixels that are NOT white-ish
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img  # fallback

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.shape[0] - 1, y1 + pad)
    x1 = min(img.shape[1] - 1, x1 + pad)

    return img[y0 : y1 + 1, x0 : x1 + 1]


def split_two_panels_by_whitespace(img, thr=0.99, search_frac=(0.25, 0.75),
                                   max_nonwhite_frac=0.005, min_run=20):
    """
    Split a 2-panel horizontal composite into left/right panels by finding a vertical
    whitespace gutter near the middle.

    Parameters
    ----------
    thr : float
        Whiteness threshold used to define "non-white" pixels.
    search_frac : (float, float)
        Only search for the gutter between these fractions of width (avoid margins).
    max_nonwhite_frac : float
        A column is considered "gutter" if fraction of non-white pixels is below this.
    min_run : int
        Minimum contiguous gutter width (columns).

    Returns
    -------
    left_img, right_img : np.ndarray
    """
    arr = _to_float01(img)
    rgb = arr[..., :3]
    nonwhite = np.any(rgb < thr, axis=-1)          # True where content exists
    col_frac = nonwhite.mean(axis=0)               # fraction of content per column

    W = col_frac.shape[0]
    lo = int(search_frac[0] * W)
    hi = int(search_frac[1] * W)

    gutter_cols = col_frac[lo:hi] < max_nonwhite_frac

    best = None  # (run_len, -dist_to_mid, start, end)
    run_start = None
    for i, is_gutter in enumerate(gutter_cols):
        if is_gutter and run_start is None:
            run_start = i
        if (not is_gutter or i == len(gutter_cols) - 1) and run_start is not None:
            run_end = i if not is_gutter else i + 1
            run_len = run_end - run_start
            if run_len >= min_run:
                center = (run_start + run_end) / 2 + lo
                score = (run_len, -abs(center - W / 2), run_start + lo, run_end + lo)
                if best is None or score[:2] > best[:2]:
                    best = score
            run_start = None

    if best is None:
        # fallback: choose the least-content column in the search region
        split_col = lo + int(np.argmin(col_frac[lo:hi]))
    else:
        _, _, s, e = best
        split_col = int((s + e) / 2)

    left = img[:, :split_col, :]
    right = img[:, split_col:, :]
    return left, right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="figures", help="Folder containing panel PNGs")
    ap.add_argument("--outdir", type=str, default="figures", help="Folder to write composites")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--thr", type=float, default=0.99, help="White threshold for cropping (0..1)")
    ap.add_argument("--pad", type=int, default=8, help="Padding (px) kept around cropped content")
    args = ap.parse_args()

    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    A = crop_white(load_img(indir / "Fig6A_cohort_summary.png"), thr=args.thr, pad=args.pad + 2)
    B = crop_white(load_img(indir / "Fig6B_schematic.png"), thr=args.thr, pad=args.pad + 18)
    C = crop_white(load_img(indir / "Fig6C_posteriors_by_ecotype.png"), thr=args.thr, pad=args.pad + 6)
    D = crop_white(load_img(indir / "Fig6D_example_trajectories.png"), thr=args.thr, pad=args.pad + 6)
    E = crop_white(load_img(indir / "Fig6E_k_sensitivity.png"), thr=args.thr, pad=args.pad)

    # Optional: extra trim for 6B only (reduce gap below 6B)
    h = B.shape[0]
    cut = int(0.01 * h)  # tune 0.05–0.15 if needed
    B = B[: h - cut, :, :]

    # -------------------------
    # Figure 6 composite: A–D
    # Layout:
    #   Row 1: A spans full width
    #   Row 2: B | C   (C wider)
    #   Row 3: D spans full width
    # -------------------------
    fig6 = plt.figure(figsize=(12, 13))
    
    # 3 rows, 1 column container
    gs6 = fig6.add_gridspec(
        3, 1,
        height_ratios=[1.05, 1.05, 1.65],
        hspace=0.03
    )
    
    # Row 1: A full width
    axA = fig6.add_subplot(gs6[0, 0])
    axA.imshow(A)
    axA.axis("off")
    
    # Row 2: B | C (C wider)
    gs_r2 = gs6[1, 0].subgridspec(1, 2, wspace=0.05, width_ratios=[1.00, 1.60])
    axB = fig6.add_subplot(gs_r2[0, 0])
    axB.imshow(B)
    axB.axis("off")
    
    # shrink inside its grid cell (tune these)
    pos = axB.get_position()
    shrink_w = 0.85
    shrink_h = 0.85
    y_shift = 0.03  # fraction of the original cell height; try 0.05–0.20
    x_shift = 0.00  # optional
    
    axB.set_position([
        pos.x0 + (1 - shrink_w) * pos.width / 2 + x_shift * pos.width,
        pos.y0 + (1 - shrink_h) * pos.height / 2 + y_shift * pos.height,
        pos.width * shrink_w,
        pos.height * shrink_h
    ])
    axC = fig6.add_subplot(gs_r2[0, 1]); axC.imshow(C); axC.axis("off")
    
    # Row 3: D full width
    axD = fig6.add_subplot(gs6[2, 0])
    axD.imshow(D)
    axD.axis("off")
    
    out6_png = outdir / "Fig6_composite.png"
    out6_pdf = outdir / "Fig6_composite.pdf"
    fig6.savefig(out6_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    fig6.savefig(out6_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig6)

    # -------------------------
    # Figure 7: E panel only
    # -------------------------
    fig7 = plt.figure(figsize=(12, 6))
    axE = fig7.add_subplot(1, 1, 1)
    axE.imshow(E)
    axE.axis("off")

    out7_png = outdir / "Fig7_k_sensitivity.png"
    out7_pdf = outdir / "Fig7_k_sensitivity.pdf"
    fig7.savefig(out7_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    fig7.savefig(out7_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig7)

    print(f"[OK] Wrote {out6_png}")
    print(f"[OK] Wrote {out6_pdf}")
    print(f"[OK] Wrote {out7_png}")
    print(f"[OK] Wrote {out7_pdf}")


if __name__ == "__main__":
    main()
