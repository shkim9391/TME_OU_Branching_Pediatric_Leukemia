#fig4_full.py

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ------------------------------------------------------
# Paths
# ------------------------------------------------------
base_dir = "/results"

panel_A_path = os.path.join(base_dir, "violin_mu_by_ecotype.png")
panel_B_path = os.path.join(base_dir, "violin_mu_with_patients.png")
panel_C_path = os.path.join(base_dir, "violin_theta_log10_by_ecotype.png")
panel_D_path = os.path.join(base_dir, "violin_theta_log10_with_patients.png")
panel_E_path = os.path.join(base_dir, "scatter_mu_theta_ecotype_centroids.png")

out_path = os.path.join(base_dir, "Figure4_OU_dynamics_by_ecotype_fixedE.png")

# ------------------------------------------------------
# Load images
# ------------------------------------------------------
img_A = mpimg.imread(panel_A_path)
img_B = mpimg.imread(panel_B_path)
img_C = mpimg.imread(panel_C_path)
img_D = mpimg.imread(panel_D_path)
img_E = mpimg.imread(panel_E_path)

# ------------------------------------------------------
# Figure + Grid for A–D
#   Row 1: A | B | (empty)
#   Row 2: C | D | (empty)
#   Panel E will be added manually to the right, centered
# ------------------------------------------------------
fig = plt.figure(figsize=(14, 8))

gs = fig.add_gridspec(
    2, 3,
    height_ratios=[1, 1],
    width_ratios=[1, 1, 0.2],  # thin empty third column
    hspace=0.0005,               # tighter row spacing
    wspace=0.001
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

# Draw A–D
for ax, img, lab in zip(
    [axA, axB, axC, axD],
    [img_A, img_B, img_C, img_D],
    ["A", "B", "C", "D"]
):
    ax.imshow(img)
    ax.axis("off")
    ax.text(
        0.02, 0.97, lab,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
        ha="left"
    )

# ------------------------------------------------------
# Add Panel E manually, centered between B and D
# ------------------------------------------------------
posB = axB.get_position()
posD = axD.get_position()

# vertical center between B and D
center_y = 0.5 * (posB.y0 + posD.y1)
height_E = posB.height * 1.2           # a bit taller than B/D
bottom_E = center_y - height_E / 2

# place E to the right of B/D with a small gap
gap = 0.01
left_E = posB.x1 + gap
width_E = posB.width * 1.2

axE = fig.add_axes([left_E, bottom_E, width_E, height_E])
axE.imshow(img_E)
axE.axis("off")
axE.text(
    0.02, 0.98, "E",
    transform=axE.transAxes,
    fontsize=22,
    fontweight="bold",
    va="top",
    ha="left"
)

# ------------------------------------------------------
# Save
# ------------------------------------------------------
fig.savefig(out_path, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"Saved composite figure to:\n{out_path}")
