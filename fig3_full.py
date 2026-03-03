#fig3_full.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    import umap
except ImportError as e:
    raise ImportError(
        "The 'umap-learn' package is required for this script.\n"
        "Install it via: pip install umap-learn"
    ) from e

# ------------------------------------------------------
# Paths and data
# ------------------------------------------------------
input_path = "/covariate_matrix.csv"
outdir = "/Figure_3"
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(input_path)

# ------------------------------------------------------
# TME features
# ------------------------------------------------------
rename_map = {
    "frac_T_given_known_z": "T",
    "frac_B_given_known_z": "B",
    "frac_myeloid_given_known_z": "Myeloid",
    "frac_NK_given_known_z": "NK",
    "frac_stromal_given_known_z": "Stromal",
    "frac_unknown_z": "Unknown",
}
df = df.rename(columns=rename_map)

# ------------------------------------------------------
# Attach true diagnosis labels (avoid drop_first pitfalls)
# ------------------------------------------------------
master_path = "/patient_master_table.csv"
meta = pd.read_csv(master_path, usecols=["Patient_ID", "diagnosis", "subdiagnosis"])

df = df.merge(meta, on="Patient_ID", how="left")

# ------------------------------------------------------
# Define TME feature columns (after renaming)
# ------------------------------------------------------
tme_features = ["T", "B", "Myeloid", "NK", "Stromal", "Unknown"]

# Keep only rows with complete TME features
df_tme = df.dropna(subset=tme_features).copy()

# Use true diagnosis strings for plotting
df_tme["diagnosis"] = df_tme["diagnosis"].fillna("Unknown").astype(str).str.strip()
diagnosis = df_tme["diagnosis"]

# ------------------------------------------------------
# Define immune ecotypes by KMeans on TME features
# ------------------------------------------------------
X = df_tme[tme_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_ecotypes = 4
kmeans = KMeans(n_clusters=n_ecotypes, random_state=123, n_init=50)
eco_labels = kmeans.fit_predict(X_scaled)
ecotype = (eco_labels + 1).astype(str)
df_tme["immune_ecotype"] = ecotype

# ------------------------------------------------------
# Unique categories and color maps
# ------------------------------------------------------
diag_unique = sorted(pd.unique(diagnosis))
eco_unique = sorted(pd.unique(ecotype))

diag_cmap = plt.get_cmap("tab10")
eco_cmap = plt.get_cmap("Set2")

diag_color_map = {d: diag_cmap(i % diag_cmap.N) for i, d in enumerate(diag_unique)}
eco_color_map = {e: eco_cmap(i % eco_cmap.N) for i, e in enumerate(eco_unique)}

# ------------------------------------------------------
# PCA & UMAP
# ------------------------------------------------------
pca = PCA(n_components=2, random_state=123)
X_pca = pca.fit_transform(X_scaled)
expl_var = pca.explained_variance_ratio_ * 100

umap_model = umap.UMAP(
    n_neighbors=10,
    min_dist=0.3,
    n_components=2,
    metric="euclidean",
    random_state=123,
)
X_umap = umap_model.fit_transform(X_scaled)

# ------------------------------------------------------
# Mean TME per ecotype
# ------------------------------------------------------
ecotype_means = (
    df_tme.groupby("immune_ecotype")[tme_features]
    .mean()
    .reindex(eco_unique)
)

# ------------------------------------------------------
# Figure layout
# ------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], wspace=1.6, hspace=0.5)

ax_pca = fig.add_subplot(gs[0, 0])   # Panel A
ax_umap = fig.add_subplot(gs[0, 1])  # Panel B
ax_bar  = fig.add_subplot(gs[1, 0])  # Panel C
ax_rad  = fig.add_subplot(gs[1, 1], polar=True)  # Panel D

# ------------------------------------------------------
# Panel A: PCA by diagnosis
# ------------------------------------------------------
for d in diag_unique:
    idx = (diagnosis == d).values
    ax_pca.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        s=30,
        alpha=0.8,
        label=d,
        color=diag_color_map[d],
        edgecolor="white",
        linewidth=0.5,
    )

ax_pca.set_xlabel(f"PC1 ({expl_var[0]:.1f}% var)")
ax_pca.set_ylabel(f"PC2 ({expl_var[1]:.1f}% var)")
ax_pca.set_title("PCA of TME features by diagnosis")

# Legend (Diagnosis box) JUST OUTSIDE the plot on the right
leg = ax_pca.legend(
    #title="Diagnosis",
    loc="upper left",          # anchor the top-left corner of the legend…
    bbox_to_anchor=(1.02, 1),  # …at a point slightly to the right of the axes
    frameon=False,
)

# Left-align only the title within the legend box
title = leg.get_title()
title.set_ha("left")   # horizontal alignment
title.set_x(0.0)       # move from center (0.5) to left edge (0.0)

# Panel label A
ax_pca.text(
    -0.21, 1.15, "A",
    transform=ax_pca.transAxes,
    fontsize=22,
    fontweight="bold",
    va="top",
)

# ------------------------------------------------------
# Panel B: UMAP by immune ecotype
# ------------------------------------------------------
for e in eco_unique:
    idx = (ecotype == e)
    ax_umap.scatter(
        X_umap[idx, 0],
        X_umap[idx, 1],
        s=30,
        alpha=0.8,
        label=f"Ecotype {e}",
        color=eco_color_map[e],
        edgecolor="white",
        linewidth=0.5,
    )

ax_umap.set_xlabel("UMAP 1")
ax_umap.set_ylabel("UMAP 2")
ax_umap.set_title("UMAP of TME features by immune ecotype")

legB = ax_umap.legend(
    #title="Ecotype",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
)

# Left-align the legend title as well
legB._legend_title_box._text.set_horizontalalignment("left")

# Bold panel letter B
ax_umap.text(
    -0.25, 1.15, "B",
    transform=ax_umap.transAxes,
    fontsize=22,
    fontweight="bold",
    va="top",
)

# ------------------------------------------------------
# Panel C: Mean TME per ecotype (stacked bar)
# ------------------------------------------------------
x = np.arange(len(eco_unique))
bottom = np.zeros(len(eco_unique))

tme_cmap = plt.get_cmap("Accent")
tme_colors = {feat: tme_cmap(i % tme_cmap.N) for i, feat in enumerate(tme_features)}

for feat in tme_features:
    vals = ecotype_means[feat].values
    ax_bar.bar(
        x,
        vals,
        bottom=bottom,
        label=feat.replace("_raw",""),
        color=tme_colors[feat],
        edgecolor="black",
        linewidth=0.3,
    )
    bottom += vals

ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f"Ecotype {e}" for e in eco_unique])
ax_bar.set_ylabel("Mean fraction")
ax_bar.set_ylim(0, 1.05)
ax_bar.set_title("Mean TME composition per immune ecotype")

# smaller x-tick label size
ax_bar.tick_params(axis="x", labelsize=8)

ax_bar.legend(
    title="TME component",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
)

# Panel label C
ax_bar.text(
    -0.25, 1.15, "C",
    transform=ax_bar.transAxes,
    fontsize=22,
    fontweight="bold",
    va="top",
)

# ------------------------------------------------------
# Panel D: Radar plot of TME profile per ecotype
# ------------------------------------------------------
n_vars = len(tme_features)
angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

radar_labels = ["T cells", "B cells", "Myeloid", "NK cells", "Stromal", "Unknown"]

ax_rad.set_theta_offset(np.pi / 2)
ax_rad.set_theta_direction(-1)

ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(radar_labels)



ax_rad.set_rlabel_position(0)
ax_rad.set_yticks([0.2, 0.4, 0.6, 0.8])
ax_rad.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
ax_rad.set_ylim(0, 1.0)

for e in eco_unique:
    row = ecotype_means.loc[e, tme_features].values
    values = np.concatenate([row, [row[0]]])
    ax_rad.plot(
        angles,
        values,
        label=f"Ecotype {e}",
        color=eco_color_map[e],
        linewidth=2,
    )
    ax_rad.fill(
        angles,
        values,
        color=eco_color_map[e],
        alpha=0.15,
    )

ax_rad.set_title("TME radar profiles by immune ecotype", pad=15)

# Shrink and move Panel D downward slightly
box = ax_rad.get_position()
ax_rad.set_position([
    box.x0 - 0.01,       # move slightly left
    box.y0 - 0.00000001, # move DOWN to align with Panel C
    box.width * 0.9,    # reduce width
    box.height * 0.9,   # reduce height
])

ax_rad.legend(
    bbox_to_anchor=(1.23, 1.0),
    loc="upper left",
    frameon=False,
    fontsize=8,
)

# Panel label D
ax_rad.text(
    -0.22, 1.45, "D",
    transform=ax_rad.transAxes,
    fontsize=22,
    fontweight="bold",
    va="top",
)

# ------------------------------------------------------
# Save (no global title)
# ------------------------------------------------------
plt.tight_layout()

out_png = os.path.join(outdir, "Figure3_immune_ecotypes.png")
out_pdf = os.path.join(outdir, "Figure3_immune_ecotypes.pdf")
fig.savefig(out_png, dpi=600)
fig.savefig(out_pdf)
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

plt.show()
