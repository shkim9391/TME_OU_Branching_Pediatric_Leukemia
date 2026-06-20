import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ------------------------------------------------------
# Paths and data
# ------------------------------------------------------
input_path = "/TME_OU_Branching/covariate_matrix.csv"
outdir = "/TME_OU_Branching/Figure_2"
os.makedirs(outdir, exist_ok=True)

# ------------------------------------------------------
# TME features (must exist as columns in covariate_matrix.csv)
# ------------------------------------------------------
df = pd.read_csv(input_path)

rename_map = {
    "frac_T_given_known_z": "T",
    "frac_B_given_known_z": "B",
    "frac_myeloid_given_known_z": "Myeloid",
    "frac_NK_given_known_z": "NK",
    "frac_stromal_given_known_z": "Stromal",
    "frac_unknown_z": "Unknown",
}
df = df.rename(columns=rename_map)

tme_features = ["T", "B", "Myeloid", "NK", "Stromal", "Unknown"]

missing = [c for c in tme_features if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing TME columns in covariate_matrix.csv: {missing}\n"
        f"Available columns include: {list(df.columns)[:40]} ..."
    )

# ------------------------------------------------------
# Attach true diagnosis/subdiagnosis labels
# ------------------------------------------------------
master_path = "/Users/seung-hwan.kim/Desktop/TME_OU_Branching/patient_master_table.csv"
meta = pd.read_csv(master_path, usecols=["Patient_ID", "diagnosis", "subdiagnosis"])

df = df.merge(meta, on="Patient_ID", how="left")

df["subdiagnosis"] = df["subdiagnosis"].fillna("").astype(str).str.strip()
df.loc[df["subdiagnosis"].eq(""), "subdiagnosis"] = "NOS"
df["subdiagnosis"] = df["subdiagnosis"].replace({"hyperdiploid": "Hyperdiploid"})

diag_col = "subdiagnosis"

# Optional: stable category order for legend
subdiag_order = sorted(df[diag_col].dropna().unique())

# ------------------------------------------------------
# Precomputations
# ------------------------------------------------------
long_df = df.melt(
    id_vars=[diag_col],
    value_vars=tme_features,
    var_name="Feature",
    value_name="Value"
)
feature_order = tme_features

corr = df[tme_features].corr(method="spearman")

X = df[tme_features].values
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=123)
X_pca = pca.fit_transform(X_scaled)

pc_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"], index=df.index)
pc_df[diag_col] = df[diag_col].values

var_exp = pca.explained_variance_ratio_ * 100
pc1_var, pc2_var = var_exp[0], var_exp[1]

ks = list(range(1, 11))
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=123, n_init=20)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# ------------------------------------------------------
# Plot style
# ------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 19,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
})

# Large palette for many subdiagnosis classes
palette = sns.color_palette("tab20", n_colors=len(subdiag_order))
palette_map = dict(zip(subdiag_order, palette))

# ------------------------------------------------------
# Composite figure
# ------------------------------------------------------
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.15, 1.15],
    height_ratios=[1, 1],
    wspace=0.35,
    hspace=0.75
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

# Leave room on right for shared legend
fig.subplots_adjust(left=0.07, right=0.83, top=0.93, bottom=0.10)

# ------------------------------------------------------
# Panel letters
# ------------------------------------------------------
panel_letter_kwargs = dict(
    fontsize=26,
    fontweight="bold",
    va="top",
    ha="left",
    bbox=dict(facecolor="white", edgecolor="none", pad=0.2)
)

axA.text(-0.08, 1.08, "A", transform=axA.transAxes, **panel_letter_kwargs)
axB.text(-0.50, 1.08, "B", transform=axB.transAxes, **panel_letter_kwargs)
axC.text(-0.08, 1.08, "C", transform=axC.transAxes, **panel_letter_kwargs)
axD.text(-0.21, 1.08, "D", transform=axD.transAxes, **panel_letter_kwargs)

# ------------------------------------------------------
# Panel A: TME distributions
# ------------------------------------------------------
sns.boxplot(
    data=long_df,
    x="Feature",
    y="Value",
    order=feature_order,
    showfliers=False,
    showcaps=True,
    width=0.60,
    boxprops={"alpha": 0.35, "facecolor": "white", "edgecolor": "black"},
    whiskerprops={"linewidth": 1.4, "color": "black"},
    medianprops={"linewidth": 2.0, "color": "black"},
    ax=axA,
)

sns.stripplot(
    data=long_df,
    x="Feature",
    y="Value",
    hue=diag_col,
    hue_order=subdiag_order,
    palette=palette_map,
    order=feature_order,
    dodge=True,
    jitter=0.25,
    alpha=0.65,
    size=4,
    linewidth=0,
    ax=axA,
)

axA.set_ylabel("z-scored TME fraction")
axA.set_xlabel("")
plt.setp(axA.get_xticklabels(), rotation=30, ha="right")
axA.set_title("Standardized TME feature distributions", pad=12)
axA.tick_params(axis="both", labelsize=14)

# remove local legend
legA = axA.get_legend()
if legA is not None:
    legA.remove()

# capture handles for shared legend
handles, labels = axA.get_legend_handles_labels()

# ------------------------------------------------------
# Panel B: Heatmap
# ------------------------------------------------------
hm = sns.heatmap(
    corr,
    annot=False,
    cmap="vlag",
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"shrink": 0.70},
    ax=axB,
)

axB.set_title("Spearman correlations among TME features", pad=10)
plt.setp(axB.get_xticklabels(), rotation=45, ha="right")
plt.setp(axB.get_yticklabels(), rotation=0)
axB.tick_params(axis="both", labelsize=14)

# enlarge colorbar fonts
cbar = hm.collections[0].colorbar
cbar.set_label("Spearman correlation", fontsize=16)
cbar.ax.tick_params(labelsize=14)

# ------------------------------------------------------
# Panel C: PCA
# ------------------------------------------------------
sns.scatterplot(
    data=pc_df,
    x="PC1",
    y="PC2",
    hue=diag_col,
    hue_order=subdiag_order,
    palette=palette_map,
    s=70,
    alpha=0.85,
    edgecolor="black",
    linewidth=0.4,
    ax=axC,
)

axC.set_xlabel(f"PC1 ({pc1_var:.1f}% var. explained)")
axC.set_ylabel(f"PC2 ({pc2_var:.1f}% var. explained)")
axC.set_title("PCA of standardized TME features by subdiagnosis", pad=12)
axC.tick_params(axis="both", labelsize=14)

legC = axC.get_legend()
if legC is not None:
    legC.remove()

# ------------------------------------------------------
# Panel D: Elbow plot
# ------------------------------------------------------
axD.plot(ks, inertias, marker="o", linewidth=2, markersize=7)
axD.set_xticks(ks)
axD.set_xlabel("Number of clusters (k)")
axD.set_ylabel("Within-cluster sum of squares (inertia)")
axD.set_title("K-means elbow plot for ecological-context resolution", pad=12)
axD.tick_params(axis="both", labelsize=14)

if 4 in ks:
    axD.axvline(x=4, linestyle="--", linewidth=1.5, color="steelblue")
    ymin, ymax = min(inertias), max(inertias)
    ytext = ymin + 0.18 * (ymax - ymin)
    axD.text(4 + 0.12, ytext, "working k = 4", rotation=90, va="bottom", fontsize=14)

# ------------------------------------------------------
# Save
# ------------------------------------------------------
out_png = os.path.join(outdir, "Figure2_composite_revised.png")
out_tiff = os.path.join(outdir, "Figure2_composite_revised.tiff")

fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_tiff, dpi=600, bbox_inches="tight")
plt.close(fig)

print("Saved composite figure to:")
print("  ", out_png)
print("  ", out_tiff)
