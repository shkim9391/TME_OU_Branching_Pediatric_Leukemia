#fig2_full.py

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
input_path = "/covariate_matrix.csv"
outdir = "/Figure_2"
os.makedirs(outdir, exist_ok=True)

# ------------------------------------------------------
# TME features (must exist as columns in covariate_matrix.csv)
# ------------------------------------------------------
df = pd.read_csv(input_path)

# 1) Rename old column names -> short names (if present)
rename_map = {
    "frac_T_given_known_z": "T",
    "frac_B_given_known_z": "B",
    "frac_myeloid_given_known_z": "Myeloid",
    "frac_NK_given_known_z": "NK",
    "frac_stromal_given_known_z": "Stromal",
    "frac_unknown_z": "Unknown",
}
df = df.rename(columns=rename_map)

# 2) Now define the feature list
tme_features = ["T", "B", "Myeloid", "NK", "Stromal", "Unknown"]

# 3) Now check they exist
missing = [c for c in tme_features if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing TME columns in covariate_matrix.csv: {missing}\n"
        f"Available columns include: {list(df.columns)[:40]} ..."
    )

# ------------------------------------------------------
# Attach true diagnosis/subdiagnosis labels (avoid drop_first pitfalls)
# ------------------------------------------------------
master_path = "/Users/seung-hwan.kim/Desktop/TME_OU_Branching/patient_master_table.csv"
meta = pd.read_csv(master_path, usecols=["Patient_ID", "diagnosis", "subdiagnosis"])

df = df.merge(meta, on="Patient_ID", how="left")

# Clean subdiagnosis labels
df["subdiagnosis"] = df["subdiagnosis"].fillna("").astype(str).str.strip()
df.loc[df["subdiagnosis"].eq(""), "subdiagnosis"] = "NOS"  # blanks -> NOS
df["subdiagnosis"] = df["subdiagnosis"].replace({"hyperdiploid": "Hyperdiploid"})

# Use subdiagnosis for coloring (matches your legend content)
diag_col = "subdiagnosis"

# ------------------------------------------------------
# Precomputations for each panel
# ------------------------------------------------------

# Panel A: long-format dataframe
long_df = df.melt(
    id_vars=[diag_col],
    value_vars=tme_features,
    var_name="Feature",
    value_name="Value"
)
feature_order = tme_features

# Panel B: correlation matrix
corr = df[tme_features].corr(method="spearman")

# Panel C: PCA
X = df[tme_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=123)
X_pca = pca.fit_transform(X_scaled)

pc_df = pd.DataFrame(
    X_pca[:, :2],
    columns=["PC1", "PC2"],
    index=df.index
)
pc_df[diag_col] = df[diag_col].values

var_exp = pca.explained_variance_ratio_ * 100
pc1_var, pc2_var = var_exp[0], var_exp[1]

# Panel D: k-means elbow
ks = list(range(1, 11))
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=123, n_init=20)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# ------------------------------------------------------
# Plot style
# ------------------------------------------------------
sns.set(context="talk", style="whitegrid")

# ------------------------------------------------------
# Improved 2×2 composite figure with more spacing
# ------------------------------------------------------
fig = plt.figure(figsize=(16, 13))

gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.1], height_ratios=[1, 1],
                      wspace=0.35, hspace=0.75)
fig.subplots_adjust(right=0.92, left=0.07, top=0.93, bottom=0.10)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

label_kw = dict(transform=axA.transAxes, fontsize=22, fontweight="bold",
                ha="right", va="bottom")

axA.text(-0.06, 1.03, "A", **label_kw)
axB.text(-0.23, 1.03, "B", transform=axB.transAxes, fontsize=22, fontweight="bold",
         ha="right", va="bottom")
axC.text(-0.06, 1.03, "C", transform=axC.transAxes, fontsize=22, fontweight="bold",
         ha="right", va="bottom")
axD.text(-0.01, 1.03, "D", transform=axD.transAxes, fontsize=22, fontweight="bold",
         ha="right", va="bottom")

# ------------------------------------------------------
# Panel A (TME Distributions)
# ------------------------------------------------------
sns.boxplot(
    data=long_df,
    x="Feature",
    y="Value",
    order=feature_order,
    showcaps=True,
    boxprops={"alpha": 0.6},
    whiskerprops={"linewidth": 1.5},
    medianprops={"linewidth": 2},
    ax=axA,
)

sns.stripplot(
    data=long_df,
    x="Feature",
    y="Value",
    hue=diag_col,
    order=feature_order,
    dodge=True,
    jitter=0.25,
    alpha=0.6,
    linewidth=0,
    ax=axA,
)

axA.set_ylabel("z-scored TME fraction")
axA.set_xlabel("")
axA.set_xticklabels(axA.get_xticklabels(), rotation=30, ha="right")
axA.set_title("TME distributions", pad=15)

# remove local legend
legA = axA.get_legend()
if legA:
    legA.remove()

# Capture legend handles for shared legend
handles, labels = axA.get_legend_handles_labels()

# ------------------------------------------------------
# Panel B (Heatmap)
# ------------------------------------------------------
sns.heatmap(
    corr,
    annot=False,
    cmap="vlag",
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"shrink": 0.7, "label": "Spearman correlation"},
    ax=axB,
)

axB.set_title("Pairwise Spearman correlations of TME features", pad=15)
plt.setp(axB.get_xticklabels(), rotation=45, ha="right")

# ------------------------------------------------------
# Panel C (PCA)
# ------------------------------------------------------
sns.scatterplot(
    data=pc_df,
    x="PC1",
    y="PC2",
    hue=diag_col,
    s=80,
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5,
    ax=axC,
)

axC.set_xlabel(f"PC1 ({pc1_var:.1f}% var. explained)")
axC.set_ylabel(f"PC2 ({pc2_var:.1f}% var. explained)")
axC.set_title("PCA of TME covariates (colored by subdiagnosis)", pad=15)

legC = axC.get_legend()
if legC:
    legC.remove()

# ------------------------------------------------------
# Panel D (Elbow plot)
# ------------------------------------------------------
axD.plot(ks, inertias, marker="o")
axD.set_xticks(ks)
axD.set_xlabel("Number of clusters (k)")
axD.set_ylabel("Within-cluster sum of squares (inertia)")
axD.set_title("K-means elbow plot for TME ecotypes", pad=15)

if 4 in ks:
    axD.axvline(x=4, linestyle="--", linewidth=1.5)
    ymin, ymax = min(inertias), max(inertias)
    ytext = ymin + 0.18 * (ymax - ymin)
    axD.text(4 + 0.15, ytext, "k = 4", rotation=90, va="bottom")

# ------------------------------------------------------
# Shared legend — pulled slightly inward
# ------------------------------------------------------
labels_unique = list(dict.fromkeys(labels))  # remove duplicates
hl = dict(zip(labels, handles))        # last handle per label
labels_unique = list(hl.keys())
handles_unique = [hl[l] for l in labels_unique]

fig.legend(
    handles_unique, labels_unique,
    title="Subdiagnosis",
    loc="center left",
    bbox_to_anchor=(0.95, 0.55),
    fontsize=15,
    title_fontsize=16,
)

# Reduce right margin so legend is closer
fig.subplots_adjust(right=0.88, left=0.08, top=0.93, bottom=0.10)

# ------------------------------------------------------
# Save
# ------------------------------------------------------
out_png = os.path.join(outdir, "Figure2_composite.png")
out_pdf = os.path.join(outdir, "Figure2_composite.pdf")

fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
plt.close(fig)

print("Saved composite figure to:")
print("  ", out_png)
print("  ", out_pdf)
