#generate_patient_immune_ecotypes.py

"""
Generate immune ecotypes (k-means clusters) based on TME covariates.

Inputs:
  - covariate_matrix.csv
  - patient_master_table.csv

Outputs:
  - patient_immune_ecotypes.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

ROOT = Path("/")

# Load covariates + metadata
cov = pd.read_csv(ROOT / "covariate_matrix.csv")
master = pd.read_csv(ROOT / "patient_master_table.csv")

# Merge metadata (diagnosis / subdiagnosis)
df = cov.merge(
    master[["Patient_ID", "diagnosis", "subdiagnosis"]],
    on="Patient_ID",
    how="left"
)

# Select TME features for immune clustering
tme_cols = [
    "frac_unknown_z",
    "frac_T_given_known_z",
    "frac_B_given_known_z",
    "frac_myeloid_given_known_z",
    "frac_NK_given_known_z",
    "frac_stromal_given_known_z",
]

tme_cols = [c for c in tme_cols if c in df.columns]

X = df[tme_cols].values

# Standardize TME features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose number of ecotypes (k=4)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_scaled)

df["immune_ecotype"] = labels

# Save final output
out_path = ROOT / "patient_immune_ecotypes.csv"
df.to_csv(out_path, index=False)

print("Saved immune ecotypes to:", out_path)
print("\n# Patients per ecotype:")
print(df["immune_ecotype"].value_counts())
