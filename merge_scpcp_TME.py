#merge_scpcp_TME.py

import pandas as pd
from pathlib import Path

base = Path("/")
feat_dir = base / "derived_features"

# load broad TME features from both projects
df22 = pd.read_csv(feat_dir / "scpcp22_sample_TME_features_broad.csv", index_col=0)
df8  = pd.read_csv(feat_dir / "scpcp8_sample_TME_features_broad.csv", index_col=0)

print("SCPCP000022:", df22.shape)
print("SCPCP000008:", df8.shape)

# add a cohort label so you can stratify later
df22["cohort"] = "SCPCP000022_diverse"
df8["cohort"]  = "SCPCP000008_ALL"

# make sure both have the same columns & order
common_cols = [
    "participant_id", "diagnosis", "subdiagnosis",
    "tissue_location", "disease_timing",
    "frac_T", "frac_B", "frac_myeloid", "frac_NK",
    "frac_stromal", "frac_known", "cohort"
]

df22 = df22[common_cols]
df8  = df8[common_cols]

combined = pd.concat([df22, df8], axis=0)
print("Combined:", combined.shape)

out = feat_dir / "scpcp_combined_sample_TME_features_broad.csv"
combined.to_csv(out)
print("Wrote combined TME features to:")
print(" ", out)
