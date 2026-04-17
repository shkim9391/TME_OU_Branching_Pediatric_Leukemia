#scpcp_combined_sample_TMe_feature_broad.py

import pandas as pd
from pathlib import Path

base = Path("/")
feat_dir = base / "derived_features"

combined = pd.read_csv(feat_dir / "scpcp_combined_sample_TME_features_broad.csv", index_col=0)

tme_cols = ["frac_T", "frac_B", "frac_myeloid", "frac_NK", "frac_stromal", "frac_known"]

participant_TME = (
    combined
    .groupby("participant_id")[tme_cols]
    .mean()
)

# keep some useful categorical info (e.g. first diagnosis per participant)
meta = (
    combined
    .groupby("participant_id")[["diagnosis", "subdiagnosis"]]
    .first()
)

participant_TME = meta.join(participant_TME)

out = feat_dir / "scpcp_combined_participant_TME_features_broad.csv"
participant_TME.to_csv(out)
print("Wrote participant-level TME features to:")
print(" ", out, "shape:", participant_TME.shape)
