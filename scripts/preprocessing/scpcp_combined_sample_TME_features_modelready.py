import pandas as pd
from pathlib import Path

base = Path("/TME_OU_Branching/derived_features")
df = pd.read_csv(base / "scpcp_combined_sample_TME_features_broad.csv", index_col=0)

tme = ["frac_T","frac_B","frac_myeloid","frac_NK","frac_stromal","frac_known"]

# tumor/unannotated proxy
df["frac_unknown"] = 1.0 - df["frac_known"]

# immune fractions among "known" (avoid divide-by-zero)
den = df["frac_known"].clip(lower=1e-6)
for c in ["frac_T","frac_B","frac_myeloid","frac_NK","frac_stromal"]:
    df[c + "_given_known"] = df[c] / den

# optional: total immune (you can adjust what you count as "immune")
df["frac_immune_total"] = df["frac_T"] + df["frac_B"] + df["frac_myeloid"] + df["frac_NK"]
df["frac_immune_given_known"] = df["frac_immune_total"] / den

out = base / "scpcp_combined_sample_TME_features_modelready.csv"
df.to_csv(out)
print("Wrote:", out, df.shape)
