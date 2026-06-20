import pandas as pd
from pathlib import Path

FIG3_DIR = Path("/TME_OU_Branching/Figure_3")

assign = pd.read_csv(FIG3_DIR / "patient_ecological_context_assignments.csv")
master = pd.read_csv(FIG3_DIR / "ecological_context_master_table.csv")

print("Assignment columns:")
print(assign.columns.tolist())

print("\nContext counts from assignments:")
print(assign["ecological_context"].value_counts().sort_index())

print("\nMaster table:")
print(master[["ecological_context", "context_label", "n_participants", "color"]])

required = {
    "Patient_ID",
    "diagnosis",
    "subdiagnosis",
    "T_z",
    "B_z",
    "Myeloid_z",
    "NK_z",
    "Stromal_z",
    "Unknown_z",
    "ecological_context",
}

missing = required - set(assign.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("\nFigure 3 assignment file looks OK.")
