from pathlib import Path
import pandas as pd

ROOT = Path("/TME_OU_Branching")
FEAT_DIR = ROOT / "derived_features"

# mapping file manually created later
MAP_PATH = ROOT / "patient_id_mapping.csv"

# clinical metadata Excel
META_PATH = ROOT / "kmt2a_longitudinal_clean.xlsx"

# ---------------------------------------------------------------------
# 1) Load TME row metadata + baseline embeddings
# ---------------------------------------------------------------------
tme_meta = pd.read_csv(FEAT_DIR / "Ep_baseline_rowmeta.csv", index_col=0)
tme_meta.index.name = "participant_id"

Ep_df = pd.read_csv(FEAT_DIR / "Ep_baseline_z.csv", index_col=0)
Ep_df.index.name = "participant_id"

tme_full = tme_meta.join(Ep_df, how="inner").reset_index()

# ---------------------------------------------------------------------
# 2) Load clinical table (Series sheet)
# ---------------------------------------------------------------------
xls = pd.ExcelFile(META_PATH)
print("Available sheets:", xls.sheet_names)

sheet_to_use = "Series"
patients_df = pd.read_excel(xls, sheet_name=sheet_to_use)
patients_df = patients_df.rename(columns=str)

# Pick useful clinical columns if they exist
core_cols = [c for c in ["Patient_ID", "Disease", "Group"] if c in patients_df.columns]
patients_core = patients_df[core_cols].drop_duplicates()

# ---------------------------------------------------------------------
# 3) Load or create mapping table (participant_id → Patient_ID)
# ---------------------------------------------------------------------
if not MAP_PATH.exists():
    # create a template mapping file from current TME participant IDs
    template = tme_full[["participant_id"]].drop_duplicates().copy()
    template["Patient_ID"] = ""  # fill this manually in Excel
    template.to_csv(MAP_PATH, index=False)

    print("\nCreated template mapping file at:")
    print(" ", MAP_PATH)
    print("\nOpen this CSV in Excel, fill in the 'Patient_ID' column (P15, P18, ...),")
    print("save it, and then re-run this script.")
    raise SystemExit(0)

# if file exists, load it
id_map = pd.read_csv(MAP_PATH).rename(columns=str)

required_cols = {"participant_id", "Patient_ID"}
missing = required_cols - set(id_map.columns)
if missing:
    raise ValueError(
        f"Mapping file {MAP_PATH} is missing columns: {missing}.\n"
        f"It must have at least: participant_id, Patient_ID."
    )

id_map = id_map[["participant_id", "Patient_ID"]].drop_duplicates()

# ---------------------------------------------------------------------
# 4) Merge TME + mapping + clinical
# ---------------------------------------------------------------------
tme_with_ids = tme_full.merge(id_map, on="participant_id", how="left")
master = tme_with_ids.merge(patients_core, on="Patient_ID", how="left")

master = master.set_index("Patient_ID").sort_index()

# ---------------------------------------------------------------------
# 5) Write output
# ---------------------------------------------------------------------
out = ROOT / "patient_master_table.csv"
master.to_csv(out)

print("\nFinal patient master table written to:\n", out)
