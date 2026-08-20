# Data Dictionary

This document describes the principal processed datasets, model-input tables,
diagnostic outputs, validation results, sensitivity-analysis tables, and
figure-source files used in the revised pediatric leukemia tumor
microenvironment Ornstein–Uhlenbeck (OU) analysis.

The repository distinguishes three major analytical units:

1. **sample-level single-cell composition**
2. **participant-level ecological composition**
3. **transition-level longitudinal observations**

Candidate ecological contexts are descriptive coarse-grainings of
participant-level immune composition. They are not interpreted as leukemia
subtypes, immutable ecotypes, strict evolutionary taxa, or independently
validated biological states.

---

# 1. Naming conventions

## Participant identifiers

Several identifier systems occur because the single-cell and longitudinal
datasets originated from different source workflows.

Common identifier fields include:

| Field | Meaning |
|---|---|
| `participant_id` | ScPCA-derived participant identifier |
| `Patient_ID` | reconciled/legacy longitudinal identifier, often P1–P100 style |
| `scpca_sample_id` | ScPCA sample identifier |
| `project` | source ScPCA project, typically `SCPCP000008` or `SCPCP000022` |

Participant reconciliation is documented explicitly in the Wave 1 audit files.

---

## Ecological composition variables

The primary revised ecological representation uses the three-component immune
simplex:

| Variable | Interpretation |
|---|---|
| `TNK` | combined T-cell and natural-killer compartment |
| `B` | B-lineage compartment |
| `Myeloid` | myeloid compartment |

Corresponding fraction fields may appear with names such as:

```text
frac_TNK
frac_B
frac_Myeloid
comp_TNK
comp_B
comp_Myeloid
