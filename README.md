# TME OU Dynamics in Pediatric Leukemia

Code, processed analysis outputs, and reproducible workflows for integrating
single-cell-derived immune composition with continuous-time
Ornstein–Uhlenbeck (OU) modeling of pediatric leukemia longitudinal dynamics.

This repository supports the manuscript:

> **Continuous immune composition outperforms coarse ecological-context labels
> in Ornstein–Uhlenbeck modeling of pediatric leukemia longitudinal dynamics**

The revised analysis asks whether harmonized continuous immune composition
contains longitudinal information beyond coarse ecological-context labels
derived from the same cross-sectional single-cell data.

## Archived Release

The revised manuscript analysis and associated reproducibility materials are
archived in Zenodo as:

> **v1.3.2 — Revised Continuous-Ecology OU Analysis and Reproducibility Release**

**Version DOI:**  
[https://doi.org/10.5281/zenodo.22033731](https://doi.org/10.5281/zenodo.22033731)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22033731.svg)](https://doi.org/10.5281/zenodo.22033731)

### Citation

Kim, S.-H. (2026). *shkim9391/TME_OU_Branching_Pediatric_Leukemia:
v1.3.2 — Revised Continuous-Ecology OU Analysis and Reproducibility Release*
(Version v1.3.2) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.22033731

This version-specific DOI identifies the archived repository release
corresponding to the revised continuous-ecology OU analysis. When reproducing
or referring specifically to the analyses reported in the revised manuscript,
please use this archived version.

## Overview

Single-cell pediatric leukemia datasets contain substantial variation in
cellular composition, but discrete ecological groupings can be influenced by
leukemia lineage, project structure, annotation differences, malignant-cell
composition, and the constrained geometry of compositional data.

The revised workflow therefore separates three analytical quantities:

1. **Continuous immune composition** derived from harmonized single-cell
   annotations.
2. **Candidate ecological contexts** obtained by coarse-graining the
   cross-sectional immune-compositional landscape.
3. **Effective OU parameters** describing the prespecified longitudinal
   leukemia-state trajectory.

Candidate ecological contexts are treated as descriptive ecological
coarse-grainings rather than fixed leukemia ecotypes, biological subtypes, or
evolutionary taxa.

The primary longitudinal analysis evaluates whether continuous B-lineage and
myeloid composition, coarse context membership, or combinations of these
variables modulate the effective OU attractor and mean-reversion strength.

## Data Sources

The single-cell analysis uses two public projects from the
Single-cell Pediatric Cancer Atlas (ScPCA):

- `SCPCP000008` — pediatric acute lymphoblastic leukemia single-cell data.
- `SCPCP000022` — diverse pediatric leukemia single-cell data.

Raw `.h5ad` files and associated metadata are obtained from the ScPCA Portal.
Large raw single-cell files are not redistributed in this repository.

The longitudinal analysis uses the previously curated:

```text
kmt2a_longitudinal_clean.xlsx
