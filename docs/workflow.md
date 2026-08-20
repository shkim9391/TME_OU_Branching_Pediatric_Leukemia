# Analysis Workflow

This document describes the reproducible analysis workflow for the revised pediatric leukemia tumor-microenvironment Ornstein–Uhlenbeck (OU) study.

The analysis is organized into three computational waves:

- **Wave 1:** single-cell annotation harmonization, participant reconciliation, ecological-context discovery, and clustering robustness;
- **Wave 2:** longitudinal input construction, primary OU ablation models, true leave-one-participant-out validation, transformed-trait sensitivity, and secondary ordinal-state modeling;
- **Wave 3:** prior sensitivity, simulation-based parameter recovery, posterior-predictive calibration, and final figure regeneration.

The final manuscript emphasizes **continuous immune composition** rather than discrete ecological-context labels as the primary ecological information used for longitudinal OU modeling.

Candidate ecological contexts are retained as descriptive coarse-grainings of the cross-sectional immune-compositional landscape. They are not treated as fixed leukemia ecotypes, biological subtypes, strict evolutionary taxa, or independently validated longitudinal states.

---

## 1. Analysis overview

The revised workflow deliberately separates three analytical quantities:

1. **Observed immune composition**
   - harmonized participant-level TNK, B-lineage, and myeloid fractions derived from ScPCA single-cell data;

2. **Candidate ecological contexts**
   - descriptive K-means clusters obtained by coarse-graining the participant-level immune-compositional landscape;

3. **Longitudinal OU dynamics**
   - effective continuous-time stochastic parameters fitted to the prespecified semiquantitative longitudinal `x` endpoint.

These quantities are evaluated separately so that cross-sectional clustering stability is not conflated with longitudinal predictive relevance.

The main workflow is:

```text
ScPCA single-cell data
        |
        v
Common cell-type mapping
        |
        v
Sample-level harmonized TME composition
        |
        v
Participant-level TNK/B/Myeloid simplex
        |
        +----------------------------+
        |                            |
        v                            v
Raw K-means clustering            CLR sensitivity
k = 2 ... 6                       k = 2 ... 6
        |                            |
        +-------------+--------------+
                      |
                      v
      Candidate ecological-context robustness
                      |
                      v
         Participant-ID reconciliation
                      |
                      v
        Longitudinal x transition table
                      |
                      v
          Primary OU ablation models
                      |
         +------------+-------------+
         |                          |
         v                          v
Transition-level PSIS-LOO      True participant LOPO
diagnostic                     predictive validation
         |                          |
         +------------+-------------+
                      |
                      v
        Secondary robustness analyses
       /             |              \
transformed x    ordinal n      prior sensitivity
                      |
                      v
       Posterior-predictive calibration
                      |
                      v
         Simulation parameter recovery
                      |
                      v
           Final manuscript figures
