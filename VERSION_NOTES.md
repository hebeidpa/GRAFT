# GRAFT v4 separate internal validation cohort

This clean research release implements the revised training procedure.

- Recurrence loss: positive-class-weighted binary cross-entropy.
- Complication loss: alpha-balanced binary focal loss.
- Defaults: focal alpha 0.25, gamma 2, recurrence coefficient 1.0, and
  complication coefficient 0.75.
- Epoch selection: lowest loss in a stratified subset drawn only from the
  training cohort.
- Final model: initialized anew and trained on the complete training cohort for
  the selected epoch count.
- Evaluation: a separate internal validation cohort supplied through its own
  CSV and never used for preprocessing, early stopping or parameter updates.

Run the cohort validator and dry-run before full training. Replace all example
paths with local paths; no institution-specific paths are included.

The reviewed final checkpoint in `weights/` was fitted on all 941 development
cases for six epochs. Separate internal validation is pending. The release does
not include patient data, patient-level outputs, local paths, or the separately
licensed Pillar0-AbdomenCT base model.
