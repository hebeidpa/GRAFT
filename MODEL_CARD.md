# GRAFT mask-guided model card

## Intended use

Research on preoperative prediction of gastric cancer recurrence and postoperative complications from abdominal CT, primary-tumour and L3 masks, and preoperative clinical variables.

The model is not intended for diagnosis, treatment selection, autonomous clinical decisions, or direct patient use.

## Architecture

- Pillar0-AbdomenCT global CT representation with conservative LoRA adaptation.
- Multi-scale tumour ROI pooling from Pillar intermediate feature maps.
- Label-specific L3 ROI pooling for mask labels 1, 2, and 3.
- Clinical encoder fitted within each training partition.
- Separate task-aware modality gates and binary heads for recurrence and complication.

## Validation design

The public trainer implements five outer joint-label-stratified folds. Each outer training partition is split again for early stopping. The outer test fold is not used for preprocessing, epoch selection, or checkpoint selection and is evaluated once after selecting the inner checkpoint.

This design provides internal out-of-fold estimates. It is not external or prospective validation.

## Released weights

No mask-guided weights are currently committed. The complete five-fold run, checksums, preprocessing artifacts, and machine-readable metrics must be reviewed before release. The earlier handcrafted-feature fusion checkpoint is incompatible and has been removed.

## Limitations

- Performance can shift with scanner protocol, reconstruction, segmentation practice, cohort spectrum, staging, assays, and missingness.
- Segmentation errors directly affect tumour and L3 embeddings.
- A single retrospective cohort cannot establish clinical utility or transportability.
- Discrimination alone is insufficient; calibration, clinical thresholds, decision curves, uncertainty, subgroups, and independent validation are required.
- Pillar-0 is a separate dependency with its own license and limitations.

## Privacy

The repository contains only synthetic example inputs. Patient data, identifiers, local paths, logs, outputs, and checkpoints are excluded from version control.
