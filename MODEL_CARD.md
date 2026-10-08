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

The development cohort is split internally only to select the training epoch by validation loss. A fresh final model is then initialized and fitted on the complete development cohort for the selected epoch count. A separate internal-validation cohort must be supplied through its own CSV and is evaluated without preprocessing refits, early stopping, or parameter updates.

The released final weight completed training on 941 development-cohort cases for six epochs. Its separate internal validation remains pending. Training loss is not a validation-performance estimate.

## Released weights

`weights/graft_v4_final_model.pt` contains the trained LoRA adapters and non-Pillar model parameters. `weights/graft_v4_final_model.clinical_preprocessor.joblib` contains the fitted clinical preprocessing transform. The Pillar0-AbdomenCT base checkpoint is required separately and is not redistributed. The earlier handcrafted-feature fusion checkpoint is incompatible and is not included.

## Limitations

- Performance can shift with scanner protocol, reconstruction, segmentation practice, cohort spectrum, staging, assays, and missingness.
- Segmentation errors directly affect tumour and L3 embeddings.
- A single retrospective cohort cannot establish clinical utility or transportability.
- Discrimination alone is insufficient; calibration, clinical thresholds, decision curves, uncertainty, subgroups, and independent validation are required.
- Pillar-0 is a separate dependency with its own license and limitations.

## Privacy

The repository contains only synthetic example inputs and reviewed model artifacts. Patient data, identifiers, local paths, logs, and patient-level outputs are excluded from version control. The released checkpoint contains model parameters, architecture settings, and aggregate schema metadata; it contains no case identifiers or image data.
