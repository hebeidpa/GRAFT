# GRAFT

GRAFT is a research pipeline for dual-task prediction of gastric cancer
recurrence and postoperative complications from abdominal CT, L3/body-mask
measurements, primary-tumour-mask measurements, and fixed preoperative
clinical variables.

> Research use only. This software is not a medical device and must not be used
> as a standalone diagnostic or treatment system.

## What is included

- Complete preprocessing, feature extraction, training, prediction, and
  external-validation code.
- Leakage-safe repeated stratified cross-validation.
- A fixed, documented single-CSV cohort schema.
- Two fully synthetic NIfTI examples with no patient-derived content.
- A released trained fusion-head checkpoint in `weights/`.
- Tests and a GitHub Actions workflow.

The released checkpoint checksum and architecture details are documented in
[`weights/README.md`](weights/README.md) and [`MODEL_CARD.md`](MODEL_CARD.md).

The third-party `Pillar0-AbdomenCT` base model is not redistributed. Users must
obtain it from its official provider under the provider's access terms and
license.

## Input: three NIfTI files and one CSV per cohort

There is no required patient count or NIfTI filename pattern. Every row of one
`cohort.csv` explicitly maps a case to three files:

```text
data/
├── cohort.csv
└── images/
    ├── any_ct_filename.nii.gz
    ├── any_l3_mask_filename.nii.gz
    └── any_tumour_mask_filename.nii.gz
```

Relative image paths are resolved against `paths.data_root`; absolute paths are
also accepted. `case_id` values only need to be non-empty and unique.

### Fixed CSV columns

The following column names are required and case-sensitive:

| Group | Required columns |
|---|---|
| File mapping | `case_id`, `ct_path`, `l3_mask_path`, `tumor_mask_path` |
| Training/validation labels | `recurrence`, `complication` (binary 0/1) |
| Continuous/ordinal clinical | `Age (years)`, `Charlson index`, `BMI`, `PNI (cont.)`, `NLR (cont.)`, `PLR (cont.)`, `CONUT score`, `SII`, `CEA`, `CA19-9`, `CA72-4` |
| Categorical clinical | `Sex`, `ECOG`, `cT`, `cN`, `cTNM stage`, `Borrmann type`, `Tumor location` |

Additional CSV columns are ignored. Do not include postoperative outcomes,
pathological variables acquired after the prediction time point, survival
outcomes, or predictions from another model as input features.

For prediction with the released checkpoint, categorical values must use the
same tokens seen during training: `Sex` = `1`/`2`, `ECOG` = `0`/`1`, `cT` =
`1`/`2`, `cN` = `1`/`2`, `cTNM stage` = `II`/`III`, `Borrmann type` =
`1`/`2`/`3`/`4`, and `Tumor location` = `1`/`2`/`3`. These are source-cohort
codes; users must map local definitions against the original study codebook and
must not infer their meaning from the integers alone. Retrained checkpoints
store the categories learned from their own training cohort.

See [`examples/synthetic/cohort.csv`](examples/synthetic/cohort.csv) for the
complete schema. The example images and values are programmatically generated
and contain no patient information.

## Model inputs

1. Frozen Pillar0-AbdomenCT global CT representation (1,152 values).
2. Primary tumour and L3 mask measurements: volume, extent, axial area, HU
   distributions, and L3-label proportions.
3. The fixed preoperative clinical variables above.

Within every cross-validation split, numerical imputation and scaling and
categorical imputation and one-hot encoding are fitted on the training fold
only. The held-out fold is never used to fit preprocessing parameters.

## Installation

Python 3.10 is supported. Install the CUDA build of PyTorch appropriate for
your system first. For CUDA 12.6:

```bash
conda create -n graft python=3.10 pip -y
conda activate graft
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps .
```

Copy and edit the configuration:

```bash
cp config.example.yaml config.yaml
```

Place the complete base-model directory at the configured `model_dir`; see
[`models/README.md`](models/README.md).

## Validate a cohort

```bash
graft-validate --config config.yaml
```

This validates CSV columns, unique case IDs, labels, clinical values, NIfTI
geometry, mask values, and file availability. Reports are written under the
configured output directory.

The public synthetic data can be checked immediately:

```bash
graft-validate --config examples/synthetic/config.yaml
```

## Extract imaging and mask features

First test two cases:

```bash
graft-extract --config config.yaml --max-cases 2
```

Then process the full cohort:

```bash
graft-extract --config config.yaml
```

Per-case features are cached, so an interrupted extraction can be resumed.

## Train on a labelled cohort

```bash
graft-validate-clinical --config config.yaml
graft-train --config config.yaml
```

The default is 10 repetitions of 5-fold joint-label stratified
cross-validation with an inner validation split for epoch selection. No fixed
sample count is built into the code. The minimum count in every observed joint
label stratum must be at least `training.n_splits`.

Outputs include:

```text
outputs/training_clinical/
├── metrics.json
├── oof_predictions_all.csv
├── oof_predictions_mean.csv
├── evaluation_curves.png
├── clinical_model_features.csv
├── linear_feature_weights.csv
└── final_fusion_head.pt
```

The reported ROC-AUC, PR-AUC, Brier score, log loss, sensitivity, specificity,
and F1 are based on out-of-fold predictions. Current-cohort results are internal
validation, not evidence of clinical utility.

## Predict with the released fusion head

After extracting features for a cohort:

```bash
graft-predict \
  --features outputs/features.npz \
  --clinical data/cohort.csv \
  --checkpoint weights/final_fusion_head.pt \
  --output outputs/external_predictions.csv
```

The checkpoint stores the fitted image scaler, clinical imputer, categorical
encoder, input schema, and task head. The Pillar0 base model is still required
to create `features.npz` from new CT data.

## Evaluate an independent labelled cohort

```bash
graft-evaluate \
  --cohort data/cohort.csv \
  --predictions outputs/external_predictions.csv \
  --output outputs/external_validation_metrics.json
```

External evaluation requires both positive and negative cases for each task.
Never refit the released head or its preprocessing on the external-validation
cohort.

## Reproducibility and public-data policy

- Real patient CSV files, NIfTI volumes, extracted features, outputs, and local
  configuration files are excluded by `.gitignore`.
- Only `examples/synthetic/` is allowed through the repository data filters.
- The synthetic example is too small for model training or performance claims.
- The released fusion head contains no patient IDs or raw patient records.
- Before any contribution, run `git status` and inspect every staged file for
  protected health information, credentials, absolute personal paths, and
  institution-only data.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Acknowledgements

GRAFT uses image representations from
[Pillar-0](https://arxiv.org/abs/2511.17803), developed by the Yala Lab and
collaborators. The abdomen CT checkpoint used by this project is
[`YalaLab/Pillar0-AbdomenCT`](https://huggingface.co/YalaLab/Pillar0-AbdomenCT),
which is part of the official
[Pillar-0 model collection](https://huggingface.co/collections/YalaLab/pillar-0).
We thank the Pillar-0 authors for releasing their research and pretrained
models. Pillar-0 is a separate project and retains its own license and access
terms.

If you use GRAFT, please also cite the Pillar-0 paper:

```bibtex
@misc{agrawal2025pillar0,
  title         = {Pillar-0: A New Frontier for Radiology Foundation Models},
  author        = {Agrawal, Kumar Krishna and Liu, Longchao and Lian, Long and
                   Nercessian, Michael and Harguindeguy, Natalia and Wu, Yufu and
                   Mikhael, Peter and Lin, Gigin and Sequist, Lecia V. and
                   Fintelmann, Florian and Darrell, Trevor and Bai, Yutong and
                   Chung, Maggie and Yala, Adam},
  year          = {2025},
  eprint        = {2511.17803},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2511.17803}
}
```

## License and citation

Code is provided under the MIT License. Third-party models and datasets retain
their own licenses and access conditions. Citation metadata are provided in
[`CITATION.cff`](CITATION.cff).
