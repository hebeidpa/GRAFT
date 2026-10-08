# GRAFT

GRAFT is a research pipeline for dual-task prediction of gastric cancer recurrence and postoperative complications using abdominal CT, a primary-tumour mask, an L3 body-composition mask, and preoperative clinical variables.

> Research use only. GRAFT is not a medical device and must not be used as a standalone diagnostic or treatment system.

## Current model

This repository contains one current architecture:

1. CT is processed by `Pillar0-AbdomenCT` with LoRA adapters in the final Atlas stage.
2. The Pillar global feature is projected to a 128-dimensional CT embedding.
3. Tumour and L3 masks are aligned to the Pillar input and used only for multi-scale ROI pooling from intermediate Pillar feature maps.
4. L3 labels 1, 2, and 3 are pooled separately and fused with learned region weights.
5. CT, tumour, L3, and clinical embeddings are combined with separate task-aware gates for recurrence and complication.

Masks are never passed into Pillar-0 as image channels. The previous 67-dimensional handcrafted mask-phenotype branch and its checkpoint are not part of this version.

## Repository contents

- `src/graft_pillar/train_mask_guided.py`: locked internal-holdout training entry point.
- `src/graft_pillar/evaluate_released.py`: evaluation entry point for the released final model.
- `src/graft_pillar/mask_guided.py`: mask-guided multi-scale Pillar encoder and four-modal fusion head.
- `src/graft_pillar/training_support.py`: LoRA, clinical preprocessing, splitting, loss, and metric utilities.
- `src/graft_pillar/`: cohort validation, preprocessing, and offline Hugging Face cache repair.
- `examples/synthetic/`: two synthetic cases with no patient-derived content.
- `examples/deidentified_four_cases/`: four reviewed, deidentified cases spanning the four binary prediction combinations from the released model.

The Pillar-0 checkpoint is not redistributed. The repository includes the reviewed GRAFT LoRA adapters, fusion model, clinical preprocessor, training history, and checksums in `weights/`. See `VERSION_NOTES.md` for the revised focal-loss objective and final-model fitting rule.

## Input data

Each cohort uses one CSV. There is no required patient count or filename pattern. Paths may be absolute or relative to `paths.data_root` when using `graft-validate`.

Required columns:

| Group | Columns |
|---|---|
| Identifier and files | `case_id`, `ct_path`, `l3_mask_path`, `tumor_mask_path` |
| Binary outcomes | `recurrence`, `complication` |
| Clinical variables | `Age (years)`, `Sex`, `ECOG`, `Charlson index`, `BMI`, `PNI (cont.)`, `NLR (cont.)`, `PLR (cont.)`, `CONUT score`, `SII`, `cT`, `cN`, `cTNM stage`, `Borrmann type`, `Tumor location`, `CEA`, `CA19-9`, `CA72-4` |

`recurrence` and `complication` must be 0/1. CT and masks must share size, spacing, origin, and direction. L3 labels are expected to be 1, 2, and 3; the tumour mask must be binary.

Only variables available at the prediction index time should be included. Postoperative pathology, treatment, follow-up, recurrence-derived, and complication-derived variables must not be used as predictors.

## Installation

Python 3.10 is supported. Install the CUDA build of PyTorch first. The development system used PyTorch 2.8.0 with CUDA 12.6:

```bash
conda create -n graft python=3.10 pip -y
conda activate graft

python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
```

Place the complete Pillar model directory at `models/Pillar0-AbdomenCT` or provide another local path. Training uses `local_files_only=True` and can run without network access.

## Validate a cohort

```bash
cp config.example.yaml config.yaml
graft-validate --config config.yaml
```

Validation checks the CSV schema, unique IDs, binary labels, clinical columns, file availability, image geometry, and mask contents.

The included synthetic cohort can be checked immediately:

```bash
graft-validate --config examples/synthetic/config.yaml
```

The deidentified four-case demonstration cohort can be checked in the same way:

```bash
graft-validate --config examples/deidentified_four_cases/config.yaml
```

## Smoke test

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

graft-train-mask-guided \
  --train-csv /path/to/training_cohort.csv \
  --validation-csv /path/to/internal_validation_cohort.csv \
  --model-dir /path/to/Pillar0-AbdomenCT \
  --project-src src \
  --output-dir outputs/smoke \
  --device cuda:0 \
  --dry-run
```

The smoke test performs a real CT/mask/clinical forward and backward pass.

`--validation-csv` may be omitted while the separate validation cohort is being
prepared. In that training-only mode, epoch selection and full-cohort training
run normally and save `final_model.pt`; validation predictions and metrics are
deferred until a separate validation CSV is supplied.

## Training and separate internal validation

```bash
export TRAIN_CSV=/path/to/training_cohort.csv
export VALIDATION_CSV=/path/to/internal_validation_cohort.csv
export PILLAR_MODEL_DIR=/path/to/Pillar0-AbdomenCT
export OUTPUT_DIR=outputs/mask_guided_internal_validation

mkdir -p "$OUTPUT_DIR"
nohup bash scripts/run_training.sh > "$OUTPUT_DIR/training.log" 2>&1 &
tail -n 60 -F "$OUTPUT_DIR/training.log"
```

The two CSV files represent distinct patient cohorts:

- the complete training cohort is used to develop the model;
- a stratified 20% subset of the training cohort is used temporarily to select the epoch by validation loss;
- a new final model is then trained from initialization on the complete training cohort for the selected number of epochs;
- clinical preprocessing is refitted on the complete training cohort;
- the separate internal validation cohort is evaluated once without refitting or model selection.

Recurrence uses positive-class-weighted binary cross-entropy. Complications use alpha-balanced binary focal loss (default alpha 0.25 and gamma 2), with default task coefficients of 1.0 and 0.75. Output includes the training-only epoch-selection split, learning history, `final_model.pt`, separate validation predictions, `internal_validation_metrics.json`, and percentile-bootstrap 95% confidence intervals.

## Released final model

`weights/graft_v4_final_model.pt` was trained on all 941 development-cohort cases for six epochs after epoch selection within the training cohort. It contains the GRAFT LoRA adapters and non-Pillar fusion parameters; the separate `Pillar0-AbdomenCT` base model remains required. Separate internal-validation performance is pending and no performance claim is made for this released weight.

Evaluate a labelled cohort without retraining:

```bash
graft-evaluate-released \
  --cohort-csv /path/to/validation_cohort.csv \
  --checkpoint weights/graft_v4_final_model.pt \
  --model-dir /path/to/Pillar0-AbdomenCT \
  --project-src src \
  --output-dir outputs/released_model_validation \
  --device cuda:0
```

This writes patient-level probabilities and task-specific modality gates to `predictions.csv`, with discrimination and bootstrap confidence intervals in `metrics.json`. The cohort must contain both binary outcome columns and the documented 18 clinical variables.

## Interpretation of results

Performance is reported only from the separate internal validation cohort. This remains a retrospective internal evaluation and does not establish transportability, clinical utility or prospective benefit. Independent temporal or external validation remains desirable.

## Grad-CAM examples

After fold checkpoints have been produced, six reproducible random examples can be visualized with:

```bash
python scripts/generate_gradcam_examples.py \
  --csv /path/to/training_cohort.csv \
  --splits-csv /path/to/fold_assignments.csv \
  --checkpoint-root /path/to/fold_outputs \
  --model-dir /path/to/Pillar0-AbdomenCT \
  --project-src src \
  --output-dir outputs/interpretability_random6 \
  --device cuda:0 \
  --seed 2026 \
  --n-examples 6
```

The script draws tumour-level CT, tumour contours, L3 regions, recurrence and complication Grad-CAM overlays, and task-specific modality gates. The gates are learned attention weights and must not be described as SHAP values. A modality-level SHAP analysis requires coalition-based ablation of CT, tumour, L3, and clinical embeddings against a training-cohort background.

## Public-data policy

Real patient CSV files, NIfTI volumes, extracted features, outputs, logs, local configurations, and base-model files are excluded by `.gitignore`. The only patient-derived files permitted in this repository are the four explicitly reviewed, deidentified demonstration cases under `examples/deidentified_four_cases/`; their source identifiers, source paths, and NIfTI metadata are removed. These examples demonstrate the input and output format and must not be used to estimate model performance. Review `git status` and staged content before every public release.

## Acknowledgements / 致谢

GRAFT builds on the [Pillar-0 project](https://yalalab.github.io/) and the [YalaLab/Pillar0-AbdomenCT checkpoint](https://huggingface.co/YalaLab/Pillar0-AbdomenCT). The checkpoint is distributed separately under its own license and access terms.

Please cite Pillar-0 when using this pipeline:

```bibtex
@article{pillar0,
  title   = {Pillar-0: A New Frontier for Radiology Foundation Models},
  author  = {Agrawal, Kumar Krishna and Liu, Longchao and Lian, Long and Nercessian, Michael and Harguindeguy, Natalia and Wu, Yufu and Mikhael, Peter and Lin, Gigin and Sequist, Lecia V. and Fintelmann, Florian and Darrell, Trevor and Bai, Yutong and Chung, Maggie and Yala, Adam},
  journal = {arXiv preprint arXiv:2511.17803},
  year    = {2025}
}
```

## License

The GRAFT source code is released under the MIT License. Pillar-0 code and weights retain their original licenses.
