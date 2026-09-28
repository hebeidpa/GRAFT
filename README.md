# GRAFT

GRAFT is a research pipeline for dual-task prediction of gastric cancer recurrence and postoperative complications using abdominal CT, a primary-tumour mask, an L3 body-composition mask, and preoperative clinical variables.

> Research use only. GRAFT is not a medical device and must not be used as a standalone diagnostic or treatment system.

## Current model

This repository contains architecture:

1. CT is processed by `Pillar0-AbdomenCT` with LoRA adapters in the final Atlas stage.
2. The Pillar global feature is projected to a 128-dimensional CT embedding.
3. Tumour and L3 masks are aligned to the Pillar input and used only for multi-scale ROI pooling from intermediate Pillar feature maps.
4. L3 labels 1, 2, and 3 are pooled separately and fused with learned region weights.
5. CT, tumour, L3, and clinical embeddings are combined with separate task-aware gates for recurrence and complication.

Masks are never passed into Pillar-0 as image channels. The previous 67-dimensional handcrafted mask-phenotype branch and its checkpoint are not part of this version.

## Repository contents

- `src/graft_pillar/train_mask_guided.py`: nested cross-validation training entry point.
- `src/graft_pillar/mask_guided.py`: mask-guided multi-scale Pillar encoder and four-modal fusion head.
- `src/graft_pillar/training_support.py`: LoRA, clinical preprocessing, splitting, loss, and metric utilities.
- `src/graft_pillar/`: cohort validation, preprocessing, and offline Hugging Face cache repair.
- `examples/synthetic/`: two synthetic cases with no patient-derived content.

The Pillar-0 checkpoint is not redistributed. Trained GRAFT weights will be published only after the complete five-fold run and artifact review.

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

## Smoke test

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

graft-train-mask-guided \
  --csv /path/to/cohort.csv \
  --model-dir /path/to/Pillar0-AbdomenCT \
  --project-src src \
  --output-dir outputs/smoke \
  --device cuda:0 \
  --dry-run
```

The smoke test performs a real CT/mask/clinical forward and backward pass.

## Five-fold training

```bash
export COHORT_CSV=/path/to/cohort.csv
export PILLAR_MODEL_DIR=/path/to/Pillar0-AbdomenCT
export OUTPUT_DIR=outputs/mask_guided_nested_cv

mkdir -p "$OUTPUT_DIR"
nohup bash scripts/run_training.sh > "$OUTPUT_DIR/training.log" 2>&1 &
tail -n 60 -F "$OUTPUT_DIR/training.log"
```

The outer five-fold split is stratified by the joint recurrence/complication label. For each outer fold:

- the outer test fold is locked;
- preprocessing is fitted on the inner training partition only;
- early stopping and epoch selection use a stratified inner validation partition;
- the selected checkpoint is evaluated once on the untouched outer test fold.



## Acknowledgements

GRAFT builds on the [Pillar-0 project](https://yalalab.github.io/) and the [YalaLab/Pillar0-AbdomenCT checkpoint](https://huggingface.co/YalaLab/Pillar0-AbdomenCT). The checkpoint is distributed separately under its own license and access terms.

Please cite Pillar-0 when using this pipeline:

```bibtex
@article{pillar0,
  title   = {Pillar-0: A New Frontier for Radiology Foundation Models},
  author  = {Agrawal, Kumar Krishna and Liu, Longchao and Lian, Long and Nercessian, Michael and Harguindeguy, Natalia and Wu, Yufu and Mikhael, Peter and Lin, Gigin and Sequist, Lecia V. and Fintelmann, Florian and Darrell, Trevor and Bai, Yutong and Chung, Maggie and Yala, Adam},
  journal = {arXiv preprint arXiv:2511.17803
        
        },
  year    = {2025}
}
```

## License

The GRAFT source code is released under the MIT License. Pillar-0 code and weights retain their original licenses.
