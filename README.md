# GRAFT

GRAFT is a frozen multimodal model for preoperative prediction of 24-month
recurrence and postoperative complications in gastric cancer. It combines an
abdominal CT volume, a primary-tumour mask, an L3 body-composition mask, and 18
preoperative clinical variables.

This public repository is an **inference and evaluation release**. It includes
the trained GRAFT parameters, fitted clinical preprocessor, model architecture,
input validation, a direct inference command, and four deidentified example
cases. Model-development and training code is not included.

> Research use only. GRAFT is not a medical device and must not be used as a
> standalone diagnostic or treatment system.

## Repository layout

```text
GRAFT/
├── examples/deidentified_four_cases/  # runnable CT/mask/clinical examples
├── models/                            # place Pillar0-AbdomenCT here
├── scripts/run_inference.sh           # one-command wrapper
├── src/graft_pillar/                  # frozen inference implementation
├── weights/                           # released GRAFT weight + preprocessor
├── MODEL_CARD.md
└── requirements.txt
```

## Requirements

- Linux x86-64
- Python 3.10
- NVIDIA GPU with a CUDA-compatible PyTorch installation
- the official `Pillar0-AbdomenCT` base-model directory

The released model was tested with PyTorch 2.8.0 and CUDA 12.6. Install the
PyTorch build suitable for your system before installing the remaining
dependencies:

```bash
conda create -n graft python=3.10 pip -y
conda activate graft

python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps .
```

## Model files

The repository includes:

```text
weights/graft_v4_final_model.pt
weights/graft_v4_final_model.clinical_preprocessor.joblib
```

GRAFT also requires the separately licensed Pillar-0 abdomen checkpoint. Obtain
[`YalaLab/Pillar0-AbdomenCT`](https://huggingface.co/YalaLab/Pillar0-AbdomenCT)
from its official provider and place the complete downloaded directory at:

```text
models/Pillar0-AbdomenCT/
```

Inference is local-only (`local_files_only=True`) and does not contact
Hugging Face.

## Input format

Each row in one CSV represents one patient. Filenames and cohort size are not
fixed. Relative image paths are resolved from the directory containing the CSV.

Required file columns:

| Column | Content |
|---|---|
| `case_id` | Unique non-empty case identifier |
| `ct_path` | Abdominal CT in NIfTI format |
| `l3_mask_path` | CT-aligned L3 mask: 1=visceral fat, 2=subcutaneous fat, 3=skeletal muscle |
| `tumor_mask_path` | CT-aligned binary primary-tumour mask |

Required clinical columns:

```text
Age (years), Sex, ECOG, Charlson index, BMI, PNI (cont.),
NLR (cont.), PLR (cont.), CONUT score, SII, cT, cN,
cTNM stage, Borrmann type, Tumor location, CEA, CA19-9, CA72-4
```

For the released checkpoint, categorical values must use the training codebook:

| Variable | Accepted values |
|---|---|
| `Sex` | `1`, `2` |
| `ECOG` | `0`, `1` |
| `cT` | `1`, `2` |
| `cN` | `1`, `2` |
| `cTNM stage` | `II`, `III` |
| `Borrmann type` | `1`, `2`, `3`, `4` |
| `Tumor location` | `1`, `2`, `3` |

The meanings of coded categories must be matched to the study codebook; they
must not be guessed from their numeric order. Unknown categories are ignored by
the frozen encoder and should be reported during external validation.

The outcome columns `recurrence` and `complication` are optional. Omit both for
ordinary prediction. Include both as binary `0/1` columns to calculate
evaluation metrics on a labelled validation cohort.

## Direct inference

From the repository root:

```bash
graft-infer \
  --cohort-csv examples/deidentified_four_cases/cohort.csv \
  --checkpoint weights/graft_v4_final_model.pt \
  --model-dir models/Pillar0-AbdomenCT \
  --output-dir outputs/demo \
  --device cuda:0
```

Or use the wrapper:

```bash
MODEL_DIR=/path/to/Pillar0-AbdomenCT bash scripts/run_inference.sh
```

The command writes:

```text
outputs/demo/
├── predictions.csv
└── metrics.json
```

`predictions.csv` contains patient-level recurrence and complication
probabilities, predictions at the chosen threshold, and the four task-specific
modality-gate weights. Gate weights are learned attention weights and are not
SHAP values.

If both labels are present, `metrics.json` additionally reports ROC-AUC,
PR-AUC, Brier score, log loss, sensitivity, specificity, accuracy, F1, and
bootstrap ROC-AUC confidence intervals. The command never refits the clinical
preprocessor or updates model parameters.

To change the classification threshold or bootstrap count:

```bash
graft-infer ... --threshold 0.40 --bootstrap-samples 2000
```

## Deidentified demonstration cohort

`examples/deidentified_four_cases/` contains four reviewed examples spanning
the four combinations of negative/positive recurrence and complication
predictions at threshold 0.5. These cases demonstrate the file format and
software execution only. They are not a random validation set and must not be
used to estimate model performance.

Verify their integrity before use:

```bash
cd examples/deidentified_four_cases
sha256sum -c SHA256SUMS
cd ../..
```

## Independent validation

For a locked independent validation, prepare a labelled CSV using the same input
schema and run the same `graft-infer` command. Do not tune the model, refit the
preprocessor, alter thresholds after inspecting outcomes, or select cases based
on predictions. Report cohort construction, missingness, segmentation process,
discrimination, calibration, uncertainty, and clinically prespecified
thresholds.

The released weight has no public independent-validation performance claim.
Training loss and the four demonstration cases are not validation evidence.

## Privacy

Do not commit patient data, identifiers, local paths, prediction files, logs, or
institution-only configuration. The four public examples were deidentified and
re-serialized with neutral NIfTI metadata. Always complete local governance and
privacy review before sharing medical images.

## Acknowledgements / 致谢

GRAFT uses image representations from
[`Pillar-0`](https://arxiv.org/abs/2511.17803), developed by the Yala Lab and
collaborators. The required abdomen checkpoint is
[`YalaLab/Pillar0-AbdomenCT`](https://huggingface.co/YalaLab/Pillar0-AbdomenCT).
Pillar-0 is a separate project and retains its own license and access terms.

If you use GRAFT, please also cite:

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
  primaryClass  = {cs.CV}
}
```

## License

GRAFT code is released under the MIT License. Third-party models and datasets
retain their own licenses and access conditions. See `MODEL_CARD.md` for
limitations and intended use.
