# Deidentified four-case demonstration cohort

This directory contains four deidentified examples selected after inference with the released GRAFT v4 model. At the documented threshold of 0.5, the four examples span:

| Public ID | Recurrence prediction | Complication prediction |
|---|---:|---:|
| `PUBLIC_001` | Negative | Negative |
| `PUBLIC_002` | Negative | Positive |
| `PUBLIC_003` | Positive | Negative |
| `PUBLIC_004` | Positive | Positive |

Each example contains an abdominal CT, a binary primary-tumour mask, an L3 body-composition mask, and the clinical variables required by the model. Source identifiers and paths were replaced. CT and masks were transformed with the same public GRAFT preprocessing used by the model to a 1.25-mm isotropic, 384 x 384 x 384 grid, then re-serialized with neutral origin and direction and without source header metadata. Predictions were recomputed from these released files with the released v4 weight.

`cohort.csv` contains the runnable model inputs and reference labels. `predictions.csv` records the released model's probabilities and thresholded predictions. The examples were deliberately selected to illustrate all four output combinations; they are not a random or independent validation sample and must not be used to estimate discrimination, calibration, or clinical utility.

Validate the files:

```bash
graft-validate --config examples/deidentified_four_cases/config.yaml
(cd examples/deidentified_four_cases && sha256sum -c SHA256SUMS)
```

Run released-model inference:

```bash
graft-evaluate-released \
  --cohort-csv examples/deidentified_four_cases/cohort.csv \
  --checkpoint weights/graft_v4_final_model.pt \
  --model-dir /path/to/Pillar0-AbdomenCT \
  --project-src src \
  --output-dir outputs/deidentified_four_cases \
  --device cuda:0
```
