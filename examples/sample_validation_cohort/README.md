# Sample validation cohort

This directory contains four reviewed, deidentified cases for verifying the
public GRAFT inference workflow and released model files. Each case contains an
abdominal CT, a binary primary-tumour mask, an L3 body-composition mask, the 18
required clinical variables, and two reference labels.

Source identifiers and paths were replaced. CT and masks were transformed to a
1.25-mm isotropic, 384 x 384 x 384 grid, then re-serialized with neutral origin
and direction and without source header metadata.

At threshold 0.5, the published reference predictions span all four output
combinations:

| Public ID | Recurrence | Complication |
|---|---:|---:|
| `PUBLIC_001` | Negative | Negative |
| `PUBLIC_002` | Negative | Positive |
| `PUBLIC_003` | Positive | Negative |
| `PUBLIC_004` | Positive | Positive |

The cases were deliberately selected to display all four prediction
combinations. They form a software reproducibility cohort, not a random or
independent validation sample. Do not use metrics from these four cases as a
model-performance claim.

## Files

- `cohort.csv`: runnable inputs and reference labels.
- `images/`: deidentified CT, tumour-mask, and L3-mask NIfTI files.
- `expected_predictions.csv`: probabilities, thresholded predictions, and
  modality-gate weights produced by the released weight.
- `SHA256SUMS`: file-integrity hashes.

## Run inference

From the repository root:

```bash
graft-infer \
  --cohort-csv examples/sample_validation_cohort/cohort.csv \
  --checkpoint weights/graft_v4_final_model.pt \
  --model-dir /path/to/Pillar0-AbdomenCT \
  --output-dir outputs/sample_validation \
  --device cuda:0
```

Verify that the new output matches the published reference:

```bash
python scripts/verify_sample_validation.py \
  --actual outputs/sample_validation/predictions.csv
```

The two outcome columns are included so the same run also exercises the
evaluation path. Delete both `recurrence` and `complication` columns to test
label-free prediction.

## Check file integrity

```bash
cd examples/sample_validation_cohort
sha256sum -c SHA256SUMS
```
