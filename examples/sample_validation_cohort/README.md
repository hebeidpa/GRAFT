# Sample validation cohort

This directory contains ten reviewed, deidentified cases for verifying the
public GRAFT inference workflow and released model files. Each case contains an
abdominal CT, a binary primary-tumour mask, an L3 body-composition mask, the 18
required clinical variables, and two reference labels.

Source identifiers and paths were replaced. CT and masks were re-serialized to
remove source-specific header metadata while preserving pixel values and the
spatial relationship among the three volumes.

At threshold 0.5, the published reference results contain six fully correct
cases and four cases with a misclassified complication outcome:

| Public ID | Recurrence result | Complication result |
|---|---:|---:|
| `PUBLIC_001` | Correct | Correct |
| `PUBLIC_002` | Correct | Correct |
| `PUBLIC_003` | Correct | Correct |
| `PUBLIC_004` | Correct | Correct |
| `PUBLIC_005` | Correct | Misclassified |
| `PUBLIC_006` | Correct | Misclassified |
| `PUBLIC_007` | Correct | Misclassified |
| `PUBLIC_008` | Correct | Misclassified |
| `PUBLIC_009` | Correct | Correct |
| `PUBLIC_010` | Correct | Correct |

The cases were deliberately selected to display different prediction patterns
and known error modes. They form a software reproducibility cohort, not a
random or independent validation sample. Do not use metrics from these ten
cases as a model-performance claim.

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
