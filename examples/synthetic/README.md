# Synthetic public example

This folder contains two entirely synthetic, non-clinical cases. Each case has
one CT-like NIfTI volume, one L3 multi-label mask, one tumour mask, and a row in
`cohort.csv`. No file or value was derived from a patient.

Use it to verify schema and geometry handling:

```bash
cp config.example.yaml config.yaml
graft-validate --config examples/synthetic/config.yaml
```

The two cases are insufficient for meaningful model training or performance
estimation. The example config reduces fold counts only as documentation; its
outputs must not be interpreted scientifically.

Regenerate the files with:

```bash
python scripts/create_synthetic_example.py
```
