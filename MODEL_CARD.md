# GRAFT fusion-head model card

## Intended use

Research on recurrence and postoperative-complication risk modelling in gastric
cancer using abdominal CT, segmentation-derived measurements, and preoperative
clinical variables.

## Inputs

The fixed schema is documented in `README.md`. Imaging features must be created
with the same Pillar0-AbdomenCT preprocessing implemented in this repository.

## Outputs

Two probabilities: recurrence and postoperative complication. A threshold of
0.5 is reported for descriptive classification metrics; users should not assume
that it is clinically optimal.

## Released checkpoint

The repository includes `weights/final_fusion_head.pt`, a 1,247-input linear
fusion head with two outputs. Its input consists of 1,219 imaging and
mask-derived features and 28 encoded clinical features. The checkpoint also
stores the fitted image scaler and clinical preprocessing pipeline. Its SHA-256
digest is recorded in `weights/SHA256SUMS`.

The artifact was serialized with scikit-learn 1.3.2. Use the pinned package
versions in `requirements.txt` to avoid estimator-serialization incompatibility.
The accepted categorical tokens are listed in `README.md`. Their semantic
mapping must follow the development-cohort codebook; integer codes must not be
reinterpreted from their numeric order.

## Validation status

The training pipeline reports leakage-controlled repeated cross-validation.
Performance from the development cohort is internal validation. Independent
temporal and multi-centre validation, calibration assessment, and clinical
utility analysis are required before any prospective use.

No performance number is published in this model card because the corresponding
final machine-readable cross-validation report was not supplied with the
released checkpoint. Users should generate and retain `metrics.json` when they
retrain the model and should report confidence intervals and cohort definitions
with any performance claim.

## Limitations

- Predictions may shift with scanner protocol, reconstruction, segmentation,
  patient population, staging practice, laboratory assay, or missing-data
  mechanism.
- The model does not replace clinical assessment.
- Linear coefficients are not causal effects.
- The released fusion head depends on a separately licensed base model.
- Unknown categorical levels at inference are encoded as all-zero one-hot
  blocks and should be reported during validation.

## Privacy

The checkpoint does not contain raw images, row-level patient data, or patient
identifiers. The repository examples are synthetic.
