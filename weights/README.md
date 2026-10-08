# Released GRAFT v4 model artifacts

- `graft_v4_final_model.pt`: LoRA adapters and non-Pillar GRAFT parameters.
- `graft_v4_final_model.clinical_preprocessor.joblib`: fitted preprocessing for the 18 documented clinical variables.
- `training_history.csv`: aggregate training loss by epoch.
- `training_summary.json`: cohort size, selected epoch, and validation status.
- `SHA256SUMS`: integrity hashes for these artifacts.

The model was trained on 941 development-cohort cases for six epochs. Separate internal validation remains pending; training loss must not be reported as validation performance.

The separately licensed `Pillar0-AbdomenCT` base checkpoint is required to use these parameters and is not included. No patient images, masks, identifiers, local paths, or patient-level predictions are stored in these artifacts.
