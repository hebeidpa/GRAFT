# Released GRAFT inference artifacts

- `graft_v4_final_model.pt`: frozen LoRA adapters and non-Pillar GRAFT parameters.
- `graft_v4_final_model.clinical_preprocessor.joblib`: frozen preprocessing for
  the 18 documented preoperative clinical variables.
- `SHA256SUMS`: integrity hashes for both files.

The separately licensed `Pillar0-AbdomenCT` base checkpoint is required and is
not redistributed. The two released artifacts contain no patient images,
identifiers, source paths, or patient-level prediction records.

No independent-validation performance claim is made for this weight. Use
`graft-infer` on a prespecified external or internal validation cohort without
refitting the preprocessor or updating model parameters.
