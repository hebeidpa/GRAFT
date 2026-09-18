# Released GRAFT fusion head

`final_fusion_head.pt` is the trained GRAFT imaging-clinical fusion head. It
contains the linear task head, imaging scaling parameters, clinical imputation
and encoding pipeline, and the fixed input schema. The Pillar0 base model is
not embedded in this checkpoint.

Released artifact:

- File: `weights/final_fusion_head.pt`
- SHA-256: `06d1bd5c388a5aac7b9e28eee9a3e3e88d0620490fd119ad10f122716a44aa28`
- Tasks: recurrence and postoperative complication
- Imaging input: 1,152 Pillar0 values plus 67 mask-derived values
- Clinical input: 11 numerical/ordinal variables plus 7 categorical variables
- Head: linear, 1,247 inputs and 2 logits
- Training configuration stored in the checkpoint: 10 repetitions of 5-fold
  cross-validation, seed 2026, learning rate 0.001, weight decay 0.01, maximum
  500 epochs, and early-stopping patience 50

Verify the downloaded artifact before use:

```bash
sha256sum -c weights/SHA256SUMS
```

Use only with the package versions documented in this repository. PyTorch
checkpoints use Python pickle internally; load only files obtained from this
official repository or another trusted source.
