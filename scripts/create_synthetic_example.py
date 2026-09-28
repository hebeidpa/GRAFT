from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


def write_image(array: np.ndarray, path: Path, is_mask: bool = False) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.5, 1.5, 2.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image, str(path), useCompression=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "examples" / "synthetic"
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    rows = []
    shape = (24, 32, 32)  # z, y, x; deliberately small synthetic smoke-test data
    zz, yy, xx = np.indices(shape)
    for index, case_id in enumerate(["SYNTH_001", "SYNTH_002"]):
        ct = rng.normal(-80 + index * 15, 120, size=shape).astype(np.int16)
        tumor = (
            (zz - (12 + index)) ** 2
            + (yy - 16) ** 2
            + (xx - (15 + index)) ** 2
            <= 16
        ).astype(np.uint8)
        l3 = np.zeros(shape, dtype=np.uint8)
        slab = np.abs(zz - 12) <= 1
        l3[slab & (((yy - 16) ** 2 + (xx - 16) ** 2) <= 100)] = 1
        l3[slab & (yy < 10)] = 2
        l3[slab & (yy > 22)] = 3
        ct_path = images / f"{case_id}_ct.nii.gz"
        l3_path = images / f"{case_id}_l3.nii.gz"
        tumor_path = images / f"{case_id}_tumor.nii.gz"
        write_image(ct, ct_path)
        write_image(l3, l3_path, is_mask=True)
        write_image(tumor, tumor_path, is_mask=True)
        rows.append(
            {
                "case_id": case_id,
                "ct_path": f"examples/synthetic/images/{ct_path.name}",
                "l3_mask_path": f"examples/synthetic/images/{l3_path.name}",
                "tumor_mask_path": f"examples/synthetic/images/{tumor_path.name}",
                "recurrence": index,
                "complication": 1 - index,
                "Age (years)": 55 + 10 * index,
                "Sex": "female" if index == 0 else "male",
                "ECOG": str(index),
                "Charlson index": index,
                "BMI": 22.5 + index,
                "PNI (cont.)": 48.0 - index,
                "NLR (cont.)": 2.1 + index,
                "PLR (cont.)": 120.0 + 10 * index,
                "CONUT score": index + 1,
                "SII": 500.0 + 100 * index,
                "cT": "cT2" if index == 0 else "cT3",
                "cN": "cN0" if index == 0 else "cN1",
                "cTNM stage": "II" if index == 0 else "III",
                "Borrmann type": "II" if index == 0 else "III",
                "Tumor location": "body" if index == 0 else "antrum",
                "CEA": 2.5 + index,
                "CA19-9": 18.0 + index,
                "CA72-4": 3.0 + index,
            }
        )
    pd.DataFrame(rows).to_csv(root / "cohort.csv", index=False, encoding="utf-8-sig")
    print(root)


if __name__ == "__main__":
    main()
