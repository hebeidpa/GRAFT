from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .clinical import clinical_summary, load_clinical_frame, schema_from_config
from .common import configure_logging, load_config, project_paths, write_json
from .manifest import build_manifest


LOG = logging.getLogger(__name__)


def _geometry(image: sitk.Image) -> dict:
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": [float(x) for x in image.GetSpacing()],
        "origin_xyz": [float(x) for x in image.GetOrigin()],
        "direction": [float(x) for x in image.GetDirection()],
    }


def _same_geometry(a: sitk.Image, b: sitk.Image, tol: float) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol, rtol=0)
    )


def validate_case(row, tol: float) -> tuple[dict, list[str]]:
    issues: list[str] = []
    record = {"case_id": row.case_id}
    for key in ["ct_path", "l3_path", "tumor_path"]:
        if not getattr(row, key) or not Path(getattr(row, key)).exists():
            issues.append(f"缺少 {key}")
    if issues:
        return record, issues

    ct = sitk.ReadImage(row.ct_path)
    l3 = sitk.ReadImage(row.l3_path)
    tumor = sitk.ReadImage(row.tumor_path)
    record["ct_geometry"] = _geometry(ct)
    record["l3_geometry"] = _geometry(l3)
    record["tumor_geometry"] = _geometry(tumor)

    if not _same_geometry(ct, l3, tol):
        issues.append("L3 与 CT 的 size/spacing/origin/direction 不一致")
    if not _same_geometry(ct, tumor, tol):
        issues.append("tumor mask 与 CT 的 size/spacing/origin/direction 不一致")

    ct_arr = sitk.GetArrayViewFromImage(ct)
    l3_arr = sitk.GetArrayViewFromImage(l3)
    tumor_arr = sitk.GetArrayViewFromImage(tumor)
    l3_values = np.unique(l3_arr).astype(float).tolist()
    tumor_values = np.unique(tumor_arr).astype(float).tolist()
    record.update(
        {
            "ct_min_hu": float(np.min(ct_arr)),
            "ct_max_hu": float(np.max(ct_arr)),
            "l3_values": l3_values,
            "tumor_values": tumor_values,
            "l3_nonzero_voxels": int(np.count_nonzero(l3_arr)),
            "tumor_nonzero_voxels": int(np.count_nonzero(tumor_arr)),
        }
    )
    if any(abs(v - round(v)) > 1e-6 for v in l3_values):
        issues.append("L3 mask 包含非整数标签")
    if not set(tumor_values).issubset({0.0, 1.0}):
        issues.append(f"tumor mask 不是二值，发现 {tumor_values[:20]}")
    if record["l3_nonzero_voxels"] == 0:
        issues.append("L3 mask 为空")
    if record["tumor_nonzero_voxels"] == 0:
        issues.append("tumor mask 为空")
    return record, issues


def run(config_path: str, max_cases: int | None = None) -> int:
    cfg = load_config(config_path)
    paths = project_paths(cfg)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_warnings = build_manifest(cfg)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    for warning in manifest_warnings:
        LOG.warning(warning)

    clinical_report = None
    clinical_issues: list[str] = []
    clinical_warnings: list[str] = []
    if cfg.get("clinical", {}).get("enabled", False):
        try:
            schema = schema_from_config(cfg)
            clinical_frame, clinical_warnings = load_clinical_frame(
                schema, manifest["case_id"].astype(str).tolist()
            )
            clinical_report = clinical_summary(clinical_frame, schema)
            missing = clinical_report["missing_by_column"]
            for name, count in missing.items():
                if count:
                    clinical_warnings.append(
                        f"临床变量 {name} 缺失 {count} 例；训练时仅用训练折中位数/众数填补"
                    )
        except (FileNotFoundError, ValueError) as exc:
            clinical_issues.append(str(exc))
    for warning in clinical_warnings:
        LOG.warning(warning)

    tol = float(cfg["data"].get("geometry_tolerance", 1e-4))
    records = []
    all_issues: dict[str, list[str]] = {}
    if clinical_issues:
        all_issues["__clinical__"] = clinical_issues
    rows = list(manifest.itertuples(index=False))
    if max_cases is not None:
        rows = rows[:max_cases]
    for idx, row in enumerate(rows, start=1):
        LOG.info("质检 %s/%s: %s", idx, len(rows), row.case_id)
        record, issues = validate_case(row, tol)
        records.append(record)
        if issues:
            all_issues[row.case_id] = issues

    report = {
        "n_labels": int(len(manifest)),
        "n_checked": int(len(rows)),
        "manifest_warnings": manifest_warnings,
        "clinical_warnings": clinical_warnings,
        "clinical": clinical_report,
        "case_issues": all_issues,
        "cases": records,
    }
    report_path = output_dir / "data_validation.json"
    write_json(report, report_path)
    LOG.info("manifest: %s", manifest_path)
    LOG.info("质检报告: %s", report_path)
    if all_issues:
        LOG.error("发现 %s 例问题；请先修复再提取特征。", len(all_issues))
        return 2
    LOG.info("数据质检通过。")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CT/L3/tumor pairs and create manifest.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    raise SystemExit(run(args.config, args.max_cases))


if __name__ == "__main__":
    main()
