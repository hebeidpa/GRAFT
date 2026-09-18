from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from .clinical import clinical_summary, load_clinical_frame, schema_from_config
from .common import configure_logging, load_config, project_paths, write_json


LOG = logging.getLogger(__name__)


def run(config_path: str) -> int:
    cfg = load_config(config_path)
    paths = project_paths(cfg)
    feature_path = paths["output_dir"] / "features.npz"
    manifest_path = paths["output_dir"] / "manifest.csv"
    if feature_path.exists():
        with np.load(feature_path, allow_pickle=False) as data:
            case_ids = data["case_ids"].astype(str).tolist()
        case_source = str(feature_path)
    elif manifest_path.exists():
        case_ids = pd.read_csv(manifest_path, dtype={"case_id": str})["case_id"].tolist()
        case_source = str(manifest_path)
    else:
        raise FileNotFoundError(
            f"未找到 {feature_path} 或 {manifest_path}; 请先运行 graft-validate/graft-extract"
        )

    output_path = paths["output_dir"] / "clinical_validation.json"
    try:
        schema = schema_from_config(cfg)
        clinical, warnings = load_clinical_frame(schema, case_ids)
        summary = clinical_summary(clinical, schema)
        for name, count in summary["missing_by_column"].items():
            if count:
                warnings.append(
                    f"临床变量 {name} 缺失 {count} 例；训练时仅用训练折中位数/众数填补"
                )
        report = {
            "ok": True,
            "case_source": case_source,
            "warnings": warnings,
            "clinical": summary,
        }
        for warning in warnings:
            LOG.warning(warning)
        write_json(report, output_path)
        LOG.info("临床数据质检通过: %s", output_path)
        return 0
    except (FileNotFoundError, ValueError) as exc:
        write_json(
            {"ok": False, "case_source": case_source, "errors": [str(exc)]},
            output_path,
        )
        LOG.error("临床数据质检失败: %s", exc)
        LOG.error("报告: %s", output_path)
        return 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and align clinical variables with extracted imaging cases"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    raise SystemExit(run(args.config))


if __name__ == "__main__":
    main()
