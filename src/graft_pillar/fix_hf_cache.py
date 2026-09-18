from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def cache_folder_name(model_dir: Path) -> str:
    # This matches the local-folder module name used by Transformers dynamic modules.
    return model_dir.resolve().name.replace("-", "-")


def repair(model_dir: str, cache_dir: str | None = None) -> Path:
    source = Path(model_dir).expanduser().resolve()
    if cache_dir:
        target_root = Path(cache_dir).expanduser().resolve()
    else:
        target_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    target = target_root / cache_folder_name(source)
    target.mkdir(parents=True, exist_ok=True)
    (target_root / "__init__.py").touch(exist_ok=True)
    (target / "__init__.py").touch(exist_ok=True)
    copied = 0
    for path in source.glob("*.py"):
        shutil.copy2(path, target / path.name)
        copied += 1
    if copied == 0:
        raise FileNotFoundError(f"No .py files found in {source}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair local Transformers custom-code cache")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    target = repair(args.model_dir, args.cache_dir)
    print(f"已复制模型源码到: {target}")


if __name__ == "__main__":
    main()
