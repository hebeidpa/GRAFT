from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive exactly the Git-tracked public files")
    parser.add_argument("--output", default="GRAFT-v2.0.0.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_text in tracked:
            path = root / relative_text
            if path.is_file():
                archive.write(path, Path("GRAFT-v2.0.0") / relative_text)
    print(output)


if __name__ == "__main__":
    main()
