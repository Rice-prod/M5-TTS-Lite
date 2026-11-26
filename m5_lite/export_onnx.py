import argparse
from pathlib import Path
import runpy
import sys

from .common import activate_upstream
from .infer import register_bundle


def main():
    parser = argparse.ArgumentParser(description="Export a Lite checkpoint with the pinned M5 ONNX exporter")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args, options = parser.parse_known_args()
    root = activate_upstream(args.upstream)
    register_bundle(args.model)
    script = str(root / "onnx" / "export.py")
    sys.argv = [script, "--model", str(args.model.resolve()), *options]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
