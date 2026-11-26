import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import contextmanager


UPSTREAM_REVISION = "b6050656c425a994e604f6e46c884e869f54261c"


def activate_upstream(path):
    root = Path(path).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION or not (root / "generator.py").is_file():
        raise ValueError(f"Expected M5-TTS checkout at {UPSTREAM_REVISION}: {root}")
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(root / "onnx"))
    return root


def digest(path):
    with Path(path).open("rb") as stream:
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
        return hasher.hexdigest()


def bundle_id(path):
    files = [path / name for name in ("config.json", "model.safetensors", "conditioning.json")]
    files += sorted(item for item in (path / "tokenizer").rglob("*") if item.is_file())
    return {item.relative_to(path).as_posix(): digest(item) for item in files}


def read_json(path):
    return Path(path).read_text(encoding="utf-8")


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def records(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


@contextmanager
def new_directory(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as temporary:
        staging = Path(temporary) / "contents"
        staging.mkdir()
        yield staging
        os.rename(staging, path)
