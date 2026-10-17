#!/usr/bin/env python3
"""Download configured ModelScope sources without replacing unrelated local data."""
import argparse
import json
from pathlib import Path

MARKER = ".mobi-model-source.json"


def complete_model(directory, kind):
    required = ["config.json", "tokenizer.json"]
    if kind == "gptq":
        required += ["quantization_config.json", "model.safetensors"]
    if any(not (directory / name).is_file() or not (directory / name).stat().st_size for name in required):
        return False
    if kind == "gptq":
        return True
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            return bool(shards) and all(
                not Path(name).is_absolute() and ".." not in Path(name).parts
                and (directory / name).is_file() and (directory / name).stat().st_size
                for name in shards)
        except (ValueError, KeyError, AttributeError, TypeError):
            return False
    return (directory / "model.safetensors").is_file() and (directory / "model.safetensors").stat().st_size > 0


def source_ready(directory, model_id, revision, kind):
    marker = directory / MARKER
    origin = None
    if marker.exists():
        origin = json.loads(marker.read_text(encoding="utf-8"))
        if model_id and (origin.get("model_id"), origin.get("revision"), origin.get("kind")) != (model_id, revision, kind):
            raise ValueError("Downloaded model source changed; choose a new local model directory: {}".format(directory))
    # An interrupted download is resumed even if the minimum files already exist.
    # Only a completed SDK snapshot may mark a managed directory as complete.
    if complete_model(directory, kind) and (origin is None or origin.get("completed")):
        return True
    if not model_id:
        raise ValueError("Local {} model is missing or incomplete: {}. Configure an existing model directory "
                         "or its ModelScope model ID.".format(kind, directory))
    if directory.exists() and (not directory.is_dir() or (any(directory.iterdir()) and origin is None)):
        raise ValueError("Incomplete local model directory will not be overwritten: {}. "
                         "Repair it or choose a new empty model directory.".format(directory))
    return False


def save_origin(directory, value):
    path = directory / MARKER
    temporary = directory / (MARKER + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_source(directory, model_id, revision, kind):
    directory = directory.expanduser().resolve()
    if source_ready(directory, model_id, revision, kind):
        print("[mobi-model] Using local {} model: {}".format(kind, directory), flush=True)
        return
    try:
        from modelscope import snapshot_download
    except ImportError as error:
        raise RuntimeError("ModelScope SDK is unavailable in the model-export Python environment; "
                           "install it there before downloading models (python -m pip install modelscope).") from error
    directory.mkdir(parents=True, exist_ok=True)
    origin = {"model_id": model_id, "revision": revision, "kind": kind, "completed": False}
    save_origin(directory, origin)
    print("[mobi-model] Downloading {}@{} -> {}".format(model_id, revision, directory), flush=True)
    snapshot_download(model_id=model_id, revision=revision, local_dir=str(directory))
    if not complete_model(directory, kind):
        raise ValueError("ModelScope snapshot is missing required {} model files: {}".format(kind, directory))
    save_origin(directory, dict(origin, completed=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--kind", choices=("original", "gptq"), required=True)
    args = parser.parse_args()
    try:
        prepare_source(args.directory, args.model_id, args.revision, args.kind)
    except Exception as error:
        parser.exit(1, "[mobi-model] Download failed: {}\n".format(error))


if __name__ == "__main__":
    main()
