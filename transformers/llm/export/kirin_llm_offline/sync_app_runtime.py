"""Copy the portable runtime and its existing MNN dependencies into the App.

Run after editing runtime sources. Only entry/src/main/cpp/full_npu is written;
the original App's MNN shared library and three existing executors are untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sync(app):
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    target = app / "entry/src/main/cpp/full_npu"
    previous = target / "sources.json"
    if previous.exists():
        for record in json.loads(previous.read_text(encoding="utf-8")):
            existing = target / record["path"]
            if not existing.is_file() or hashlib.sha256(existing.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("App runtime was edited locally; merge before syncing: " + str(existing))
    files = []
    files.extend((p, p.name) for p in (here / "runtime").glob("*")
                 if p.suffix in (".hpp", ".cpp") and p.name != "host_driver.cpp" and not p.name.startswith("test_"))
    for name in ("tokenizer.cpp", "tokenizer.hpp", "unicode.cpp", "unicode.hpp", "unicode_data.cpp", "unicode_data.hpp"):
        files.append((repo / "transformers/llm/engine/src/tokenizer" / name, "tokenizer/" + name))
    for path in (repo / "3rd_party/rapidjson").rglob("*"):
        if path.is_file():
            files.append((path, "rapidjson/" + str(path.relative_to(repo / "3rd_party/rapidjson"))))
    files.append((repo / "3rd_party/imageHelper/stb_image.h", "stb_image.h"))
    records = []
    for source, relative in files:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        records.append({"path": relative, "source": str(source.relative_to(repo)),
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    (target / "sources.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print("Synchronized", len(records), "files to", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    sync(parser.parse_args().app)
