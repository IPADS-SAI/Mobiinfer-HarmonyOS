"""Build an independent full-NPU package with smaller embedding Gather models.

Reuse frozen ONNX embedding weights and all other published OMCs unchanged.
Each physical Gather owns a contiguous hidden-dimension slice; NNRT outputs
are concatenated row by row, with only one embedding slice resident at a time.
This is a device-load workaround candidate, not proof of the device root cause.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import torch

from experiment import Experiment, parse_args, save_json, sha256
from package_runtime import seal_release


def embedding_weight(build, entry):
    path = build / entry["onnx"]
    if sha256(path) != entry["onnx_sha256"]:
        raise ValueError("Embedding ONNX changed")
    for relative, digest in entry["external_data_sha256"].items():
        if sha256(path.parent / relative) != digest:
            raise ValueError("Embedding external weights changed")
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    if (len(graph.node) != 1 or graph.node[0].op_type != "Gather" or len(graph.initializer) != 1
            or list(graph.node[0].input) != [graph.initializer[0].name, "token_ids"]
            or any(a.name == "axis" and a.i != 0 for a in graph.node[0].attribute)):
        raise ValueError("Expected one row-Gather embedding graph")
    weight = graph.initializer[0]
    if weight.data_type != onnx.TensorProto.FLOAT or len(weight.dims) != 2:
        raise ValueError("Expected a frozen FP32 embedding reference")
    fields = {item.key: item.value for item in weight.external_data}
    external = (path.parent / fields["location"]).resolve()
    external.relative_to(path.parent.resolve())
    return np.memmap(external, mode="r", dtype="<f4", offset=int(fields.get("offset", 0)), shape=tuple(weight.dims))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source-build", "package", "work", "output", "ddk"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--slice-width", type=int, default=256)
    parser.add_argument("--max-graph-mib", type=float, default=192,
                        help="Reject a compiled physical graph above this package budget")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.slice_width < 1 or args.threads < 1 or not 0 < args.max_graph_mib <= 200:
        parser.error("Positive slice width/threads and graph budget in (0,200] required")
    for field in ("source_build", "package", "work", "output", "ddk"):
        setattr(args, field, getattr(args, field).resolve())
    for target in (args.work, args.output):
        if target.exists() and any(target.iterdir()):
            raise ValueError("Use empty work/output directories")
        for source in (args.source_build, args.package, args.ddk):
            if target == source or source in target.parents or target in source.parents:
                raise ValueError("Sources and destinations must be independent")
    if args.work == args.output or args.work in args.output.parents or args.output in args.work.parents:
        raise ValueError("Work and output directories must be independent")
    source_manifest = json.loads((args.source_build / "manifest.json").read_text())
    release = json.loads((args.package / "release_manifest.json").read_text())
    for record in release["files"]:
        path = args.package / record["path"]
        if path.is_symlink() or path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError("Source package differs from its manifest: " + record["path"])
    config = json.loads((args.package / "config.json").read_text())
    if config.get("format") != "kirin_full_npu_runtime_v1":
        raise ValueError("Expected the dedicated full-NPU package")
    if (config["runtime"]["hidden"] + args.slice_width - 1) // args.slice_width > 128:
        raise ValueError("Runtime supports at most 128 embedding slices")
    budget = int(args.max_graph_mib * 1024 ** 2)
    retained = [g for g in config["graphs"] if not g["name"].startswith("embedding_")]
    if any(g["bytes"] > budget for g in retained):
        raise ValueError("A retained graph exceeds the requested package budget; do not claim all graphs fit")
    profile = source_manifest["profile"]
    if config["platform"] != profile["platform"]:
        raise ValueError("Package/build platform mismatch")
    build_args = parse_args(["--model", profile["source_model"], "--output", str(args.work),
                            "--ddk", str(args.ddk), "--platform", profile["platform"],
                            "--components", "embedding", "--steps", "export,compile",
                            "--prefill-tokens", str(config["runtime"]["prefill"]),
                            "--threads", str(args.threads)])
    torch.set_num_threads(args.threads)
    build = Experiment(build_args)
    build.manifest["embedding_slice_width"] = args.slice_width
    parents = {g["name"]: g for g in source_manifest["graphs"]}
    checks = []
    for mode in ("prefill", "decode"):
        parent = "embedding_" + mode
        entry = parents[parent]
        logical = next(g for g in config["graphs"] if g["name"] == parent)
        if logical.get("embedding_slices") or logical["sha256"] != entry["omc_sha256"]:
            raise ValueError("Package/build embedding mismatch or package already sliced")
        weight = embedding_weight(args.source_build, entry)
        vocabulary, hidden = weight.shape
        width = logical["inputs"][0]["shape"][1]
        if hidden != config["runtime"]["hidden"]:
            raise ValueError("Embedding hidden dimension mismatch")
        samples = np.linspace(0, vocabulary - 1, width, dtype=np.int32) if width > 1 else np.array([vocabulary - 1], dtype=np.int32)
        ids = torch.tensor(samples[None])
        for index, start in enumerate(range(0, hidden, args.slice_width)):
            end = min(start + args.slice_width, hidden)
            values = np.array(weight[:, start:end], copy=True)
            module = torch.nn.Embedding.from_pretrained(torch.from_numpy(values))
            name = parent + "_slice_" + str(index)
            build.export_graph(name, module, [ids], ["token_ids"], ["embeddings"], component="embedding",
                               mode=mode, embedding_parent=parent, hidden_start=start, hidden_end=end)
            exported = next(g for g in build.manifest["graphs"] if g["name"] == name)
            actual = onnx.numpy_helper.to_array(onnx.load(str(args.work / exported["onnx"])).graph.initializer[0])
            if not np.array_equal(actual, values):
                raise ValueError("Sliced embedding weights differ from the original: " + name)
            checks.append(dict(graph=name, hidden_start=start, hidden_end=end, weights_exact=True,
                               torch_ort=exported["torch_ort"]))
            del module, values, actual
            gc.collect()
    build.compile()
    if any(g["omc_bytes"] > budget for g in build.manifest["graphs"]):
        raise ValueError("A sliced embedding graph exceeds the requested package budget")
    args.output.mkdir(parents=True, exist_ok=True)
    replaced = {g["file"] for g in config["graphs"] if g["name"].startswith("embedding_")}
    for record in release["files"]:
        if record["path"] in replaced:
            continue
        source = args.package / record["path"]
        target = args.output / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for logical in config["graphs"]:
        if not logical["name"].startswith("embedding_"):
            continue
        slices = []
        for part in build.manifest["graphs"]:
            if part["embedding_parent"] != logical["name"]:
                continue
            relative = "om/" + part["name"] + ".omc"
            shutil.copyfile(args.work / part["omc"], args.output / relative)
            slices.append(dict(file=relative, hidden_start=part["hidden_start"], hidden_end=part["hidden_end"],
                               bytes=part["omc_bytes"], sha256=part["omc_sha256"]))
        logical["embedding_slices"] = slices
        # The legacy physical path points at the first slice. Older runtimes
        # reject its reduced output shape instead of silently using wrong data.
        logical.update(file=slices[0]["file"], bytes=slices[0]["bytes"], sha256=slices[0]["sha256"])
    save_json(args.output / "config.json", config)
    physical = len(retained) + len(checks)
    report = dict(format="kirin_embedding_slices_v1", device_tested=False, device_load_fix_verified=False,
                  logical_graphs=len(config["graphs"]), physical_graphs=physical,
                  source_package_manifest_sha256=sha256(args.package / "release_manifest.json"),
                  source_build_manifest_sha256=sha256(args.source_build / "manifest.json"),
                  largest_graph_bytes=max([g["bytes"] for g in retained] + [g["omc_bytes"] for g in build.manifest["graphs"]]),
                  embedding_slice_width=args.slice_width, unchanged_non_embedding_graphs=len(retained), checks=checks)
    save_json(args.output / "embedding_slice_validation.json", report)
    readme = args.output / "README.md"
    text = readme.read_text().replace("run as 25 precompiled Kirin9030 OMC graphs.",
                                     f"run as 25 logical stages using {physical} precompiled Kirin9030 OMC graphs.")
    text += ("\n## Embedding loading workaround candidate\n\n"
             "Requires App runtime support for `embedding_slices`. Each original ~594 MiB Gather is split "
             "along its hidden dimension into smaller NPU Gather models. CPU only concatenates their outputs. "
             "Frozen embedding weights and all other OMCs/quantization parameters are unchanged. "
             "Only one embedding slice stays loaded at a time; this increases model-loading overhead, "
             "especially during decoding. The previous package failed to load its first embedding graph "
             "on device. This revision has not yet passed device loading/inference. The public NNRT "
             "source has an extra authentication path for large models; the private device cause "
             "still requires system logs. See `embedding_slice_validation.json`.\n")
    readme.write_text(text, encoding="utf-8")
    summary_path = args.output / "validation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(pure_npu_graphs=physical, logical_graphs=len(config["graphs"]),
                   embedding_slices=report, npu_numerical_alignment_verified=False)
    save_json(summary_path, summary)
    seal_release(args.output)
    print(json.dumps({"output": str(args.output), "physical_graphs": physical,
                      "largest_graph_mib": report["largest_graph_bytes"] / 1024 ** 2}), flush=True)


if __name__ == "__main__":
    main()
