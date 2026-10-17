"""Package audited static graphs for the opt-in HarmonyOS full-NPU runtime.

No calibration checkpoints, source weights or compiler intermediates are shipped.
The package is separate from every existing CPU/ViT-only model repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seal_release(output):
    """Refresh checksums after attaching host evidence; never hash the lists themselves."""
    files = [{"path": str(path.relative_to(output)), "size": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(output.rglob("*")) if path.is_file() and
             path.relative_to(output).as_posix() not in ("SHA256SUMS", "release_manifest.json")]
    (output / "SHA256SUMS").write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in files), encoding="utf-8")
    save(output / "release_manifest.json", {"format": "kirin_full_npu_release_v1", "files": files})
    return files


def package(build, model, tokenizer, output, height, width, max_new_tokens, app_validation=None):
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    manifest = json.loads((build / "manifest.json").read_text(encoding="utf-8"))
    source = json.loads((model / "config.json").read_text(encoding="utf-8"))
    profile = manifest["profile"]
    kv_dtype = profile.get("kv_dtype")
    if kv_dtype not in ("int8", "fp16") or profile.get("precision") != "w4a16":
        raise ValueError("The device runtime requires W4A16 with INT8 or FP16 KV")
    if digest(model / "config.json") != profile["source_config_sha256"]:
        raise ValueError("Source model config differs from the compiled profile")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty release directory")
    text, vision = source["text_config"], source["vision_config"]
    patch, merge = vision["patch_size"], vision["spatial_merge_size"]
    if height % (patch * merge) or width % (patch * merge) or height * width // patch ** 2 != profile["vision_tokens"]:
        raise ValueError("Image shape must match the compiled vision profile")
    output.mkdir(parents=True, exist_ok=True)
    graphs = []
    for entry in manifest["graphs"]:
        path = Path(entry["omc"])
        artifact = build / path
        if not entry.get("pure_npu") or digest(artifact) != entry["omc_sha256"]:
            raise ValueError("Unverified offline graph: " + entry["name"])
        relative = "om/" + entry["name"] + ".omc"
        dst = output / relative
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(artifact, dst)
        if digest(dst) != entry["omc_sha256"]:
            raise ValueError("Graph copy hash mismatch")
        graph = {"name": entry["name"], "file": relative,
                       "inputs": entry["inputs"], "output_info": entry["output_info"],
                       "output_aliases": entry.get("output_aliases", {}),
                       "layer_start": entry.get("layer_start", 0), "layer_end": entry.get("layer_end", 0),
                       "sha256": entry["omc_sha256"], "bytes": artifact.stat().st_size}
        if entry.get("external_weights"):
            records = entry["external_weights"]
            identity = json.dumps([(Path(w["file"]).name, w["sha256"]) for w in records]).encode()
            directory = "weights/" + hashlib.sha256(identity).hexdigest()[:24]
            names = []
            for record in records:
                src = build / record["file"]
                if digest(src) != record["sha256"] or src.stat().st_size != record["bytes"]:
                    raise ValueError("External weight changed: " + entry["name"])
                name = src.name
                if name in names:
                    raise ValueError("Duplicate external weight filename")
                names.append(name)
                dst = output / directory / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copyfile(src, dst)
                if digest(dst) != record["sha256"]:
                    raise ValueError("External weight copy hash mismatch")
            graph["external_weights"] = {"directory": directory, "files": names,
                                         "bytes": sum(w["bytes"] for w in records)}
        graphs.append(graph)
    fused_vision = profile.get("fused_vision", False)
    expected = {"embedding_prefill", "embedding_decode", "lm_head"}
    expected.update("llm_" + mode + "_" + str(i) for mode in ("prefill", "decode")
                    for i in range((text["num_hidden_layers"] + profile["layers_per_chunk"] - 1) // profile["layers_per_chunk"]))
    if fused_vision:
        expected.add("vision_full")
        full = next(g for g in manifest["graphs"] if g["name"] == "vision_full")
        if full.get("deepstack_layers") != vision["deepstack_visual_indexes"] or full.get("layer_start") != 0 or full.get("layer_end") != vision["depth"]:
            raise ValueError("Fused visual graph DeepStack coverage differs from source model")
    else:
        expected.update(("vision_pre", "vision_post"))
        expected.update("vision_blocks_" + str(i) for i in range((vision["depth"] + profile["vision_layers_per_chunk"] - 1) // profile["vision_layers_per_chunk"]))
    if {g["name"] for g in graphs} != expected:
        raise ValueError("The build does not contain the complete VLM")
    shutil.copyfile(tokenizer, output / "tokenizer.mtok")
    grid = torch.tensor([[1, height // patch, width // patch]])
    side = int(vision["num_position_embeddings"] ** .5)
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=side, mode="bilinear", align_corners=True, spatial_merge_size=merge)
    # Transformers Qwen3-VL defaults are checked against the actual exported
    # vision_pre graph by the replay test before publishing.
    positions = get_vision_position_ids(grid, merge)
    dim = vision["hidden_size"] // vision["num_heads"]
    inverse = 1.0 / (10000 ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
    angles = (positions[..., None].float() * inverse).flatten(1)
    angles = torch.cat((angles, angles), -1)
    with (output / "vision_constants.bin").open("wb") as f:
        for value in (indices.int().numpy(), weights.numpy(), angles.cos().numpy(), angles.sin().numpy()):
            np.asarray(value).tofile(f)
    rope = text.get("rope_scaling", text.get("rope_parameters", {}))
    runtime = dict(capacity=profile["context"], prefill=profile["prefill_tokens"],
                   layers=text["num_hidden_layers"], hidden=text["hidden_size"], kv_heads=text["num_key_value_heads"],
                   head_dim=text["head_dim"], kv_group=profile.get("kv_group_size", 32), kv_dtype=kv_dtype,
                   kv_recent=profile.get("kv_residual_tokens", 0) if kv_dtype == "int8" else 0,
                   kv_sink=profile.get("kv_sink_tokens", 0) if kv_dtype == "int8" else 0,
                   image_height=height, image_width=width, patch=patch, merge=merge,
                   temporal=vision["temporal_patch_size"], vision_hidden=vision["hidden_size"],
                   vision_heads=vision["num_heads"], image_token=source["image_token_id"],
                   pad_token=source["pad_token_id"], max_new_tokens=max_new_tokens,
                   rope_theta=text["rope_theta"], mrope_section=rope["mrope_section"])
    external = any("external_weights" in g for g in graphs)
    if external:
        runtime.update(npu_max_resident_graphs=4, npu_resident_model_mib=2048, npu_persistent_kv_mib=256,
                       npu_incremental_kv=True)
    if fused_vision:
        runtime.update(vision_layers=vision["depth"], vision_deepstack_layers=vision["deepstack_visual_indexes"])
    version = "kirin_full_npu_runtime_v3" if fused_vision else ("kirin_full_npu_runtime_v2" if external or kv_dtype == "fp16" else "kirin_full_npu_runtime_v1")
    save(output / "config.json", dict(format=version, platform=profile["platform"],
         experimental=True, tokenizer_file="tokenizer.mtok", vision_constants="vision_constants.bin",
         runtime=runtime, graphs=graphs))
    save(output / "llm_config.json", dict(model_type="qwen3_vl", is_visual=True,
         hidden_size=text["hidden_size"], layer_nums=text["num_hidden_layers"],
         max_position_embeddings=profile["context"]))
    evidence = {}
    for name in ("kv_validation_torch.json", "kv_validation_onnx.json", "artifact_audit.json",
                 "quantization_audit.json", "image_quantization_audit.json", "image_logit_audit.json",
                 "cpu_text_canary.json", "repair_summary.json", "fused_vision_validation.json"):
        if (build / name).is_file():
            # Reports are evidence, not files needed by the runtime.
            report = json.loads((build / name).read_text(encoding="utf-8"))
            save(output / name, report)
            evidence[name] = report
    if app_validation:
        report = json.loads(app_validation.read_text(encoding="utf-8"))
        if not report.get("cases") or not all(case.get("passed") for case in report["cases"]):
            raise ValueError("C++ App validation report did not pass")
        save(output / "cpp_host_validation.json", report)
    save(output / "validation_summary.json", dict(device_tested=False, app_integrated=True, hap_build_verified=False,
         npu_numerical_alignment_verified=False, pure_npu_graphs=len(graphs), cache=manifest["cache"],
         host_validation=manifest.get("host_validation", {}), kv_validation=manifest.get("kv_validation", {}),
         source_config_sha256=profile["source_config_sha256"],
         build_manifest_sha256=digest(build / "manifest.json"),
         activation_range_multiplier=profile.get("activation_range_multiplier", 1),
         output_range_multiplier=profile.get("output_range_multiplier", 1),
         fused_vision=fused_vision, visual_graphs=sum(name.startswith("vision_") for name in expected),
         vision_deepstack_layers=vision["deepstack_visual_indexes"],
         static_math=profile.get("static_math", "original")))
    kv_description = ("FP16" if kv_dtype == "fp16" else
                      "INT8 with FP32 scales and configured FP16 protected windows")
    readme = f"""---
tags:
- Kirin9030
- W4A16
- experimental
---

# Experimental full VLM offline NPU model

Requires the updated mobiinfra-oh App and its **全图 NPU** mode. The previous
CPU, online NPU and ViT-only offline modes use different model packages.

Embedding, all ViT blocks/mergers, all LLM prefill/decode layers and LM head
run as {len(graphs)} precompiled Kirin9030 OMC graphs. CPU handles tokenization, image
decoding/resizing, positions/masks, greedy sampling and compressed KV storage.
“Full” describes model coverage; the VLM is scheduled across static graphs.

Main matrix weights: W4 / group{profile.get('group_size', 64)} / {profile.get('weight_method', 'rtn').upper()}.
SDK A16 uses INT16 activation quantization; embeddings remain floating point.
KV: {kv_description}. Context: **{profile['context']} total tokens**,
including image tokens and generated output. Every image is resized to
256×608 (152 image tokens); each request rebuilds its conversation cache.
Still images and text are supported. No audio/video models are included.

External graph/weight separation: {external}. Graph arithmetic: {profile.get('static_math', 'original')}.
See validation_summary.json and the attached evidence for this exact build.
This is an experimental device test candidate; NPU accuracy, memory and speed are unverified.
Compiler success does not establish device correctness.

Download this repository into a new model directory, select 全图 NPU, enable
the App Debug switch before loading, then test text and distinct images.
Export runtime logs containing `[FULL_NPU]` descriptor, execution, cache and
timing records. Loading/execution errors are reported without CPU fallback.
The App keeps only one LLM phase resident and unloads each visual component.

See `validation_summary.json`, the two KV reports and `SHA256SUMS` for evidence.
"""
    if version.endswith(("v2", "v3")):
        kv_warning = ("The included KV numerical validation FAILED its unchanged tolerance.\n"
                      if any(name.startswith("kv_validation_") and report.get("passed") is False
                             for name, report in evidence.items()) else "")
        readme = f"""# Experimental Kirin9030 full VLM, external W4 weights

Requires the updated App with `{version}` support and 全图 NPU mode.
{len(graphs)} offline OMC graphs cover the VLM. External weight files are loaded
after executor construction using `HMS_HiAIExecutor_InitWeights`; an absent API
is an explicit error. Only byte-identical weight files are shared on disk.

LLM layers per graph: {profile['layers_per_chunk']}. Weight quantization remains
the frozen SDK W4/group{profile['group_size']} GPTQ state. KV storage: {kv_dtype}.
INT16 activation input range factor: {profile.get('activation_range_multiplier', 1)};
additional output range factor: {profile.get('output_range_multiplier', 1)}.
Static graph arithmetic: {profile.get('static_math', 'original')}.
Image processing, positions, sampling and KV commits run on the CPU. The neural
network components, including embedding, run on the NPU.

Context: {profile['context']} total tokens. Image grid: {height}x{width}.
Model file residency budget includes external weight sizes; it does not measure
actual driver memory. The updated App evicts sessions gradually on resource
pressure, preserves KV buffers when possible, and restores budgets at phase
and request boundaries. Actual model residency remains device-dependent.

{kv_warning}CPU answer quality must be reviewed separately from compiler and transfer tests.
No device precision or throughput claim: inspect `validation_summary.json` and
the accompanying host audits before device testing. This package is separate
from the earlier embedded-weight release and requires a newly built App.
"""
    if fused_vision:
        readme += f"""
The split visual graphs are fused into one `vision_full` OMC. It includes
patch/position embedding, all {vision['depth']} visual blocks, the final merger,
and {len(vision['deepstack_visual_indexes'])} separate DeepStack mergers.
DeepStack taps remain after visual blocks {vision['deepstack_visual_indexes']}
(zero-based); their projected outputs feed the first {len(vision['deepstack_visual_indexes'])} language layers in order.
See `fused_vision_validation.json` for CPU equivalence against the split build.
This change reduces visual graph transitions; decoder residency and external
weight initialization can still dominate device latency. No device speedup is claimed.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    files = seal_release(output)
    print(json.dumps({"output": str(output), "graphs": len(graphs), "bytes": sum(f["size"] for f in files)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "model", "tokenizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-height", type=int, default=608)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--app-validation", type=Path, help="Completed verify_app_runtime.py report.json")
    args = parser.parse_args()
    package(args.build, args.model, args.tokenizer, args.output, args.image_height, args.image_width, args.max_new_tokens,
            args.app_validation)


if __name__ == "__main__":
    main()
