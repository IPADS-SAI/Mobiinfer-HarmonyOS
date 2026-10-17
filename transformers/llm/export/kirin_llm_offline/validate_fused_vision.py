"""Audit fused visual outputs against the frozen split W4 CPU reference.

This checks graph fusion, not the original quantization's absolute accuracy,
and does not execute OMC. Every DeepStack output is checked separately.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import onnx
from PIL import Image
import torch
from transformers import AutoConfig
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

from experiment import ONNXGraphs, metrics, save_json, sha256


def validate(args):
    old, new = ONNXGraphs(args.source_build, args.threads), ONNXGraphs(args.build, args.threads)
    source, target = old.manifest, new.manifest
    if source["profile"].get("fused_vision") or not target["profile"].get("fused_vision"):
        raise ValueError("Expected split source and fused candidate")
    if source["profile"] != {k: v for k, v in target["profile"].items() if k != "fused_vision"}:
        raise ValueError("Fusion must preserve source shapes, math, quantization and KV profile")
    frozen = source["calibration"]["checkpoint_sha256"]
    if target["calibration"]["checkpoint_sha256"] != frozen:
        raise ValueError("Frozen checkpoint identity changed")
    entry = new.entries["vision_full"]
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True).vision_config
    taps = list(cfg.deepstack_visual_indexes)
    if entry["deepstack_layers"] != taps:
        raise ValueError("DeepStack order differs from source config")
    graph_audit = json.loads((args.build / "quantization/vision_full/encoding.json").read_text())
    if graph_audit["calibrated_sha256"] != frozen or len(graph_audit["layers"]) != cfg.depth * 4 + (len(taps) + 1) * 2:
        raise ValueError("Fused quantization coverage or frozen identity differs")
    if any(v["dtype"] != "UINT4" or not v["lut_mode"] or not v["decoded_matches_reference"] for v in graph_audit["layers"]):
        raise ValueError("Not all visual block/merger matrices are verified W4 LUT")
    # Compare the actual encoded W4 tensors, including all merger parameters.
    candidate = torch.load(args.build / "quantization/vision_full/quant_output/fake_quant_weight.pth", map_location="cpu", weights_only=True)
    pairs = {}
    for g in source["graphs"]:
        if g["component"] not in ("vision_blocks", "vision_post"):
            continue
        q = g["quantization"]
        weights = torch.load(args.source_build / Path(q["parameters"]).parent / "fake_quant_weight.pth", map_location="cpu", weights_only=True)
        if g["component"] == "vision_post":
            pairs.update(("post." + k, v) for k, v in weights.items())
        else:
            for k, v in weights.items():
                parts = k.split(".")
                parts[1] = str(int(parts[1]) + g["layer_start"])
                pairs["blocks." + ".".join(parts)] = v
    # Plain patch/position weights are also present in the fused SDK state.
    # They were previously exported in the separate unquantized vision_pre.
    front_entry = old.entries["vision_pre"]
    front_path = args.source_build / front_entry["onnx"]
    with tempfile.TemporaryDirectory(prefix="fused_front_audit_") as directory:
        # ONNX's external-data reader rejects hardlinks used by immutable build
        # reuse. Copy this small component rather than weakening that check.
        for name in [front_path.name, *front_entry["external_data_sha256"]]:
            shutil.copyfile(front_path.parent / name, Path(directory) / name)
        front = onnx.load(str(Path(directory) / front_path.name))
    initializers = {v.name: onnx.numpy_helper.to_array(v) for v in front.graph.initializer}
    for key in ("pre.weight", "pre.bias", "pre.position.weight"):
        value = candidate[key].numpy()
        if not any(np.array_equal(value, v) or (value.ndim == 2 and np.array_equal(value.T, v)) for v in initializers.values()):
            raise ValueError("Floating visual front weight changed: " + key)
    if set(candidate) != set(pairs) | {"pre.weight", "pre.bias", "pre.position.weight"} or any(not torch.equal(candidate[k], pairs[k]) for k in pairs):
        raise ValueError("Fused encoded W4 tensors changed from split graphs")
    del candidate, pairs
    # Unchanged language graphs must retain their exact graph and weight files.
    unchanged = []
    for name, g in old.entries.items():
        if name.startswith("vision_"): continue
        n = new.entries[name]
        for field in ("onnx_sha256", "omc_sha256", "external_weights", "reference", "quantization"):
            if g.get(field) != n.get(field):
                raise ValueError("Unselected language graph changed: " + name + ": " + field)
        unchanged.append(name)
    grid = torch.tensor([[1, args.height // cfg.patch_size, args.width // cfg.patch_size]])
    tokens = int(grid.prod())
    if tokens != target["profile"]["vision_tokens"]: raise ValueError("Wrong image grid")
    with torch.device("meta"):
        visual = Qwen3VLVisionModel(cfg)
    indices, weights = get_vision_interpolation_indices_and_weights(grid,
        num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners, spatial_merge_size=cfg.spatial_merge_size)
    positions = get_vision_position_ids(grid, cfg.spatial_merge_size)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding
    rotary = Qwen3VLVisionRotaryEmbedding(cfg.hidden_size // cfg.num_heads // 2)
    angles = rotary(positions).reshape(tokens, -1)
    angles = torch.cat((angles, angles), -1)
    cos, sin = angles.cos()[None, None].numpy(), angles.sin()[None, None].numpy()
    mask = np.zeros((1, 1, tokens, tokens), dtype=np.float32)
    processor = Qwen2VLImageProcessorPil.from_pretrained(args.model, local_files_only=True)
    calibration_hashes = {sha256(Path(row["image"])) for row in source["calibration"]["samples"] if "image" in row}
    cases = []
    for path in args.image:
        image_hash = sha256(path)
        if image_hash in calibration_hashes: raise ValueError("Validation image was used for calibration: " + str(path))
        with Image.open(path) as image:
            pixels = processor(images=[image.convert("RGB").resize((args.width, args.height), Image.Resampling.BICUBIC)],
                               return_tensors="np")["pixel_values"]
        front = [pixels, indices.int().numpy(), weights.numpy()]
        hidden = old.run("vision_pre", front)[0]
        deep = []
        chunks = sorted((g for g in source["graphs"] if g["component"] == "vision_blocks"), key=lambda g: g["layer_start"])
        for chunk in chunks:
            out = old.run(chunk["name"], [hidden, cos, sin, mask])
            hidden = out[0]; deep.extend(out[1:])
        split = old.run("vision_post", [hidden, *deep])
        fused = new.run("vision_full", [*front, cos, sin, mask])
        results = {"image_embeddings": metrics(split[0], fused[0])}
        results.update(("deepstack_" + str(layer), metrics(split[1][i], fused[1][i])) for i, layer in enumerate(taps))
        passed = all(v["relative_l2"] <= args.relative_l2 and v["cosine"] >= .9999 for v in results.values())
        case = dict(image=str(path), sha256=image_hash, held_out=True, outputs=results, passed=passed)
        cases.append(case); print(json.dumps(case, ensure_ascii=False), flush=True)
    report = dict(device_tested=False, omc_executed=False, scope="split W4 CPU ONNX vs fused W4 CPU ONNX",
        absolute_quantization_quality_reassessed=False, frozen_checkpoint_sha256=frozen,
        source_manifest_sha256=sha256(args.source_build / "manifest.json"),
        candidate_manifest_sha256=sha256(args.build / "manifest.json"),
        candidate_manifest_stage="compiled" if entry.get("omc") else "exported_before_external_compile",
        candidate_reference_onnx_sha256=entry["reference"]["onnx_sha256"],
        candidate_native_onnx_sha256=entry["onnx_sha256"],
        w4_encoded_tensors_unchanged=True, w4_visual_matrices=len(graph_audit["layers"]),
        language_graphs_unchanged=unchanged, visual_deepstack_layers=taps, llm_injection_layers=list(range(len(taps))),
        relative_l2_threshold=args.relative_l2, cosine_threshold=.9999, cases=cases,
        passed=bool(cases) and all(c["passed"] for c in cases))
    save_json(args.build / "fused_vision_validation.json", report)
    if not report["passed"]: raise ValueError("Fused visual equivalence failed; see saved report")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-build", "build", "model"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--image", type=Path, action="append", required=True)
    p.add_argument("--height", type=int, default=608); p.add_argument("--width", type=int, default=256)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--relative-l2", type=float, default=.01)
    args = p.parse_args(); torch.set_num_threads(args.threads); validate(args)
