"""CPU gates for newly built hybrid ViT: frozen math, ONNX fidelity, and held-out quality.

Native UINT4/LUT graphs are never executed by ORT. The separate dequantized
reference is used; these checks make no claim about OMC/device correctness.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from experiment import ONNXGraphs, metrics, save_json, sha256
from external_weights import arguments
from validation import static_image
from w4a16 import W4Experiment, image_inputs


@torch.no_grad()
def verify(options):
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
    manifest = json.loads((options.build / "manifest.json").read_text(encoding="utf-8"))
    # A failed/restarted verification must never leave a previous passing report usable.
    save_json(options.build / "hybrid_vision_validation.json",
              {"passed": False, "status": "verification_started", "device_tested": False})
    profile = manifest["profile"]
    calibration = manifest["calibration"]
    if sha256(options.image) in {row.get("image_sha256") for row in calibration["samples"]}:
        raise ValueError("Held-out image was used during calibration")
    args = arguments(manifest, options.build, options.model, options.ddk,
                     layers=profile["layers_per_chunk"], components="vision", fuse_vision=True,
                     activation_range_multiplier=profile.get("activation_range_multiplier", 1),
                     output_range_multiplier=profile.get("output_range_multiplier", 1))
    args.threads, args.image, args.alignment_rtol = options.threads, options.image, options.alignment_rtol
    torch.set_num_threads(options.threads)
    experiment = W4Experiment(args)
    quantized = experiment.load_model()
    # Only visual features are tested here. Release unneeded 2B language copies.
    quantized.model.language_model = None
    quantized.lm_head = None
    experiment.sdk_model = None
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    inputs = image_inputs(quantized, tokenizer, options.image, args)
    image, features, fidelity = static_image(quantized, inputs, args, ONNXGraphs(options.build, options.threads))
    original = Qwen3VLForConditionalGeneration.from_pretrained(
        options.model, dtype=torch.float32, attn_implementation="eager", local_files_only=True).eval()
    original.model.language_model = None
    original.lm_head = None
    gc.collect()
    reference, deepstack, _ = static_image(original, inputs, args)
    records = {"image_embeddings": metrics(reference.numpy(), image.numpy())}
    for i, (a, b) in enumerate(zip(deepstack, features)):
        records["deepstack_" + str(i)] = metrics(a.numpy(), b.numpy())
    graph = next(g for g in manifest["graphs"] if g["name"] == "vision_full")
    encoding = json.loads((options.build / Path(graph["quantization"]["parameters"]).parent.parent / "encoding.json").read_text())
    encoded = encoding.get("layers", [])
    coverage = (len(encoded) == 104 and all(layer.get("dtype") == "UINT4" and layer.get("lut_mode")
                and layer.get("decoded_matches_reference") for layer in encoded))
    passed = (coverage and len(features) == 3 and all(np.isfinite(v) for r in records.values() for v in r.values())
              and all(r["relative_l2"] <= options.quality_rtol and r["cosine"] >= options.min_cosine for r in records.values()))
    report = {"passed": bool(passed), "device_tested": False, "omc_executed": False,
              "image": str(options.image), "image_sha256": sha256(options.image),
              "checkpoint_sha256": calibration["checkpoint_sha256"],
              "reference_onnx_sha256": graph["reference"]["onnx_sha256"],
              "reference_external_data_sha256": graph["reference"]["external_data_sha256"],
              "native_onnx_sha256": graph["onnx_sha256"],
              "native_external_data_sha256": graph["external_data_sha256"],
              "parameters_sha256": graph["quantization"]["parameters_sha256"],
              "w4_lut_coverage_passed": bool(coverage), "encoded_matrices": len(encoded),
              "onnx_vs_frozen_quantized_hf": fidelity, "fp32_vs_quantized_vision": records,
              "thresholds": {"alignment_rtol": options.alignment_rtol, "quality_rtol": options.quality_rtol,
                             "min_cosine": options.min_cosine},
              "scope": "One held-out image, image embeddings and each DeepStack; not an end-to-end quality benchmark"}
    save_json(options.build / "hybrid_vision_validation.json", report)
    if not passed:
        raise ValueError("Held-out vision quality/encoding gate failed; inspect hybrid_vision_validation.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "model", "ddk", "image"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--alignment-rtol", type=float, default=.01)
    parser.add_argument("--quality-rtol", type=float, default=.3)
    parser.add_argument("--min-cosine", type=float, default=.95)
    options = parser.parse_args()
    for key in ("build", "model", "ddk", "image"):
        setattr(options, key, getattr(options, key).resolve())
    verify(options)


if __name__ == "__main__":
    main()
