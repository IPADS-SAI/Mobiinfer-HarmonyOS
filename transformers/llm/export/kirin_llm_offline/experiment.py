#!/usr/bin/env python3
"""Host-only feasibility experiment: Qwen3-VL -> static ONNX -> Kirin OMC.

No deployment, phone access, publication, or changes to the existing MNN runtime.
FP32 ONNX is the mathematical reference. Optional W4A16 exports paired SDK
LUT indices and compression parameters separately from the CPU reference.
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch

from static_kv import (KVCache, KV_DTYPES, StaticDecoderChunk, StaticHead, StaticSession, StaticVisionChunk,
                       StaticVisionPost, StaticVisionPre, StaticVisionFull, attention_mask, cache_memory)

EXPORT = Path(__file__).resolve().parent.parent
VISUAL_ROUTE = EXPORT / "plugin_quant_visual_matmul_route_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + "_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def metrics(reference, actual):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Mismatched/non-finite reference result")
    a, b = reference.ravel(), actual.ravel()
    return {"relative_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-20)),
            "max_abs": float(np.max(np.abs(a - b))),
            "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-20))}


def torch_callback(module):
    def run(_decode, inputs):
        with torch.no_grad():
            return [value.numpy() for value in module(*[torch.from_numpy(v) for v in inputs])]
    return run


def chunk_ranges(total, width):
    return [(start, min(start + width, total)) for start in range(0, total, width)]


class ONNXGraphs:
    """CPU reference execution of the exported static graph interface."""
    def __init__(self, output, threads=4):
        self.output = Path(output)
        self.manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.entries = {value["name"]: value for value in self.manifest["graphs"]}
        self.sessions = {}
        self.threads = threads

    def run(self, name, inputs):
        if name not in self.sessions:
            import onnxruntime as ort
            entry = self.entries[name]
            if entry.get("quantization") and "reference" not in entry:
                raise ValueError("Native LUT graph requires a separate CPU reference: " + name)
            entry = entry.get("reference", entry)
            path = self.output / entry["onnx"]
            if sha256(path) != entry["onnx_sha256"] or any(
                sha256(path.parent / key) != value for key, value in entry["external_data_sha256"].items()
            ):
                raise ValueError("Exported graph changed: " + name)
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            self.sessions[name] = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        session = self.sessions[name]
        if len(inputs) != len(session.get_inputs()):
            raise ValueError("Wrong input count for " + name)
        return session.run(None, {value.name: array for value, array in zip(session.get_inputs(), inputs)})

    def decoder_callback(self, index):
        def run(decode, inputs):
            return self.run(f"llm_{'decode' if decode else 'prefill'}_{index}", inputs)
        return run

    def embedding(self, ids, decode=False):
        width = 1 if decode else self.manifest["profile"]["prefill_tokens"]
        values = []
        for offset in range(0, ids.shape[1], width):
            real = min(width, ids.shape[1] - offset)
            padded = np.zeros((1, width), dtype=np.int32)
            padded[:, :real] = ids[:, offset:offset + real]
            values.append(self.run("embedding_" + ("decode" if decode else "prefill"), [padded])[0][:, :real])
        return np.concatenate(values, axis=1)


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        source = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
        if source.get("model_type") != "qwen3_vl":
            raise ValueError("Only the existing Qwen3-VL/mobi architecture is supported")
        self.source_config = source
        self.text_config = None
        self.model = None
        self.manifest_path = self.output / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {"graphs": []}
        if self.manifest_path.exists() and self.manifest.get("format") != "kirin_static_vlm_experiment_v1":
            raise ValueError("Output manifest belongs to another format")
        profile = {"source_model": str(args.model), "source_config_sha256": sha256(args.model / "config.json"),
                   "platform": args.platform, "context": args.context, "prefill_tokens": args.prefill_tokens,
                   "layers_per_chunk": args.layers_per_chunk, "vision_tokens": args.vision_tokens,
                   "vision_layers_per_chunk": args.vision_layers_per_chunk}
        # Static graph arithmetic is part of the build identity. Do not reuse
        # old native graphs after changing the normalization/attention export.
        profile["static_math"] = "scaled_rmsnorm_prescaled_qk_v1"
        if args.fuse_vision:
            profile["fused_vision"] = True
        quantized = args.precision == "w4a16"
        if quantized:
            profile.update(precision=args.precision, group_size=args.group_size, head_bit=args.head_bit)
            if args.activation_range_multiplier != 1:
                profile["activation_range_multiplier"] = args.activation_range_multiplier
            if args.output_range_multiplier != 1:
                profile["output_range_multiplier"] = args.output_range_multiplier
            if args.weight_method != "rtn":
                profile.update(weight_method=args.weight_method, gptq_damping=args.gptq_damping,
                               gptq_block_size=args.gptq_block_size)
        if args.kv_dtype != "fp32":
            profile["kv_dtype"] = args.kv_dtype
        if args.kv_dtype == "int8":
            profile["kv_group_size"] = args.kv_group_size
            profile.update(kv_residual_tokens=args.kv_residual_tokens, kv_sink_tokens=args.kv_sink_tokens)
        if self.manifest.get("profile", profile) != profile:
            raise ValueError("Build profile changed; use a new output directory")
        if args.output.exists() and any(args.output.iterdir()) and not self.manifest_path.exists():
            raise ValueError("Output must be empty or belong to this experiment")
        self.manifest.update(format="kirin_static_vlm_experiment_v1", profile=profile, experimental=True,
                             onnx_precision="float32", compiler_weight_type="FP16", quantized=quantized,
                             device_tested=False, app_integrated=False)
        if quantized:
            self.manifest["quantization_scheme"] = {
                "linear_weight_bits": 4, "linear_activation_bits": 16, "activation_dtype": "INT16",
                "group_size": args.group_size, "head_bits": args.head_bit, "encoding": "SDK UINT4/LUT",
                "embedding": "floating point", "kv_quantized": args.kv_dtype == "int8",
                "cpu_execution": "separate dequantized reference graphs"}
        cfg = source["text_config"]
        fp16_cache = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * args.context * cfg["head_dim"] * 2
        self.manifest["cache"] = {"layout": "[1, kv_heads, capacity, head_dim] per layer and K/V",
                                  "capacity": args.context, "fp16_bytes": fp16_cache,
                                  "reference_float32_bytes": fp16_cache * 2,
                                  "update": "host appends only real delta KV; no sliding-window eviction"}
        self.manifest["cache"].update(cache_memory(cfg["num_hidden_layers"], cfg["num_key_value_heads"],
                                                  args.context, cfg["head_dim"], args.kv_dtype, args.kv_group_size,
                                                  args.kv_residual_tokens, args.kv_sink_tokens))
        if args.kv_dtype == "int8":
            self.manifest["cache"].update(
                quantization="symmetric per-token per-KV-head per-channel-group max-abs; q=clip(round(x/scale),-127,127)",
                group_size=min(args.kv_group_size, cfg["head_dim"]), scale_dtype="float32",
                scale_layout="[1, kv_heads, capacity, head_dim/group_size] per layer and K/V",
                zero_point=0, zero_vector_scale=1, key_quantization="after RoPE",
                residual_tokens=min(args.kv_residual_tokens, args.context), sink_tokens=min(args.kv_sink_tokens, args.context),
                protected_dtype="float16", protected_layout="prefix then recent ring; duplicated INT8 rows are masked",
                quantize_location="host encodes only new real delta KV; current block attention stays floating point",
                dequantize_location="inside decoder graph: Cast(INT8->float) * scale",
                delta_output_dtype="float32", full_precision_shadow_cache=False)
        self.output.mkdir(parents=True, exist_ok=True)
        self.save()

    def save(self):
        save_json(self.manifest_path, self.manifest)

    def load_model(self):
        if self.model is None:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.args.model, dtype=torch.float32, attn_implementation="eager", local_files_only=True).eval()
            self.text_config = self.model.config.text_config
        return self.model

    def reuse(self):
        """Reuse frozen weights and non-LLM graphs when only KV storage changes."""
        source = self.args.reuse_build
        if source is None:
            raise ValueError("reuse requires --reuse-build")
        previous = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        def weight_profile(profile):
            return {key: value for key, value in profile.items()
                    if key not in {"kv_dtype", "kv_group_size", "kv_residual_tokens", "kv_sink_tokens"}}
        if (previous.get("format") != self.manifest["format"]
                or weight_profile(previous["profile"]) != weight_profile(self.manifest["profile"])):
            raise ValueError("Reuse requires identical source, shapes, platform and weight quantization")
        if self.manifest["graphs"] or self.manifest.get("calibration") or (self.output / "quantization").exists():
            raise ValueError("Reuse requires a new build without calibration or graphs")
        record = previous.get("calibration")
        if record:
            for key in ("config", "checkpoint"):
                if sha256(source / record[key]) != record[key + "_sha256"]:
                    raise ValueError("Source calibration changed: " + key)
            # Independent copies: later calibration must not overwrite the
            # source checkpoint through a hard link or symlink.
            shutil.copytree(source / "quantization", self.output / "quantization")
            self.manifest["calibration"] = record
        for entry in previous["graphs"]:
            if entry["component"] == "llm":
                continue
            for reference in [entry, *([entry["reference"]] if "reference" in entry else [])]:
                path = Path(reference["onnx"])
                files = {path: reference["onnx_sha256"]}
                files.update({path.parent / key: value for key, value in reference["external_data_sha256"].items()})
                for relative, digest in files.items():
                    (source / relative).resolve().relative_to(source)
                    if sha256(source / relative) != digest:
                        raise ValueError("Source graph changed: " + str(relative))
                    (self.output / relative).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / relative, self.output / relative)
            if entry.get("omc"):
                artifact = source / entry["omc"]
                artifact.resolve().relative_to(source)
                if sha256(artifact) != entry["omc_sha256"]:
                    raise ValueError("Source compiled artifact changed: " + entry["name"])
                destination = self.output / entry["omc"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(artifact, destination)
                # External OMCs live in a subdirectory and still require their
                # paired weights. Reusing only the graph produces an unloadable
                # package, even though its graph checksum is unchanged.
                for weight in entry.get("external_weights", []):
                    original = source / weight["file"]
                    original.resolve().relative_to(source)
                    if sha256(original) != weight["sha256"] or original.stat().st_size != weight["bytes"]:
                        raise ValueError("Source external weight changed: " + weight["file"])
                    copied = self.output / weight["file"]
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, copied)
                log = (Path(entry["onnx"]).parent / "compile_external.log" if entry.get("external_weights")
                       else Path(entry["omc"]).parent / "compile.log")
                shutil.copy2(source / log, self.output / log)
            self.manifest["graphs"].append(entry)
        self.manifest["reused_build"] = {"source": str(source), "manifest_sha256": sha256(source / "manifest.json"),
                                          "llm_graphs_reused": False, "calibration_unchanged": bool(record)}
        self.save()

    def selected_ranges(self):
        ranges = chunk_ranges(self.source_config["text_config"]["num_hidden_layers"], self.args.layers_per_chunk)
        if self.args.chunks is None:
            return list(enumerate(ranges))
        selected = [int(value) for value in self.args.chunks.split(",")]
        if len(set(selected)) != len(selected) or any(i < 0 or i >= len(ranges) for i in selected):
            raise ValueError("Invalid --chunks indexes")
        return [(i, ranges[i]) for i in selected]

    def export_graph(self, name, module, inputs, names, outputs, **metadata):
        print("export " + name, flush=True)
        directory = self.output / "graphs" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.onnx"
        # External-data files are immutable per export attempt; using a fresh
        # staging directory also avoids append-to-existing ONNX weight files.
        with tempfile.TemporaryDirectory(prefix="export_", dir=directory) as temporary:
            temporary_path = Path(temporary) / "model.onnx"
            with torch.no_grad():
                torch.onnx.export(module.eval(), tuple(inputs), str(temporary_path), opset_version=12,
                                  dynamo=False, do_constant_folding=True, input_names=names, output_names=outputs)
            original_ir = onnx.load(str(temporary_path), load_external_data=False).ir_version
            sys.path.insert(0, str(VISUAL_ROUTE))
            from visual_plugin_quant_matmul_route import process_onnx_for_omg
            temporary_path = Path(process_onnx_for_omg(str(temporary_path), fp16=False))
            metadata_model = onnx.load(str(temporary_path), load_external_data=False)
            # The shared simplifier rebuilds ModelProto with the installed
            # ONNX package's latest IR. Preserve the actual exporter's IR so
            # current ORT can evaluate this opset-12 graph as well as OMG.
            metadata_model.ir_version = original_ir
            onnx.save(metadata_model, str(temporary_path))
            referenced = {field.value for tensor in metadata_model.graph.initializer
                          for field in tensor.external_data if field.key == "location"}
            for name_to_copy in {temporary_path.name, *referenced}:
                shutil.copy2(temporary_path.parent / name_to_copy, directory / name_to_copy)
        graph = onnx.load(str(path), load_external_data=False).graph
        if [value.name for value in graph.input] != names:
            raise ValueError("Export dropped or reordered an input: " + name)
        external = {entry.value for value in graph.initializer for entry in value.external_data if entry.key == "location"}
        initializers = {p: sha256(directory / p) for p in external}
        # A pure-NPU compiler result does not prove export math. Run the exact
        # ONNX on CPU before compiling, with nonzero history and causal masks.
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.args.threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            reference = module(*inputs)
        reference = reference if isinstance(reference, tuple) else (reference,)
        actual = session.run(None, {key: value.numpy() for key, value in zip(names, inputs)})
        if len(reference) != len(actual) or len(actual) != len(outputs):
            raise ValueError("Export changed output count: " + name)
        comparisons = [metrics(a.numpy(), b) for a, b in zip(reference, actual)]
        if any(value["relative_l2"] > self.args.alignment_rtol for value in comparisons):
            raise ValueError("ONNX disagrees with Torch: " + name + " " + str(comparisons))
        aliases = {node.output[0]: node.input[0] for node in graph.node if node.op_type == "Identity"}
        def root(value):
            seen = set()
            while value in aliases and value not in seen:
                seen.add(value)
                value = aliases[value]
            return value
        output_aliases = {second: first for i, first in enumerate(outputs) for second in outputs[i + 1:]
                          if root(first) == root(second)}
        entry = {"name": name, "onnx": str(path.relative_to(self.output)), "onnx_sha256": sha256(path),
                 "external_data_sha256": initializers, "inputs": [
                     {"name": key, "shape": list(value.shape), "dtype": str(value.numpy().dtype)}
                     for key, value in zip(names, inputs)],
                 "outputs": outputs, "output_info": [
                     {"name": key, "shape": list(value.shape), "dtype": str(value.dtype)}
                     for key, value in zip(outputs, actual)],
                 "output_aliases": output_aliases, "torch_ort": comparisons, **metadata}
        self.manifest["graphs"] = [value for value in self.manifest["graphs"] if value["name"] != name] + [entry]
        self.save()
        del session
        gc.collect()

    def export(self):
        model = self.load_model()
        language = model.model.language_model
        cfg = self.text_config
        for mode, tokens in (("prefill", self.args.prefill_tokens), ("decode", 1)):
            if mode not in self.args.modes:
                continue
            if "llm" in self.args.components:
                for index, (start, end) in self.selected_ranges():
                    count = end - start
                    module = StaticDecoderChunk(language.layers[start:end], cfg, self.args.kv_dtype, self.args.kv_group_size,
                                                self.args.kv_residual_tokens, self.args.kv_sink_tokens)
                    residual = min(self.args.kv_residual_tokens, self.args.context) if self.args.kv_dtype == "int8" else 0
                    sinks = min(self.args.kv_sink_tokens, self.args.context) if self.args.kv_dtype == "int8" else 0
                    inputs = [torch.randn(1, tokens, cfg.hidden_size) * 0.02,
                              torch.ones(1, 1, tokens, cfg.head_dim), torch.zeros(1, 1, tokens, cfg.head_dim),
                              torch.from_numpy(attention_mask(self.args.context, min(7, self.args.context - tokens),
                                                              tokens, tokens, residual_tokens=residual, sink_tokens=sinks)),
                              torch.randn(count, 1, tokens, cfg.hidden_size) * 0.01]
                    sample_cache = KVCache(count, cfg.num_key_value_heads, self.args.context,
                                           cfg.head_dim, KV_DTYPES[self.args.kv_dtype], self.args.kv_group_size,
                                           self.args.kv_residual_tokens, self.args.kv_sink_tokens)
                    history = [np.random.default_rng(i).standard_normal(
                        (1, cfg.num_key_value_heads, self.args.context, cfg.head_dim)).astype(np.float32) * 0.02
                        for i in range(2 * count)]
                    sample_cache.append_chunk(0, history, self.args.context)
                    inputs += [torch.from_numpy(value) for value in sample_cache.inputs(0, count)]
                    names = ["hidden_states_in", "rope_cos", "rope_sin", "attention_mask", "deepstack"]
                    for i in range(start, end):
                        names.extend((f"past_key_{i}", f"past_value_{i}"))
                        if self.args.kv_dtype == "int8":
                            names.extend((f"past_key_scale_{i}", f"past_value_scale_{i}"))
                            if sinks:
                                names.extend((f"sink_key_{i}", f"sink_value_{i}"))
                            if residual:
                                names.extend((f"recent_key_{i}", f"recent_value_{i}"))
                    outputs = ["hidden_states"] + [value for i in range(start, end)
                                                   for value in (f"key_delta_{i}", f"value_delta_{i}")]
                    self.export_graph(f"llm_{mode}_{index}", module, inputs, names, outputs,
                                      component="llm", mode=mode, layer_start=start, layer_end=end)
            if "embedding" in self.args.components:
                self.export_graph("embedding_" + mode, language.embed_tokens,
                                  [torch.zeros(1, tokens, dtype=torch.int32)], ["token_ids"], ["embeddings"],
                                  component="embedding", mode=mode)
        if "head" in self.args.components:
            self.export_graph("lm_head", StaticHead(language.norm, model.lm_head),
                              [torch.randn(1, 1, cfg.hidden_size)], ["hidden_states"], ["logits"], component="head")
        if "vision" in self.args.components:
            self.export_vision(model.model.visual)

    def export_vision(self, visual):
        cfg, tokens = visual.config, self.args.vision_tokens
        patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2
        if self.args.fuse_vision:
            self.export_graph("vision_full", StaticVisionFull(visual),
                [torch.randn(tokens, patch_dim), torch.zeros(tokens, 4, dtype=torch.int32),
                 torch.full((tokens, 4), .25),
                 torch.ones(1, 1, tokens, cfg.hidden_size // cfg.num_heads),
                 torch.zeros(1, 1, tokens, cfg.hidden_size // cfg.num_heads), torch.zeros(1, 1, tokens, tokens)],
                ["patches", "position_indices", "position_weights", "rope_cos", "rope_sin", "attention_mask"],
                ["image_embeddings", "deepstack_features"], component="vision_full",
                layer_start=0, layer_end=len(visual.blocks), deepstack_layers=list(cfg.deepstack_visual_indexes))
            return
        self.export_graph("vision_pre", StaticVisionPre(visual),
                          [torch.randn(tokens, patch_dim), torch.zeros(tokens, 4, dtype=torch.int32),
                           torch.full((tokens, 4), 0.25)], ["patches", "position_indices", "position_weights"],
                          ["hidden_states"], component="vision_pre")
        sample = torch.randn(1, tokens, cfg.hidden_size)
        rotary = [torch.ones(1, 1, tokens, cfg.hidden_size // cfg.num_heads),
                  torch.zeros(1, 1, tokens, cfg.hidden_size // cfg.num_heads)]
        for index, (start, end) in enumerate(chunk_ranges(len(visual.blocks), self.args.vision_layers_per_chunk)):
            local = [i for i in cfg.deepstack_visual_indexes if start <= i < end]
            outputs = ["hidden_states"] + [f"deepstack_{i}" for i in local]
            self.export_graph(f"vision_blocks_{index}", StaticVisionChunk(visual.blocks[start:end], start,
                              cfg.deepstack_visual_indexes), [sample, *rotary, torch.zeros(1, 1, tokens, tokens)],
                              ["hidden_states_in", "rope_cos", "rope_sin", "attention_mask"], outputs,
                              component="vision_blocks", layer_start=start, layer_end=end, deepstack_layers=local)
        names = ["hidden_states"] + [f"deepstack_{i}" for i in cfg.deepstack_visual_indexes]
        self.export_graph("vision_post", StaticVisionPost(visual), [sample] * len(names), names,
                          ["image_embeddings", "deepstack_features"], component="vision_post")

    def compile(self):
        if self.args.ddk is None:
            raise ValueError("--ddk is required for compile")
        sys.path.insert(0, str(VISUAL_ROUTE))
        from build_kirin_offline import validate_compile_log
        if not (self.args.ddk / "tools/platform" / self.args.platform).is_dir():
            raise ValueError("Missing platform plugin")
        env = dict(os.environ, PLATFORM=self.args.platform, TARGET_MODEL_TYPE="omc", USE_COMPRESS_CONF="false",
                   LOAD_ASCENDC_ENV="auto", SAVE_WEIGHTS_AS_EXTERNAL_DATA="false", PYTHONDONTWRITEBYTECODE="1",
                   OMG_TOOL=str(self.args.ddk / "tools/tools_omg/omg"),
                   OMG_MASTER_DIR=str(self.args.ddk / "tools/tools_omg/master"),
                   ASCENDC_ENV_SCRIPT=str(self.args.ddk / "tools/tools_ascendc/set_ascendc_env.sh"))
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        graphs = [entry for entry in self.manifest["graphs"] if self.selected(entry)]
        if not graphs:
            raise ValueError("No exported graphs match the selection")
        self.args.compiler_work_dir.mkdir(parents=True, exist_ok=True)
        for entry in graphs:
            name, path = entry["name"], self.output / entry["onnx"]
            if sha256(path) != entry["onnx_sha256"] or any(
                sha256(path.parent / location) != digest for location, digest in entry["external_data_sha256"].items()
            ):
                raise ValueError("Exported graph changed: " + name)
            print("compile " + name, flush=True)
            log = path.parent / "compile.log"
            quantization = entry.get("quantization")
            graph_env = dict(env, USE_COMPRESS_CONF="true" if quantization else "false")
            # Keep SDK-restricted paths independent of the user's output path.
            with tempfile.TemporaryDirectory(prefix="kirin_llm_", dir=self.args.compiler_work_dir) as directory:
                root = Path(directory)
                shutil.copy2(path, root / "model.onnx")
                for location in entry["external_data_sha256"]:
                    target = root / location
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path.parent / location, target)
                if quantization:
                    parameters = self.output / quantization["parameters"]
                    if sha256(parameters) != quantization["parameters_sha256"]:
                        raise ValueError("Quantization parameters changed: " + name)
                    (root / "quant_output").mkdir()
                    shutil.copy2(parameters, root / "quant_output/quant_params_file")
                shape = ";".join(value["name"] + ":" + ",".join(map(str, value["shape"])) for value in entry["inputs"])
                save_json(root / "route_config.json", {"model": str(root / "model.onnx"), "input_shape": shape})
                with log.open("w") as stream:
                    result = subprocess.run(["bash", str(VISUAL_ROUTE / "run_visual_plugin_matmul_omc.sh"),
                                             str(root), "fp16"], cwd=root, env=graph_env,
                                            stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"OMG failed for {name}; inspect {log}")
                validate_compile_log(log.read_text(errors="replace"))
                artifact = root / "omc_output/visual_plugin_matmul_quantized.omc"
                if not artifact.is_file() or not artifact.stat().st_size:
                    raise RuntimeError("Missing compiled OMC: " + name)
                target = path.parent / "model.omc"
                shutil.copy2(artifact, target)
            entry.update(omc=str(target.relative_to(self.output)), omc_sha256=sha256(target),
                         omc_bytes=target.stat().st_size, pure_npu=True,
                         compiled_onnx_sha256=entry["onnx_sha256"])
            if quantization:
                entry["compiled_parameters_sha256"] = quantization["parameters_sha256"]
            self.save()

    def selected(self, entry):
        if entry["component"].startswith("vision"):
            return "vision" in self.args.components
        if entry["component"] not in self.args.components:
            return False
        if entry.get("mode") is not None and entry["mode"] not in self.args.modes:
            return False
        return entry["component"] != "llm" or any(start == entry["layer_start"]
                                                for _, (start, _) in self.selected_ranges())

    def validate(self):
        from validation import validate_model
        previous = self.manifest.get("host_validation", {})
        if "passed" in previous:
            previous = {"torch": previous}
        previous[self.args.validation_backend] = validate_model(self.load_model(), self.args)
        self.manifest["host_validation"] = previous
        self.save()

    def kv_validate(self):
        from kv_validation import validate_cache
        result = validate_cache(self.load_model(), self.args)
        self.manifest.setdefault("kv_validation", {})[self.args.validation_backend] = result
        self.save()
        if not result["passed"]:
            raise ValueError("KV validation exceeded the unchanged acceptance criteria; see " + result["report"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ddk", type=Path)
    parser.add_argument("--platform", default="kirin9030")
    parser.add_argument("--precision", choices=("fp16", "w4a16"), default="fp16")
    parser.add_argument("--kv-dtype", choices=tuple(KV_DTYPES), default="fp32",
                        help="Persistent KV storage and graph input format; INT8 uses dynamic per-token/head scales")
    parser.add_argument("--kv-group-size", type=int, default=32, help="INT8 KV channels per dynamic scale; independent of W4 group size")
    parser.add_argument("--kv-residual-tokens", type=int, default=128, help="INT8 KV: recent tokens kept in FP16; 0 disables")
    parser.add_argument("--kv-sink-tokens", type=int, default=4, help="INT8 KV: initial tokens kept in FP16; 0 disables")
    parser.add_argument("--kv-alignment-rtol", type=float, default=0.05,
                        help="Maximum relative L2 against the same static model with FP32 KV")
    parser.add_argument("--kv-long-tokens", type=int, default=512, help="Long-prefix KV validation token count")
    parser.add_argument("--kv-quality-tokens", type=int, default=0,
                        help="Optional free-running generation length for KV validation; 0 skips it")
    parser.add_argument("--reuse-build", type=Path, help="Reuse weights and non-LLM graphs from an otherwise identical build")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--activation-range-multiplier", type=float, default=1,
                        help="Expand frozen INT16 input/output ranges without changing W4 weight scales")
    parser.add_argument("--output-range-multiplier", type=float, default=1,
                        help="Additional guard on per-channel output ranges; input ranges stay unchanged")
    parser.add_argument("--head-bit", type=int, choices=(4, 16), default=4)
    parser.add_argument("--weight-method", choices=("rtn", "gptq"), default="rtn")
    parser.add_argument("--quant-device", default="cuda:0", help="GPU for GPTQ and activation calibration")
    parser.add_argument("--gptq-damping", type=float, default=0.01)
    parser.add_argument("--gptq-block-size", type=int, default=128)
    parser.add_argument("--calibration-images", type=Path)
    parser.add_argument("--calibration-texts", type=Path, help="Optional JSON list of {text: prompt} rows")
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--calibration-decode-steps", type=int, default=4)
    parser.add_argument("--quality-tokens", type=int, default=32)
    parser.add_argument("--quality-images", nargs="*", type=Path, default=[])
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--layers-per-chunk", type=int, default=4)
    parser.add_argument("--vision-tokens", type=int, default=608)
    parser.add_argument("--vision-layers-per-chunk", type=int, default=4)
    parser.add_argument("--fuse-vision", action="store_true", help="Fuse visual pre, all blocks and post into one graph")
    parser.add_argument("--components", default="llm,head,embedding,vision")
    parser.add_argument("--modes", default="prefill,decode")
    parser.add_argument("--chunks", help="Optional decoder chunk indexes, e.g. 0,6")
    parser.add_argument("--steps", help="Comma-separated stages; precision-dependent default")
    parser.add_argument("--compiler-work-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--image", type=Path, help="Optional real image for VLM validation")
    parser.add_argument("--image-height", type=int, default=608)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--prompt", default="请描述这张图片的内容。")
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--validation-backend", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--alignment-rtol", type=float,
                        help="Reference numeric tolerance; floating baseline=2e-4, W4A16 or compressed KV=0.05")
    parser.add_argument("--strict-greedy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not np.isfinite(args.activation_range_multiplier) or not 1 <= args.activation_range_multiplier <= 32:
        parser.error("--activation-range-multiplier must be in [1,32]")
    if not np.isfinite(args.output_range_multiplier) or not 1 <= args.output_range_multiplier <= 32:
        parser.error("--output-range-multiplier must be in [1,32]")
    if args.alignment_rtol is None:
        args.alignment_rtol = 0.05 if args.precision == "w4a16" or args.kv_dtype != "fp32" else 2e-4
    if args.strict_greedy is None:
        args.strict_greedy = args.precision == "fp16" and args.kv_dtype == "fp32"
    if not np.isfinite(args.alignment_rtol) or args.alignment_rtol <= 0:
        parser.error("--alignment-rtol must be positive")
    if (not np.isfinite(args.kv_alignment_rtol) or args.kv_alignment_rtol <= 0
            or args.kv_long_tokens < 1 or args.kv_quality_tokens < 0
            or min(args.kv_residual_tokens, args.kv_sink_tokens) < 0):
        parser.error("KV validation requires a positive tolerance/long length and nonnegative generation length")
    if args.steps is None:
        args.steps = "calibrate,export,validate,quality,compile" if args.precision == "w4a16" else "validate,export,compile"
    for field in ("model", "output", "ddk", "compiler_work_dir", "image", "calibration_images", "calibration_texts", "reuse_build"):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).expanduser().resolve())
    for source in (args.model, args.ddk):
        if source and (source == args.output or source in args.output.parents):
            parser.error("Output must be outside source model and SDK")
    if args.reuse_build and (args.reuse_build == args.output or args.reuse_build in args.output.parents
                            or args.output in args.reuse_build.parents):
        parser.error("Reuse source and output must be independent build directories")
    args.quality_images = [value.expanduser().resolve() for value in args.quality_images]
    if min(args.context, args.prefill_tokens, args.layers_per_chunk, args.vision_tokens,
           args.vision_layers_per_chunk, args.decode_steps, args.threads, args.group_size,
           args.calibration_samples, args.calibration_decode_steps, args.quality_tokens,
           args.gptq_block_size, args.kv_group_size) < 1 or args.prefill_tokens > args.context or args.gptq_damping <= 0:
        parser.error("Positive dimensions required; prefill block must fit context")
    if args.vision_tokens % 4:
        parser.error("Vision tokens must be divisible by the Qwen3-VL merge unit")
    for key, choices in (("components", {"llm", "head", "embedding", "vision"}),
                         ("modes", {"prefill", "decode"}),
                         ("steps", {"validate", "export", "compile", "calibrate", "quality", "reuse", "kv_validate"})):
        values = getattr(args, key).split(",")
        if not values or set(values) - choices or len(set(values)) != len(values):
            parser.error("Invalid --" + key)
        setattr(args, key, values)
    if args.precision == "fp16" and set(args.steps) & {"calibrate", "quality"}:
        parser.error("calibrate/quality stages require --precision w4a16")
    if not str(args.compiler_work_dir).isascii() or any(c not in "/_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                     for c in str(args.compiler_work_dir)):
        parser.error("Compiler scratch path must contain only ASCII letters, digits, / and _")
    return args


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    if args.precision == "w4a16":
        from w4a16 import W4Experiment
        experiment = W4Experiment(args)
    else:
        experiment = Experiment(args)
    for stage in args.steps:
        getattr(experiment, stage)()


if __name__ == "__main__":
    main()
