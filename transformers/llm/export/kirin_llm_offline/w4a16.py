"""SDK W4/group64 PTQ, paired native LUT export, and CPU reference graphs.

Calibration uses real full-model prefill/decode inputs; optional GPTQ fits weights.
No QAT training is performed.
Embedding and patch/position embedding remain floating point. KV storage is
independent of weight quantization and can use FP32, FP16 or dynamic INT8.
"""
import copy
import gc
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch

from experiment import Experiment, VISUAL_ROUTE, chunk_ranges, metrics, save_json, sha256
from static_kv import (StaticDecoderChunk, StaticHead, StaticSession, StaticVisionChunk,
                       StaticVisionPost, StaticVisionPre, StaticVisionFull)


class CachedQuantLinear(torch.nn.Module):
    """Same SDK fake-quant forward, with frozen dequantized weights cached once."""
    def __init__(self, source):
        super().__init__()
        with torch.no_grad():
            self.register_buffer("weight", source.quant_op.weight_quantizer(source.weight).detach().clone())
        self.bias = source.bias
        self.input_quantizer = source.quant_op.input_quantizer
        self.output_quantizer = getattr(source.quant_op, "output_quantizer", torch.nn.Identity())

    def forward(self, hidden):
        return self.output_quantizer(torch.nn.functional.linear(self.input_quantizer(hidden), self.weight, self.bias))


def cache_quantized_weights(module):
    for name, child in list(module.named_children()):
        if hasattr(getattr(child, "quant_op", None), "weight_quantizer"):
            setattr(module, name, CachedQuantLinear(child))
        else:
            cache_quantized_weights(child)
    return module


def encoded_copy(module, state, linear_type):
    module = copy.deepcopy(module)
    def replace(parent, prefix=""):
        for name, child in list(parent.named_children()):
            key = prefix + name
            if isinstance(child, CachedQuantLinear):
                weight = state[key + ".weight"]
                linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=child.bias is not None, device="meta")
                linear.weight = torch.nn.Parameter(weight, requires_grad=False)
                if child.bias is not None:
                    linear.bias = torch.nn.Parameter(child.bias.detach().clone(), requires_grad=False)
                setattr(parent, name, linear_type(linear))
            else:
                replace(child, key + ".")
    replace(module)
    return module.eval()


def lower_linear_gemm(model):
    """Normalize biased 2-D Linear exports to the SDK's MatMul + Add form."""
    initializers = {value.name: value for value in model.graph.initializer}
    nodes, added, possibly_unused = [], [], set()
    for node in model.graph.node:
        if node.op_type != "Gemm":
            nodes.append(node)
            continue
        attributes = {value.name: onnx.helper.get_attribute_value(value) for value in node.attribute}
        if attributes.get("alpha", 1) != 1 or attributes.get("beta", 1) != 1 or attributes.get("transA", 0):
            raise ValueError("Unsupported non-Linear Gemm attributes: " + node.name)
        if node.input[1] not in initializers:
            raise ValueError("Linear Gemm requires a constant weight: " + node.name)
        weight = node.input[1]
        if attributes.get("transB", 0):
            converted = weight + "_matmul"
            data = onnx.numpy_helper.to_array(initializers[weight]).T.copy()
            added.append(onnx.numpy_helper.from_array(data, name=converted))
            possibly_unused.add(weight)
            weight = converted
        biased = len(node.input) > 2 and bool(node.input[2])
        output = node.output[0] + "_matmul" if biased else node.output[0]
        nodes.append(onnx.helper.make_node("MatMul", [node.input[0], weight], [output], name=node.name))
        if biased:
            nodes.append(onnx.helper.make_node("Add", [output, node.input[2]], list(node.output), name=node.name + "_bias"))
    used = {value for node in nodes for value in node.input}
    retained = [value for value in model.graph.initializer if value.name not in possibly_unused or value.name in used]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained + added)
    return model


def image_inputs(model, tokenizer, path, args, prompt=None):
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    processor = Qwen2VLImageProcessorPil.from_pretrained(args.model, local_files_only=True)
    with Image.open(path) as original:
        image = original.convert("RGB").resize((args.image_width, args.image_height))
    pixels = dict(processor(images=[image], return_tensors="pt"))
    if int(pixels["image_grid_thw"].prod(-1).sum()) != args.vision_tokens:
        raise ValueError("Calibration/validation image does not match --vision-tokens")
    message = [{"role": "user", "content": [{"type": "image"},
               {"type": "text", "text": prompt or args.prompt}]}]
    text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    count = args.vision_tokens // processor.merge_size ** 2
    text = text.replace("<|image_pad|>", "<|image_pad|>" * count)
    inputs = dict(tokenizer(text, return_tensors="pt", add_special_tokens=False))
    inputs["mm_token_type_ids"] = (inputs["input_ids"] == model.config.image_token_id).long()
    inputs.update(pixels)
    return inputs


@torch.no_grad()
def static_prefill(model, inputs, args, capture=None, backend=None):
    """Run real inputs through the same static KV protocol used by export."""
    from types import SimpleNamespace
    from validation import static_image
    language, visual = model.model.language_model, model.model.visual
    cfg = model.config.text_config
    def execute(name, module, values):
        if capture is not None and name not in capture:
            capture[name] = [np.array(value, copy=True) for value in values]
        if backend is not None:
            return backend.run(name, values)
        result = module(*[torch.from_numpy(value) for value in values])
        result = result if isinstance(result, tuple) else (result,)
        return [value.numpy() for value in result]
    def vision_run(name, values):
        if name == "vision_full":
            module = StaticVisionFull(visual)
        elif name == "vision_pre":
            module = StaticVisionPre(visual)
        elif name == "vision_post":
            module = StaticVisionPost(visual)
        else:
            index = int(name.rsplit("_", 1)[1])
            start, end = chunk_ranges(len(visual.blocks), args.vision_layers_per_chunk)[index]
            module = StaticVisionChunk(visual.blocks[start:end], start, visual.deepstack_visual_indexes)
        return execute(name, module, values)
    ids = inputs["input_ids"]
    embedded = (language.embed_tokens(ids).clone() if backend is None
                else torch.from_numpy(backend.embedding(ids.numpy())))
    if capture is not None:
        padded = np.zeros((1, args.prefill_tokens), dtype=np.int32)
        count = min(args.prefill_tokens, ids.shape[1])
        padded[:, :count] = ids[:, :count].numpy()
        capture["embedding_prefill"] = [padded]
    dense = None
    if "image_grid_thw" in inputs:
        image, features, _ = static_image(model, inputs, args, SimpleNamespace(run=vision_run))
        mask = ids == model.config.image_token_id
        embedded[mask] = image
        dense = torch.zeros(len(features), *embedded.shape)
        for i, feature in enumerate(features):
            dense[i][mask] = feature
        positions, delta = model.model.get_rope_index(input_ids=ids, image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs.get("attention_mask"), mm_token_type_ids=inputs.get("mm_token_type_ids"))
    else:
        positions = torch.arange(ids.shape[1])[None, None].expand(3, 1, -1)
        delta = torch.zeros(1, 1, dtype=torch.long)
    chunks = []
    for index, (start, end) in enumerate(chunk_ranges(len(language.layers), args.layers_per_chunk)):
        module = StaticDecoderChunk(language.layers[start:end], cfg, args.kv_dtype, args.kv_group_size,
                                    args.kv_residual_tokens, args.kv_sink_tokens)
        def callback(decode, values, module=module, index=index):
            return execute(f"llm_{'decode' if decode else 'prefill'}_{index}", module, values)
        chunks.append((start, end - start, callback))
    session = StaticSession(chunks, cfg, args.context, args.prefill_tokens,
                            kv_dtype=args.kv_dtype, kv_group_size=args.kv_group_size,
                            kv_residual_tokens=args.kv_residual_tokens, kv_sink_tokens=args.kv_sink_tokens)
    cos, sin = language.rotary_emb(embedded, positions)
    hidden = session.forward(embedded.numpy(), cos[:, None].numpy(), sin[:, None].numpy(), None if dense is None else dense.numpy())
    logits = execute("lm_head", StaticHead(language.norm, model.lm_head), [hidden[:, -1:]])[0]
    return session, logits, int(delta.item())


@torch.no_grad()
def static_generate(model, inputs, args, backend=None):
    session, logits, delta = static_prefill(model, inputs, args, backend=backend)
    language = model.model.language_model
    end = model.generation_config.eos_token_id
    stop = set(end if isinstance(end, list) else [end])
    tokens = []
    for _step in range(args.quality_tokens):
        token = int(logits[0, -1].argmax())
        if token in stop:
            break
        tokens.append(token)
        if len(tokens) == args.quality_tokens:
            break
        ids = np.array([[token]], dtype=np.int32)
        embedded = (language.embed_tokens(torch.from_numpy(ids)) if backend is None
                    else torch.from_numpy(backend.embedding(ids, decode=True)))
        positions = torch.full((3, 1, 1), session.cache.length + delta, dtype=torch.long)
        cos, sin = language.rotary_emb(embedded, positions)
        hidden = session.forward(embedded.numpy(), cos[:, None].numpy(), sin[:, None].numpy(), decode=True)
        logits = (model.lm_head(language.norm(torch.from_numpy(hidden))).numpy() if backend is None
                  else backend.run("lm_head", [hidden])[0])
    return tokens


class W4Experiment(Experiment):
    def sdk(self):
        if self.args.ddk is None:
            raise ValueError("--ddk is required for W4A16")
        sys.path.insert(0, str(self.args.ddk / "tools/tools_dopt/dopt_pytorch_py3"))
        sys.path.insert(0, str(VISUAL_ROUTE))
        import visual_plugin_quant_matmul_route as route
        from dopt.dopt_lm import do_opt
        # A preceding FP16 export can import the optional-DOPT route before
        # this SDK path is available. Refresh that failed optional import.
        if route.DOPT_IMPORT_ERROR is not None:
            route = importlib.reload(route)
        route.require_dopt()
        return route, do_opt

    @property
    def quant_dir(self):
        return self.output / "quantization"

    def load_model(self):
        if getattr(self, "sdk_model", None) is None:
            _route, sdk = self.sdk()
            record = self.manifest.get("calibration")
            if not record:
                raise ValueError("Run --steps calibrate before W4A16 export/validation")
            for key in ("config", "checkpoint"):
                if sha256(self.output / record[key]) != record[key + "_sha256"]:
                    raise ValueError("Calibrated artifact changed: " + key)
            original = Experiment.load_model(self)
            if self.args.weight_method == "gptq" and self.args.head_bit == 4:
                original.lm_head.weight = torch.nn.Parameter(original.lm_head.weight.detach().clone())
            self.sdk_model = sdk.optimize_model(original, str(self.output / record["config"]))
            state = torch.load(self.output / record["checkpoint"], map_location="cpu", weights_only=True)
            self.sdk_model.load_state_dict(state, strict=True)
            if self.args.activation_range_multiplier != 1 or self.args.output_range_multiplier != 1:
                with torch.no_grad():
                    for layer in self.sdk_model.modules():
                        quant = getattr(layer, "quant_op", None)
                        if quant is None:
                            continue
                        for name in ("input_quantizer", "output_quantizer"):
                            point = getattr(quant, name, None)
                            if point is not None and hasattr(point, "s"):
                                factor = self.args.activation_range_multiplier
                                if name == "output_quantizer": factor *= self.args.output_range_multiplier
                                point.s.mul_(factor)
            sdk.set_calibrate_state(self.sdk_model, False)
            sdk.set_quant_state(self.sdk_model, True, True)
            self.model = cache_quantized_weights(copy.deepcopy(self.sdk_model)).eval()
            del state
            gc.collect()
        return self.model

    def calibration_rows(self):
        a = self.args
        if not a.calibration_images or not a.calibration_images.is_dir():
            raise ValueError("--calibration-images is required for full VLM W4A16 calibration")
        excluded = {a.image, *a.quality_images}
        images = sorted(path for path in a.calibration_images.iterdir()
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and path not in excluded)
        if not images:
            raise ValueError("No calibration images (validation image is excluded)")
        texts = ["请说明如何整理一天的工作计划。", "用中文解释什么是机器学习。", "写一封简短的会议邀请。",
                 "如何在手机应用中搜索并购买车票？", "计算 17 加 25，并解释步骤。", "介绍健康饮食的基本原则。",
                 "将这句话翻译成英语：明天上午十点开会。", "请总结下面的内容。" + "用户可以在应用首页搜索商品、查看详情并加入购物车。" * 12]
        if a.calibration_texts:
            rows = json.loads(a.calibration_texts.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"] for row in rows):
                raise ValueError("Calibration text JSON must be a nonempty list of {text: prompt}")
            texts = [row["text"] for row in rows]
        # Start with an image so every visual/text/head quantizer is exercised.
        rows = []
        for i in range(a.calibration_samples):
            if i % 2 == 0:
                path = images[(i // 2) % len(images)]
                rows.append({"image": str(path), "image_sha256": sha256(path), "text": a.prompt})
            else:
                rows.append({"text": texts[(i // 2) % len(texts)]})
        return rows

    @torch.no_grad()
    def calibrate(self):
        from transformers import AutoTokenizer
        route, sdk = self.sdk()
        rows = self.calibration_rows()
        original = Experiment.load_model(self)
        self.quant_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.quant_dir / "config.json"
        sdk.generate_config_file(original, str(config_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for name, entry in config["layer_strategy"].items():
            entry["quant_strategy"] = "float"
            if entry["type"] != str(torch.nn.Linear) or (name == "lm_head" and self.args.head_bit == 16):
                continue
            layer = original.get_submodule(name)
            if layer.in_features % self.args.group_size:
                raise ValueError("Group size does not divide Linear input: " + name)
            entry.update(quant_strategy="Quant_lm_head" if name == "lm_head" else "Quant_act_weight_eco",
                         weight={"bit": 4, "group_size": self.args.group_size},
                         input={"bit": 16, "unsigned_quant": False, "input_algo": "min_max"},
                         output={"bit": 16, "per_channel": True, "input_algo": "min_max"})
        save_json(config_path, config)
        tokenizer = AutoTokenizer.from_pretrained(self.args.model, local_files_only=True)
        hessians = None
        if self.args.weight_method == "gptq":
            if not torch.cuda.is_available() or not self.args.quant_device.startswith("cuda"):
                raise ValueError("GPTQ requires a CUDA-enabled Torch environment; the existing CPU environment is unchanged")
            # Head and embedding are tied in this source model. Updating the
            # head must not silently quantize the floating-point embedding.
            if self.args.head_bit == 4:
                original.lm_head.weight = torch.nn.Parameter(original.lm_head.weight.detach().clone())
            hessians = self.collect_hessians(original, config, tokenizer, rows)
        model = sdk.optimize_model(original, str(config_path)).eval()
        sdk.set_quant_state(model, True, True)
        sdk.set_calibrate_state(model, True)
        optimization = None
        if hessians is not None:
            optimization = self.optimize_weights(model, hessians)
            model.to(self.args.quant_device)
            # GPTQ weights already lie on the SDK grid. Bypass repeated
            # weight quantization while fitting only activation ranges.
            sdk.set_quant_state(model, False, True)
            for layer in model.modules():
                quantizer = getattr(getattr(layer, "quant_op", None), "weight_quantizer", None)
                if quantizer is not None:
                    quantizer.set_calibrate_state(False)
        for i, row in enumerate(rows):
            if "image" in row:
                inputs = image_inputs(model, tokenizer, row["image"], self.args, row["text"])
            else:
                text = tokenizer.apply_chat_template([{"role": "user", "content": row["text"]}], tokenize=False, add_generation_prompt=True)
                inputs = dict(tokenizer(text, return_tensors="pt", add_special_tokens=False))
            device = next(model.parameters()).device
            inputs = {name: value.to(device) for name, value in inputs.items()}
            length = inputs["input_ids"].shape[1]
            if length + self.args.calibration_decode_steps > self.args.context:
                raise ValueError("Calibration prompt exceeds KV capacity")
            # Calibrate the head on all positions, not just one logit vector.
            result = model(**inputs, use_cache=True, return_dict=True)
            if i == 0:
                for layer in model.modules():
                    quantizer = getattr(getattr(layer, "quant_op", None), "weight_quantizer", None)
                    if quantizer is not None:
                        quantizer.set_calibrate_state(False)
            delta = int(model.model.rope_deltas.item()) if "image" in row else 0
            for step in range(self.args.calibration_decode_steps):
                token = result.logits[:, -1:].argmax(-1)
                positions = torch.full((3, 1, 1), length + step + delta, dtype=torch.long, device=device)
                result = model(input_ids=token, position_ids=positions, past_key_values=result.past_key_values,
                               use_cache=True, return_dict=True)
            row.update(prefill_tokens=length, decode_steps=self.args.calibration_decode_steps)
            print(f"calibrated {i + 1}/{len(rows)}: {'image' if 'image' in row else 'text'}, tokens={length}", flush=True)
        sdk.set_calibrate_state(model, False)
        sdk.set_quant_state(model, True, True)
        model.cpu()
        for name, value in model.state_dict().items():
            if name.endswith("quantizer.s") and (not torch.isfinite(value).all() or not (value > 0).all()):
                raise ValueError("Invalid calibrated scale: " + name)
        state_path = self.quant_dir / "calibrated.pth"
        torch.save(model.state_dict(), state_path)
        record = {"method": "SDK plugin PTQ with " + self.args.weight_method + " weights; no QAT training",
                  "weight_optimization": optimization, "config": str(config_path.relative_to(self.output)),
                  "config_sha256": sha256(config_path), "checkpoint": str(state_path.relative_to(self.output)),
                  "checkpoint_sha256": sha256(state_path), "samples": rows,
                  "settings": route.quantization_report(config_path, model.state_dict()),
                  "embedding_precision": "floating point", "kv_quantized": False}
        save_json(self.quant_dir / "calibration_report.json", record)
        self.manifest["calibration"] = record
        self.manifest["graphs"] = []
        self.manifest.pop("host_validation", None)
        self.save()
        self.sdk_model = model
        self.model = cache_quantized_weights(copy.deepcopy(model)).eval()

    @torch.no_grad()
    def collect_hessians(self, model, config, tokenizer, rows):
        from gptq import Hessian
        device = self.args.quant_device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model.to(device)
        hessians, handles = {}, []
        for name, settings in config["layer_strategy"].items():
            if settings["quant_strategy"] == "float":
                continue
            hessians[name] = Hessian(model.get_submodule(name).in_features, device)
            handles.append(model.get_submodule(name).register_forward_hook(hessians[name].add))
        try:
            for i, row in enumerate(rows):
                if "image" in row:
                    inputs = image_inputs(model, tokenizer, row["image"], self.args, row["text"])
                else:
                    text = tokenizer.apply_chat_template([{"role": "user", "content": row["text"]}], tokenize=False, add_generation_prompt=True)
                    inputs = dict(tokenizer(text, return_tensors="pt", add_special_tokens=False))
                inputs = {key: value.to(device) for key, value in inputs.items()}
                length = inputs["input_ids"].shape[1]
                if length + self.args.calibration_decode_steps > self.args.context:
                    raise ValueError("GPTQ calibration prompt exceeds KV capacity")
                result = model(**inputs, use_cache=True, return_dict=True)
                delta = int(model.model.rope_deltas.item()) if "image" in row else 0
                for step in range(self.args.calibration_decode_steps):
                    token = result.logits[:, -1:].argmax(-1)
                    positions = torch.full((3, 1, 1), length + step + delta, dtype=torch.long, device=device)
                    result = model(input_ids=token, position_ids=positions, past_key_values=result.past_key_values,
                                   use_cache=True, return_dict=True)
                print(f"Hessian inputs {i + 1}/{len(rows)}", flush=True)
        finally:
            for handle in handles:
                handle.remove()
            model.cpu()
        return hessians

    @torch.no_grad()
    def optimize_weights(self, model, hessians):
        from gptq import quantize
        report = []
        for index, name in enumerate(list(hessians)):
            history = hessians.pop(name)
            if history.tokens < 1:
                raise ValueError("GPTQ layer received no inputs: " + name)
            layer = model.get_submodule(name)
            quantizer = layer.quant_op.weight_quantizer
            quantizer.set_calibrate_state(True)
            expected = quantizer(layer.weight).detach()
            quantizer.set_calibrate_state(False)
            groups = self.args.group_size
            scales = quantizer.s.detach().reshape(layer.weight.shape[0], -1)
            expanded = scales.repeat_interleave(groups, dim=1)
            baseline = torch.round(layer.weight / expanded).clamp(-8, 7) * expanded
            if not torch.equal(expected, baseline):
                raise ValueError("SDK weight quantizer differs from the supported signed INT4 grid: " + name)
            device = history.matrix.device
            weight, loss = quantize(layer.weight.detach().to(device), history.matrix / history.tokens,
                                    scales.to(device), groups, self.args.gptq_damping, self.args.gptq_block_size)
            layer.weight.copy_(weight.cpu())
            # Serialization must see exactly the SDK's native dequantized grid.
            layer.weight.copy_(quantizer(layer.weight))
            loss.update(name=name, calibration_tokens=history.tokens)
            report.append(loss)
            print(f"GPTQ {index + 1}: {name}, reconstruction loss ratio={loss['loss_ratio']:.4f}", flush=True)
            del weight, history, expected, baseline, expanded
        path = self.quant_dir / "gptq_report.json"
        save_json(path, {"algorithm": "GPTQ with frozen SDK scales and original contiguous groups", "layers": report})
        return str(path.relative_to(self.output))

    def sdk_component(self, name, metadata):
        model = self.sdk_model
        language, visual = model.model.language_model, model.model.visual
        component = metadata["component"]
        if component == "llm":
            start, end = metadata["layer_start"], metadata["layer_end"]
            return "llm_" + str(start), StaticDecoderChunk(language.layers[start:end], model.config.text_config,
                self.args.kv_dtype, self.args.kv_group_size, self.args.kv_residual_tokens, self.args.kv_sink_tokens)
        if component == "head":
            return "head", StaticHead(language.norm, model.lm_head)
        if component == "vision_blocks":
            start, end = metadata["layer_start"], metadata["layer_end"]
            return "vision_" + str(start), StaticVisionChunk(visual.blocks[start:end], start, visual.deepstack_visual_indexes)
        if component == "vision_post":
            return "vision_post", StaticVisionPost(visual)
        if component == "vision_full":
            return "vision_full", StaticVisionFull(visual)
        raise ValueError("Unexpected quantized component: " + name)

    @torch.no_grad()
    def export(self):
        from transformers import AutoTokenizer
        model = self.load_model()
        tokenizer = AutoTokenizer.from_pretrained(self.args.model, local_files_only=True)
        path = self.args.image or next(Path(row["image"]) for row in self.manifest["calibration"]["samples"] if "image" in row)
        inputs = image_inputs(model, tokenizer, path, self.args)
        self.export_samples = {}
        session, logits, delta = static_prefill(model, inputs, self.args, self.export_samples)
        token = torch.tensor([[int(logits[0, -1].argmax())]], dtype=torch.int32)
        self.export_samples["embedding_decode"] = [token.numpy()]
        embedded = model.model.language_model.embed_tokens(token)
        positions = torch.full((3, 1, 1), session.cache.length + delta, dtype=torch.long)
        cos, sin = model.model.language_model.rotary_emb(embedded, positions)
        session.forward(embedded.numpy(), cos[:, None].numpy(), sin[:, None].numpy(), decode=True)
        super().export()
        del self.export_samples

    def export_graph(self, name, module, inputs, names, outputs, **metadata):
        if name in getattr(self, "export_samples", {}):
            inputs = [torch.from_numpy(value) for value in self.export_samples[name]]
            metadata["validation_input_source"] = "real image prompt through static whole-model KV prefill/decode"
        quantized = any(isinstance(layer, CachedQuantLinear) for layer in module.modules())
        super().export_graph(name, module, inputs, names, outputs, **metadata)
        if not quantized:
            return
        route, _sdk = self.sdk()
        entry = next(value for value in self.manifest["graphs"] if value["name"] == name)
        path = self.output / entry["onnx"]
        reference_dir = path.parent / "reference"
        reference_dir.mkdir(exist_ok=True)
        reference = {key: entry[key] for key in ("onnx", "onnx_sha256", "external_data_sha256")}
        for file in [path.name, *entry["external_data_sha256"]]:
            (path.parent / file).replace(reference_dir / file)
        reference["onnx"] = str((reference_dir / path.name).relative_to(self.output))
        group, quantized_module = self.sdk_component(name, metadata)
        directory = self.quant_dir / group
        route.ensure_route_layout(str(directory))
        audit_path = directory / "encoding.json"
        calibrated_sha = self.manifest["calibration"]["checkpoint_sha256"]
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        if (audit.get("calibrated_sha256") != calibrated_sha or
                audit.get("activation_range_multiplier", 1) != self.args.activation_range_multiplier or
                audit.get("output_range_multiplier", 1) != self.args.output_range_multiplier):
            audit = route.serialize_quantized_weights(quantized_module, str(directory), True)
            audit["calibrated_sha256"] = calibrated_sha
            audit["activation_range_multiplier"] = self.args.activation_range_multiplier
            audit["output_range_multiplier"] = self.args.output_range_multiplier
            save_json(audit_path, audit)
        parameters = directory / "quant_output/quant_params_file"
        weights_path = directory / "quant_output/fake_quant_weight.pth"
        if sha256(parameters) != audit["quant_params_sha256"] or sha256(weights_path) != audit["fake_quant_weight_sha256"]:
            raise ValueError("Native weight/parameter pair changed: " + group)
        fake = torch.load(weights_path, map_location="cpu", weights_only=True)
        encoded = encoded_copy(module, fake, route._FLinearMatmul)
        with tempfile.TemporaryDirectory(prefix="native_", dir=path.parent) as temporary:
            native = Path(temporary) / "model.onnx"
            with torch.no_grad():
                torch.onnx.export(encoded, tuple(inputs), str(native), opset_version=12, dynamo=False,
                                  do_constant_folding=True, input_names=names, output_names=outputs)
            ir = onnx.load(str(native), load_external_data=False).ir_version
            route.process_onnx_for_omg(str(native), fp16=False)
            model = lower_linear_gemm(onnx.load(str(native)))
            model.ir_version = ir
            nodes = {node.name: node for node in model.graph.node}
            initializers = {value.name: onnx.numpy_helper.to_array(value) for value in model.graph.initializer}
            bindings = []
            for layer in audit["layers"]:
                key = layer["name"]
                exported = key.replace(".layer.self_attn.", ".").replace(".layer.mlp.", ".mlp.")
                if metadata["component"] in ("vision_blocks", "vision_full"):
                    exported = exported.replace(".attn.", ".")
                node = nodes.get(key, nodes.get(exported))
                expected = fake[key + ".weight"].detach().numpy().T
                if node is None:
                    matches = [value for value in model.graph.node if value.op_type == "MatMul"
                               and value.input[1] in initializers
                               and initializers[value.input[1]].shape == expected.shape
                               and np.array_equal(initializers[value.input[1]], expected)]
                    if len(matches) == 1:
                        node = matches[0]
                if node is None or node.op_type != "MatMul" or node.input[1] not in initializers:
                    candidates = [(v.name, v.op_type) for v in model.graph.node if v.op_type in {"MatMul", "Gemm"}]
                    raise ValueError("Missing paired MatMul: " + layer["name"] + "; found " + str(candidates))
                actual = initializers[node.input[1]]
                if actual.dtype != np.float32 or not np.array_equal(actual, expected):
                    raise ValueError("Native ONNX weight differs from SDK indices: " + layer["name"])
                if layer["dtype"] != "UINT4" or not layer["lut_mode"]:
                    raise ValueError("Expected native 4-bit LUT encoding")
                # A manual attention call skips the HF parent forward scopes
                # in Torch's node names. Rename only after exact weight audit.
                bindings.append({"sdk_name": key, "exported_name": node.name})
                node.name = key
            onnx.save(model, str(native), save_as_external_data=True, all_tensors_to_one_file=True,
                      location="native.pb", size_threshold=1024)
            meta = onnx.load(str(native), load_external_data=False)
            external = {field.value for value in meta.graph.initializer for field in value.external_data if field.key == "location"}
            for file in [native.name, *external]:
                shutil.copy2(native.parent / file, path.parent / file)
        entry.update(onnx_sha256=sha256(path), external_data_sha256={file: sha256(path.parent / file) for file in external},
                     reference=reference, torch_ort_scope="dequantized weight and activation reference ONNX",
                     quantization={"parameters": str(parameters.relative_to(self.output)),
                                   "parameters_sha256": audit["quant_params_sha256"], "weight_bit": 4,
                                   "group_size": self.args.group_size, "activation_bit": 16,
                                   "node_name_bindings": bindings,
                                   "native_initializers_verified": True, "shared_weight_group": group})
        self.save()
        del encoded, model, fake, quantized_module
        gc.collect()

    def validate(self):
        from validation import validate_model
        model = self.load_model()
        controls = [(quantizer, quantizer.bypass) for layer in model.modules() if isinstance(layer, CachedQuantLinear)
                    for quantizer in (layer.input_quantizer, layer.output_quantizer) if hasattr(quantizer, "bypass")]
        weight_args = copy.copy(self.args)
        weight_args.validation_backend = "torch"
        weight_args.validation_tag = "weight_only"
        weight_args.alignment_rtol = 2e-4
        weight_args.strict_greedy = True
        weight_args.kv_dtype = "fp32"
        try:
            for quantizer, _value in controls:
                quantizer.bypass = True
            weight_result = validate_model(model, weight_args)
        finally:
            for quantizer, value in controls:
                quantizer.bypass = value
        previous = self.manifest.setdefault("host_validation", {})
        previous["weight_only"] = weight_result
        self.save()
        previous[self.args.validation_backend] = validate_model(model, self.args)
        self.save()

    @torch.no_grad()
    def quality(self):
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        quantized = self.load_model()
        original = Qwen3VLForConditionalGeneration.from_pretrained(
            self.args.model, dtype=torch.float32, attn_implementation="eager", local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.args.model, local_files_only=True)
        prompts = ["请简单介绍一下你自己。", "请计算 23 加 19，并说明结果。"]
        specifications = [(f"text_{i}", prompt, None) for i, prompt in enumerate(prompts)]
        images = list(dict.fromkeys([value for value in [self.args.image, *self.args.quality_images] if value]))
        calibration_images = {row.get("image") for row in self.manifest["calibration"]["samples"]}
        if any(str(path) in calibration_images for path in images):
            raise ValueError("Quality images must not be calibration images")
        specifications += [(f"image_{i}", self.args.prompt, path) for i, path in enumerate(images)]
        cases = []
        for name, prompt, image in specifications:
            if image:
                inputs = image_inputs(original, tokenizer, image, self.args, prompt)
            else:
                text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                inputs = dict(tokenizer(text, return_tensors="pt", add_special_tokens=False))
            count = inputs["input_ids"].shape[1]
            if count + self.args.quality_tokens > self.args.context:
                raise ValueError("Quality generation exceeds the compiled KV capacity")
            ref = original(**inputs, logits_to_keep=1, return_dict=True).logits.numpy()
            candidate = quantized(**inputs, logits_to_keep=1, return_dict=True).logits.numpy()
            answers = {}
            for backend, model in (("original", original), ("w4a16", quantized)):
                answer = model.generate(**inputs, do_sample=False, max_new_tokens=self.args.quality_tokens, use_cache=True)
                answers[backend] = tokenizer.decode(answer[0, count:], skip_special_tokens=True)
            answers["w4a16_static_kv"] = tokenizer.decode(static_generate(quantized, inputs, self.args), skip_special_tokens=True)
            if self.args.kv_dtype != "fp32":
                baseline_args = copy.copy(self.args)
                baseline_args.kv_dtype = "fp32"
                answers["w4a16_static_fp32_kv"] = tokenizer.decode(
                    static_generate(quantized, inputs, baseline_args), skip_special_tokens=True)
            result = {"name": name, "image": str(image) if image else None, "prompt": prompt,
                      "logits": metrics(ref, candidate), "answers": answers}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            cases.append(result)
        report = {"comparison": "quantization quality against original float32 HF; greedy equality is not required",
                  "device_tested": False, "kv_dtype": self.args.kv_dtype, "cases": cases}
        save_json(self.output / "quantization_quality.json", report)
        self.manifest["quantization_quality"] = {"report": "quantization_quality.json", "cases": len(cases)}
        self.save()
