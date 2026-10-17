"""Run the actual portable App C++ scheduler against CPU ONNX references.

The Python parent independently checks tokenization, HF vision constants and
M-RoPE, hidden-state transfers, masks, image/DeepStack placement, and every
compressed KV input. This does not execute OMC.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import types

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel, Qwen3VLTextRotaryEmbedding, Qwen3VLVisionModel, Qwen3VLVisionRotaryEmbedding,
)
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

from experiment import ONNXGraphs, metrics
from static_kv import KVCache, attention_mask


def exact(stream, count):
    out = bytearray()
    while len(out) < count:
        block = stream.read(count - len(out))
        if not block:
            raise EOFError("C++ runtime closed its backend pipe")
        out.extend(block)
    return bytes(out)


def verify(args):
    cfg = json.loads((args.package / "config.json").read_text())
    p = cfg["runtime"]
    contracts = {g["name"]: g for g in cfg["graphs"]}
    source = AutoConfig.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rotary = Qwen3VLTextRotaryEmbedding(source.text_config)
    # Inspect the actual HF defaults without allocating the vision weights.
    with torch.device("meta"):
        visual = Qwen3VLVisionModel(source.vision_config)
    vision_grid = torch.tensor([[1, p["image_height"] // p["patch"], p["image_width"] // p["patch"]]])
    vision_indices, vision_weights = get_vision_interpolation_indices_and_weights(
        vision_grid, num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners, spatial_merge_size=visual.spatial_merge_size)
    vision_positions = get_vision_position_ids(vision_grid, visual.spatial_merge_size)
    vision_rotary = Qwen3VLVisionRotaryEmbedding(source.vision_config.hidden_size // source.vision_config.num_heads // 2)
    vision_angles = vision_rotary(vision_positions).reshape(int(vision_grid.prod()), -1)
    vision_angles = torch.cat((vision_angles, vision_angles), dim=-1)
    vision_cos, vision_sin = vision_angles.cos()[None, None].numpy(), vision_angles.sin()[None, None].numpy()
    hf = types.SimpleNamespace(config=source)
    hf.get_vision_position_ids = types.MethodType(Qwen3VLModel.get_vision_position_ids, hf)
    processor = Qwen2VLImageProcessorPil.from_pretrained(args.model, local_files_only=True)
    backend = ONNXGraphs(args.build, threads=args.threads)
    sliced_backend = ONNXGraphs(args.embedding_slice_build, threads=args.threads) if args.embedding_slice_build else None
    if any(g.get("embedding_slices") for g in contracts.values()) and sliced_backend is None:
        raise ValueError("A sliced package requires --embedding-slice-build to verify its actual physical graphs")
    reports = []
    prompts = args.text_prompt or ["请简单介绍一下你自己。"]
    cases = [("text" if len(prompts) == 1 else "text_" + str(i), [], query)
             for i, query in enumerate(prompts)]
    cases += [("image_" + str(i), [image], "请描述这张图片的内容。") for i, image in enumerate(args.image)]
    if args.only_multi_image:
        if len(args.image) < 2:
            raise ValueError("Provide at least two --image values")
        cases = [("multiple_images", args.image, "请描述这张图片的内容。")]
    for case, image_paths, query in cases:
        content = "".join(("<img><hw>608,256</hw>" + str(path) + "</img>") if i % 2 == 0 else
                          ("<img>" + str(path) + "<hw>608,256</hw></img>") for i, path in enumerate(image_paths)) + query
        messages = ([{"role": "system", "content": args.system_prompt}] if args.system_prompt else [])
        messages.append({"role": "user", "content": content})
        raw = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        expanded = raw
        image_tokens = p["image_height"] * p["image_width"] // p["patch"] ** 2 // p["merge"] ** 2
        expanded = re.sub(r"<img>.*?</img>", "<|vision_start|>" + "<|image_pad|>" * image_tokens + "<|vision_end|>", expanded)
        ids = tokenizer(expanded, add_special_tokens=False, return_tensors="pt")["input_ids"]
        expanded_messages = [dict(message) for message in messages]
        expanded_messages[-1]["content"] = re.sub(r"<img>.*?</img>", "<|vision_start|>" + "<|image_pad|>" * image_tokens + "<|vision_end|>", content)
        assert tokenizer.apply_chat_template(expanded_messages, tokenize=False, add_generation_prompt=True) == expanded
        grid = torch.tensor([[1, p["image_height"] // p["patch"], p["image_width"] // p["patch"]]] * len(image_paths)) if image_paths else None
        positions, delta = Qwen3VLModel.get_rope_index(hf, input_ids=ids, image_grid_thw=grid,
                                                   mm_token_type_ids=(ids == source.image_token_id).long())
        gold_cos, gold_sin = rotary(torch.empty(1, ids.shape[1], p["hidden"]), positions)
        cache = KVCache(p["layers"], p["kv_heads"], p["capacity"], p["head_dim"],
                        np.float16 if p.get("kv_dtype") == "fp16" else np.int8,
                        p["kv_group"], p["kv_recent"], p["kv_sink"])
        expected_patches = []
        for path in image_paths:
            image = Image.open(path).convert("RGB").resize((p["image_width"], p["image_height"]), Image.Resampling.BICUBIC)
            expected_patches.append(processor(images=[image], return_tensors="np")["pixel_values"])
        input_path = args.output / (case + "_input.txt")
        result_path = args.output / (case + "_output.txt")
        input_path.write_text(raw, encoding="utf-8")
        stderr = (args.output / (case + "_stderr.log")).open("wb")
        child = subprocess.Popen([str(args.driver), str(args.package / "config.json"), str(input_path), str(result_path), str(args.decode_steps)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
        phase_offset = 0
        next_token = None
        real = 0
        width = 0
        calls, checks = 0, 0
        patch_results = []
        image_features = []
        image_index = 0
        generated = []
        vision_hidden, hidden = None, None
        vision_deep = []
        try:
            while True:
                prefix = child.stdout.read(4)
                if not prefix:
                    break
                length = struct.unpack("<I", prefix)[0]
                if not 0 < length < 128:
                    raise ValueError("Unexpected stdout before backend graph: " + repr(prefix))
                name = exact(child.stdout, length).decode()
                graph = contracts[name]
                arrays = []
                for port in graph["inputs"]:
                    size = struct.unpack("<Q", exact(child.stdout, 8))[0]
                    expected = int(np.prod(port["shape"])) * np.dtype(port["dtype"]).itemsize
                    assert size == expected, (name, port["name"], size, expected)
                    arrays.append(np.frombuffer(exact(child.stdout, size), dtype=port["dtype"]).reshape(port["shape"]).copy())
                calls += 1
                if name in ("vision_pre", "vision_full"):
                    result = metrics(expected_patches[image_index], arrays[0])
                    assert result["relative_l2"] < .015 and result["max_abs"] < .04, result
                    patch_results.append(result)
                    image_index += 1
                    np.testing.assert_array_equal(arrays[1], vision_indices.int().numpy())
                    np.testing.assert_allclose(arrays[2], vision_weights.numpy(), atol=1e-7, rtol=1e-7)
                    vision_deep = []
                    checks += 3
                    if name == "vision_full":
                        np.testing.assert_allclose(arrays[3], vision_cos, atol=3e-5, rtol=2e-5)
                        np.testing.assert_allclose(arrays[4], vision_sin, atol=3e-5, rtol=2e-5)
                        np.testing.assert_array_equal(arrays[5], np.zeros_like(arrays[5]))
                        checks += 3
                if name.startswith("vision_blocks_"):
                    np.testing.assert_array_equal(arrays[0], vision_hidden)
                    np.testing.assert_allclose(arrays[1], vision_cos, atol=3e-5, rtol=2e-5)
                    np.testing.assert_allclose(arrays[2], vision_sin, atol=3e-5, rtol=2e-5)
                    np.testing.assert_array_equal(arrays[3], np.zeros_like(arrays[3]))
                    checks += 4
                if name == "vision_post":
                    assert len(arrays) == 1 + len(vision_deep)
                    for actual, expected in zip(arrays, [vision_hidden, *vision_deep]):
                        np.testing.assert_array_equal(actual, expected)
                    checks += len(arrays)
                if name.startswith("embedding_"):
                    decode = name.endswith("decode")
                    width = 1 if decode else p["prefill"]
                    real = 1 if decode else min(width, ids.shape[1] - cache.length)
                    gold_ids = np.full((1, width), p["pad_token"], dtype=np.int32)
                    gold_ids[:, :real] = next_token if decode else ids.numpy()[:, cache.length:cache.length + real]
                    np.testing.assert_array_equal(arrays[0], gold_ids)
                    phase_offset = cache.length
                    checks += 1
                if name.startswith("llm_"):
                    decode = name.startswith("llm_decode")
                    np.testing.assert_array_equal(arrays[0], hidden)
                    gold_mask = attention_mask(p["capacity"], cache.length, real, width, np.float32, p["kv_recent"], p["kv_sink"])
                    np.testing.assert_array_equal(arrays[3], gold_mask)
                    if decode:
                        pos = torch.full((3, 1, 1), cache.length + int(delta.item()), dtype=torch.long)
                        rc, rs = rotary(torch.empty(1, 1, p["hidden"]), pos)
                    else:
                        rc = torch.ones(1, width, p["head_dim"])
                        rs = torch.zeros_like(rc)
                        rc[:, :real] = gold_cos[:, cache.length:cache.length + real]
                        rs[:, :real] = gold_sin[:, cache.length:cache.length + real]
                    np.testing.assert_allclose(arrays[1], rc[:, None].numpy(), atol=3e-5, rtol=2e-5)
                    np.testing.assert_allclose(arrays[2], rs[:, None].numpy(), atol=3e-5, rtol=2e-5)
                    start, end = graph["layer_start"], graph["layer_end"]
                    for actual, gold in zip(arrays[5:], cache.inputs(start, end - start)):
                        np.testing.assert_array_equal(actual, gold)
                    deep = np.zeros_like(arrays[4])
                    visual = (ids.numpy()[0] == p["image_token"])
                    if not decode:
                        image_row = 0
                        for token_index in range(ids.shape[1]):
                            if not visual[token_index]:
                                continue
                            image_id, row = divmod(image_row, image_tokens)
                            if phase_offset <= token_index < phase_offset + real:
                                for layer in range(start, min(end, image_features[image_id][1].shape[0])):
                                    deep[layer - start, 0, token_index - phase_offset] = image_features[image_id][1][layer, 0, row]
                            image_row += 1
                    np.testing.assert_array_equal(arrays[4], deep)
                    checks += 5 + len(arrays[5:])
                if name == "lm_head":
                    np.testing.assert_array_equal(arrays[0], hidden[:, real - 1:real])
                    checks += 1
                outputs = backend.run(name, arrays)
                if graph.get("embedding_slices"):
                    physical = [sliced_backend.run(name + "_slice_" + str(i), arrays)[0]
                                for i in range(len(graph["embedding_slices"]))]
                    assembled = np.concatenate(physical, axis=-1)
                    np.testing.assert_array_equal(assembled, outputs[0])
                    outputs = [assembled]
                    checks += 1
                if name == "vision_pre" or name.startswith("vision_blocks_"):
                    vision_hidden = outputs[0]
                    vision_deep.extend(outputs[1:])
                if name in ("vision_post", "vision_full"):
                    image_features.append(outputs)
                if name.startswith("embedding_"):
                    hidden = outputs[0].copy()
                    if not name.endswith("decode"):
                        visual_rows = np.flatnonzero(ids.numpy()[0] == p["image_token"])
                        for row, token_index in enumerate(visual_rows):
                            if phase_offset <= token_index < phase_offset + real:
                                image_id, image_row = divmod(row, image_tokens)
                                hidden[0, token_index - phase_offset] = image_features[image_id][0][0, image_row]
                if name.startswith("llm_"):
                    hidden = outputs[0]
                    cache.append_chunk(graph["layer_start"], outputs[1:], real)
                    if graph["layer_end"] == p["layers"]:
                        cache.commit(real)
                if name == "lm_head":
                    next_token = int(outputs[0].argmax())
                    if not tokenizer.convert_ids_to_tokens(next_token) == "<|im_end|>": generated.append(next_token)
                child.stdin.write(struct.pack("<I", len(outputs)))
                for value in outputs:
                    value = np.asarray(value, dtype=np.float32)
                    child.stdin.write(struct.pack("<Q", value.size)); child.stdin.write(value.tobytes())
                child.stdin.flush()
            assert child.wait(timeout=30) == 0, (case, result_path.read_text() if result_path.exists() else "no result")
            response = result_path.read_text(encoding="utf-8")
            assert response == tokenizer.decode(generated, skip_special_tokens=False), (response, tokenizer.decode(generated))
            report = dict(case=case, prompt_tokens=ids.shape[1], graph_calls=calls, independent_tensor_checks=checks,
                          image_patch_metrics=patch_results, response=response, passed=True)
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            child.kill() if child.poll() is None else None
            child.wait(); stderr.close()
    (args.output / "report.json").write_text(json.dumps(dict(
        device_tested=False, omc_executed=False,
        scope="actual portable App C++ scheduler with CPU dequantized ONNX references; tensor transfer checks",
        package_config_sha256=hashlib.sha256((args.package / "config.json").read_bytes()).hexdigest(),
        embedding_slice_build_verified=sliced_backend is not None, system_prompt=args.system_prompt,
        cases=reports), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "package", "build", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--only-multi-image", action="store_true", help="Check multiple images, including both App <hw> tag orders")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--system-prompt", help="Use the App's exact system template during replay")
    parser.add_argument("--text-prompt", action="append", help="Repeatable text prompts, including saved device failures")
    parser.add_argument("--embedding-slice-build", type=Path,
                        help="Separate split_embeddings.py work directory; compare physical slices to original Gather")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    verify(args)
