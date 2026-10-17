"""Isolate cache storage error using the same frozen weights and activations."""
import copy
import gc
import json

import numpy as np
import torch

from experiment import ONNXGraphs, metrics, save_json


def long_text_inputs(tokenizer, prompt, target, limit):
    """Repeat user content, preserving one complete chat template.

    Repeating already-templated token IDs would inject artificial role changes
    throughout the context. The returned length may differ slightly from target.
    """
    unit = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if unit < 1:
        raise ValueError("Long-prompt content must contain tokens")
    def encode(repeats):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt * repeats}],
                                             tokenize=False, add_generation_prompt=True)
        return dict(tokenizer(text, return_tensors="pt", add_special_tokens=False))
    count = max(1, target // max(1, unit))
    inputs = encode(count)
    while inputs["input_ids"].shape[1] < target:
        count += 1
        inputs = encode(count)
    while inputs["input_ids"].shape[1] > limit and count > 1:
        count -= 1
        inputs = encode(count)
    if inputs["input_ids"].shape[1] > limit:
        raise ValueError("Context cannot fit a complete long-prompt chat template")
    return inputs


@torch.no_grad()
def validate_cache(model, args):
    from transformers import AutoTokenizer
    from w4a16 import image_inputs, static_generate, static_prefill
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    baseline_args = copy.copy(args)
    baseline_args.kv_dtype = "fp32"
    backend = ONNXGraphs(args.output, args.threads) if args.validation_backend == "onnx" else None
    specifications = [("text", "请简单介绍一下你自己。", None),
                      ("arithmetic", "请计算 23 加 19，并说明结果。", None),
                      ("long_text", "用户可以在应用首页搜索商品、查看详情并加入购物车。", None)]
    images = list(dict.fromkeys(value for value in [args.image, *args.quality_images] if value))
    specifications += [(f"image_{i}", args.prompt, path) for i, path in enumerate(images)]
    cases = []
    for name, prompt, image in specifications:
        if image:
            inputs = image_inputs(model, tokenizer, image, args, prompt)
        else:
            text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 tokenize=False, add_generation_prompt=True)
            inputs = dict(tokenizer(text, return_tensors="pt", add_special_tokens=False))
            if name == "long_text":
                count = min(args.kv_long_tokens, args.context - args.decode_steps)
                if count < 1:
                    raise ValueError("Context cannot fit the requested decode steps")
                inputs = long_text_inputs(tokenizer, prompt, count, args.context - args.decode_steps)
        count = inputs["input_ids"].shape[1]
        continuation = args.decode_steps if name == "long_text" else max(args.decode_steps, args.kv_quality_tokens)
        if count + continuation > args.context:
            raise ValueError("KV validation prompt and continuation exceed cache capacity")
        reference, logits, delta = static_prefill(model, inputs, baseline_args)
        candidate, actual, candidate_delta = static_prefill(model, inputs, args, backend=backend)
        if delta != candidate_delta:
            raise ValueError("KV format changed M-RoPE positions")
        comparisons, expected_tokens, candidate_tokens, failures = [], [], [], []
        for step in range(args.decode_steps + 1):
            comparison = metrics(logits, actual)
            if comparison["relative_l2"] > args.kv_alignment_rtol:
                failures.append({"metric": "logits", "step": step, "relative_l2": comparison["relative_l2"]})
                save_json(args.output / ("kv_validation_failure_" + args.validation_backend + ".json"), {
                    "passed": False, "case": name, "step": step, "kv_dtype": args.kv_dtype,
                    "kv_group_size": args.kv_group_size, "comparison": comparison,
                    "previous_phases": comparisons, "reference_tokens": expected_tokens,
                    "candidate_tokens": candidate_tokens, "device_execution": False})
            comparisons.append({"phase": "prefill" if step == 0 else "decode", "step": step,
                                "logits": comparison})
            token = int(logits[0, -1].argmax())
            expected_tokens.append(token)
            candidate_tokens.append(int(actual[0, -1].argmax()))
            if step == args.decode_steps:
                break
            ids = np.array([[token]], dtype=np.int32)
            embedded = model.model.language_model.embed_tokens(torch.from_numpy(ids))
            position = torch.full((3, 1, 1), reference.cache.length + delta, dtype=torch.long)
            cos, sin = model.model.language_model.rotary_emb(embedded, position)
            raw = reference.forward(embedded.numpy(), cos[:, None].numpy(), sin[:, None].numpy(), decode=True)
            logits = model.lm_head(model.model.language_model.norm(torch.from_numpy(raw))).numpy()
            candidate_embedded = embedded.numpy() if backend is None else backend.embedding(ids, decode=True)
            raw = candidate.forward(candidate_embedded, cos[:, None].numpy(), sin[:, None].numpy(), decode=True)
            actual = (model.lm_head(model.model.language_model.norm(torch.from_numpy(raw))).numpy()
                      if backend is None else backend.run("lm_head", [raw])[0])
        kv_error = []
        for layer in range(model.config.text_config.num_hidden_layers):
            for expected, value in zip(reference.cache.dequantized(layer), candidate.cache.dequantized(layer)):
                kv_error.append(metrics(expected, value)["relative_l2"])
        if max(kv_error) > args.kv_alignment_rtol:
            failures.append({"metric": "cache", "relative_l2": max(kv_error)})
        equal = [a == b for a, b in zip(expected_tokens, candidate_tokens)]
        if args.strict_greedy and not all(equal):
            failures.append({"metric": "strict_greedy", "agreement": sum(equal) / len(equal)})
        result = {"name": name, "image": str(image) if image else None,
                  "prefill_tokens": count, "decode_steps": args.decode_steps,
                  "final_cache_tokens": candidate.cache.length,
                  "cache_dtype": args.kv_dtype, "baseline_cache_bytes": reference.cache.nbytes,
                  "kv_group_size": args.kv_group_size,
                  "passed": not failures, "failures": failures,
                  "candidate_cache_bytes": candidate.cache.nbytes,
                  "max_logits_relative_l2": max(v["logits"]["relative_l2"] for v in comparisons),
                  "kv_max_relative_l2": max(kv_error), "greedy_token_agreement": sum(equal) / len(equal),
                  "reference_tokens": expected_tokens, "candidate_tokens": candidate_tokens,
                  "numeric_comparison": "shared teacher-forced FP32-cache prefixes", "phases": comparisons}
        del reference, candidate
        gc.collect()
        print(json.dumps({"kv_numeric_case": name, "passed": not failures,
                          "max_logits_relative_l2": result["max_logits_relative_l2"],
                          "greedy_token_agreement": result["greedy_token_agreement"]}), flush=True)
        if args.kv_quality_tokens and name != "long_text":
            baseline_args.quality_tokens = args.kv_quality_tokens
            generation_args = copy.copy(args)
            generation_args.quality_tokens = args.kv_quality_tokens
            result["free_generation"] = {
                "max_new_tokens": args.kv_quality_tokens,
                "fp32_kv": tokenizer.decode(static_generate(model, inputs, baseline_args), skip_special_tokens=True),
                "candidate_" + args.kv_dtype + "_kv": tokenizer.decode(
                    static_generate(model, inputs, generation_args, backend=backend), skip_special_tokens=True),
            }
        cases.append(result)
        save_json(args.output / ("kv_validation_progress_" + args.validation_backend + ".json"),
                  {"complete": False, "cases": cases})
        print(json.dumps({key: value for key, value in result.items() if key != "phases"}, ensure_ascii=False), flush=True)
    filename = "kv_validation_" + args.validation_backend + ".json"
    report = {"reference": "same frozen model, static Torch, FP32 KV",
              "candidate": "static " + args.validation_backend + ", " + args.kv_dtype + " KV",
              "scope": "isolated KV storage effect" if backend is None else "KV storage plus ONNX backend effect",
              "weight_precision": args.precision, "relative_l2_tolerance": args.kv_alignment_rtol,
              "strict_greedy": args.strict_greedy, "device_execution": False, "cases": cases,
              "complete": True, "passed": all(case["passed"] for case in cases)}
    save_json(args.output / filename, report)
    return {"report": filename, "passed": report["passed"], "cases": [value["name"] for value in cases]}
