"""Compare static graph mathematics to the matching full HF reference VLM."""
import json

import numpy as np
import torch

from experiment import ONNXGraphs, chunk_ranges, metrics, save_json, torch_callback
from static_kv import (StaticDecoderChunk, StaticHead, StaticSession, StaticVisionChunk,
                       StaticVisionPost, StaticVisionPre, StaticVisionFull)


def compare(reference, actual, label, tolerance=2e-4):
    result = metrics(reference, actual)
    if result["relative_l2"] > tolerance:
        raise ValueError(label + " diverged from HF: " + str(result))
    return result


@torch.no_grad()
def static_image(model, inputs, args, backend=None):
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    visual = model.model.visual
    grid = inputs["image_grid_thw"]
    tokens = int(grid.prod(-1).sum())
    if len(grid) != 1 or tokens != args.vision_tokens:
        raise ValueError("Validation currently requires one image matching --vision-tokens")
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners, spatial_merge_size=visual.spatial_merge_size,
    )
    position_ids = get_vision_position_ids(grid, visual.spatial_merge_size)
    rotary = visual.rotary_pos_emb(position_ids).reshape(tokens, -1)
    rotary = torch.cat((rotary, rotary), dim=-1)
    cos, sin = rotary.cos()[None, None], rotary.sin()[None, None]
    if getattr(args, "fuse_vision", False):
        values = [inputs["pixel_values"], indices.int(), weights, cos, sin, torch.zeros(1, 1, tokens, tokens)]
        if backend is None:
            image, features = StaticVisionFull(visual)(*values)
        else:
            image, features = [torch.from_numpy(v) for v in backend.run("vision_full", [v.numpy() for v in values])]
        reference = model.model.get_image_features(inputs["pixel_values"], grid, return_dict=True)
        results = {"image_embeddings": compare(torch.cat(reference.pooler_output).numpy(), image[0].numpy(),
                                                "fused image embeddings", args.alignment_rtol),
                   "deepstack_features": compare(torch.stack(reference.deepstack_features).numpy(), features[:, 0].numpy(),
                                                  "fused visual DeepStack", args.alignment_rtol)}
        return image[0], list(features[:, 0]), results
    if backend is None:
        hidden = StaticVisionPre(visual)(inputs["pixel_values"], indices.int(), weights)
    else:
        hidden = torch.from_numpy(backend.run("vision_pre", [inputs["pixel_values"].numpy(),
                                                           indices.int().numpy(), weights.numpy()])[0])
    deepstack = []
    for index, (start, end) in enumerate(chunk_ranges(len(visual.blocks), args.vision_layers_per_chunk)):
        mask = torch.zeros(1, 1, tokens, tokens)
        if backend is None:
            outputs = StaticVisionChunk(visual.blocks[start:end], start, visual.deepstack_visual_indexes)(hidden, cos, sin, mask)
        else:
            outputs = [torch.from_numpy(value) for value in backend.run(f"vision_blocks_{index}",
                       [hidden.numpy(), cos.numpy(), sin.numpy(), mask.numpy()])]
        hidden = outputs[0]
        deepstack.extend(outputs[1:])
    if backend is None:
        image, features = StaticVisionPost(visual)(hidden, *deepstack)
    else:
        image, features = [torch.from_numpy(value) for value in backend.run("vision_post", [value.numpy()
                                                                                         for value in [hidden, *deepstack]])]
    reference = model.model.get_image_features(inputs["pixel_values"], grid, return_dict=True)
    results = {"image_embeddings": compare(torch.cat(reference.pooler_output).numpy(), image[0].numpy(), "image embeddings", args.alignment_rtol),
               "deepstack_features": compare(torch.stack(reference.deepstack_features).numpy(),
                                              features[:, 0].numpy(), "visual DeepStack", args.alignment_rtol)}
    return image[0], list(features[:, 0]), results


@torch.no_grad()
def validate_case(model, tokenizer, inputs, args, name, image=None, visual_features=None, backend=None):
    language = model.model.language_model
    config = model.config.text_config
    chunks = [(start, end - start, torch_callback(StaticDecoderChunk(language.layers[start:end], config,
                         args.kv_dtype, args.kv_group_size, args.kv_residual_tokens, args.kv_sink_tokens))
               if backend is None else backend.decoder_callback(index))
              for index, (start, end) in enumerate(chunk_ranges(len(language.layers), args.layers_per_chunk))]
    session = StaticSession(chunks, config, args.context, args.prefill_tokens,
                            kv_dtype=args.kv_dtype, kv_group_size=args.kv_group_size,
                            kv_residual_tokens=args.kv_residual_tokens, kv_sink_tokens=args.kv_sink_tokens)
    ids = inputs["input_ids"]
    embedded = (language.embed_tokens(ids).clone() if backend is None
                else torch.from_numpy(backend.embedding(ids.numpy())))
    visual_mask = ids == model.config.image_token_id
    dense = None
    if image is not None:
        embedded[visual_mask] = image
        dense = torch.zeros(len(visual_features), *embedded.shape)
        for i, features in enumerate(visual_features):
            dense[i][visual_mask] = features
    if image is not None:
        position_ids, delta = model.model.get_rope_index(
            input_ids=ids, image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs.get("attention_mask"), mm_token_type_ids=inputs.get("mm_token_type_ids"))
    else:
        position_ids = torch.arange(ids.shape[1])[None, None].expand(3, 1, -1)
        delta = torch.zeros(1, 1, dtype=torch.long)
    cos, sin = language.rotary_emb(embedded, position_ids)
    reference = model(**inputs, use_cache=True, output_hidden_states=True, logits_to_keep=1, return_dict=True)
    raw = session.forward(embedded.numpy(), cos[:, None].numpy(), sin[:, None].numpy(),
                          None if dense is None else dense.numpy())
    head = StaticHead(language.norm, model.lm_head)
    def project(hidden):
        return head(torch.from_numpy(hidden)).numpy() if backend is None else backend.run("lm_head", [hidden])[0]
    logits = project(raw[:, -1:])
    phases = [{"phase": "prefill", "tokens": int(ids.shape[1]),
               "blocks": int(np.ceil(ids.shape[1] / args.prefill_tokens)),
               "tail_real_tokens": int((ids.shape[1] - 1) % args.prefill_tokens + 1),
               "hidden": compare(reference.hidden_states[-1].numpy(), language.norm(torch.from_numpy(raw)).numpy(), name, args.alignment_rtol),
               "logits": compare(reference.logits.numpy(), logits, name + " logits", args.alignment_rtol)}]
    generated, agreements = [], []
    for step in range(args.decode_steps + 1):
        expected = int(reference.logits[0, -1].argmax())
        actual = int(logits[0, -1].argmax())
        agreements.append(actual == expected)
        if actual != expected and args.strict_greedy:
            raise ValueError(f"{name}: greedy token differs at step {step}: {actual} vs {expected}")
        generated.append(actual)
        if step == args.decode_steps:
            break
        token = torch.tensor([[expected]], dtype=torch.long)
        embedded = language.embed_tokens(token) if backend is None else torch.from_numpy(backend.embedding(token.numpy(), decode=True))
        position = ids.shape[1] + step + int(delta.item())
        positions = torch.full((3, 1, 1), position, dtype=torch.long)
        cos, sin = language.rotary_emb(embedded, positions)
        reference = model(input_ids=token, position_ids=positions, past_key_values=reference.past_key_values,
                          use_cache=True, output_hidden_states=True, logits_to_keep=1, return_dict=True)
        raw = session.forward(embedded.numpy(), cos[:, None].numpy(), sin[:, None].numpy(), decode=True)
        logits = project(raw)
        phases.append({"phase": "decode", "step": step,
                       "hidden": compare(reference.hidden_states[-1].numpy(), language.norm(torch.from_numpy(raw)).numpy(), name, args.alignment_rtol),
                       "logits": compare(reference.logits.numpy(), logits, name + " logits", args.alignment_rtol)})
    cache_error = []
    for i, layer in enumerate(reference.past_key_values.layers):
        key, value = session.cache.dequantized(i)
        cache_error += [compare(layer.keys.numpy(), key, "key cache", args.alignment_rtol),
                        compare(layer.values.numpy(), value, "value cache", args.alignment_rtol)]
    result = {"name": name, "phases": phases, "all_greedy_tokens_equal": all(agreements),
              "greedy_token_agreement": sum(agreements) / len(agreements),
              "generation_mode": "greedy" if all(agreements) else "argmax on shared teacher-forced reference prefixes",
              "token_ids": generated, "answer_prefix": tokenizer.decode(generated, skip_special_tokens=True),
              "kv_max_relative_l2": max(value["relative_l2"] for value in cache_error),
              "final_cache_tokens": session.cache.length, "kv_dtype": args.kv_dtype,
              "cache_bytes": session.cache.nbytes}
    print(json.dumps({key: value for key, value in result.items() if key != "phases"}, ensure_ascii=False), flush=True)
    return result


@torch.no_grad()
def validate_model(model, args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    backend = ONNXGraphs(args.output, args.threads) if args.validation_backend == "onnx" else None
    text = tokenizer.apply_chat_template([{"role": "user", "content": "请简单介绍一下你自己。"}],
                                         tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    cases = [validate_case(model, tokenizer, dict(inputs), args, "text", backend=backend)]
    visual_metrics = None
    if args.image:
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        # Image-only validation does not need AutoProcessor's video backend
        # (which would require torchvision). Use HF's original PIL processor
        # and the same image-placeholder expansion as Qwen3VLProcessor.
        processor = Qwen2VLImageProcessorPil.from_pretrained(args.model, local_files_only=True)
        image = Image.open(args.image).convert("RGB").resize((args.image_width, args.image_height))
        message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        pixels = dict(processor(images=[image], return_tensors="pt"))
        image_tokens = int(pixels["image_grid_thw"].prod(-1).sum()) // processor.merge_size ** 2
        text = text.replace("<|image_pad|>", "<|image_pad|>" * image_tokens)
        inputs = dict(tokenizer(text, return_tensors="pt", add_special_tokens=False))
        inputs["mm_token_type_ids"] = (inputs["input_ids"] == model.config.image_token_id).long()
        inputs.update(pixels)
        image_features, deepstack, visual_metrics = static_image(model, inputs, args, backend)
        cases.append(validate_case(model, tokenizer, inputs, args, "image", image_features, deepstack, backend))
    description = ("full HF architecture with calibrated SDK W4A16 fake quantization; eager attention"
                   if args.precision == "w4a16" else "original full HuggingFace Qwen3-VL float32 weights; eager attention")
    if getattr(args, "validation_tag", "") == "weight_only":
        description = "full HF architecture with calibrated W4 weights; activation requantization bypassed"
    report = {"reference": description,
              "candidate": f"static {args.validation_backend} graphs with fixed KV capacity and delta updates",
              "device_execution": False, "relative_l2_tolerance": args.alignment_rtol,
              "strict_greedy": args.strict_greedy, "kv_dtype": args.kv_dtype,
              "visual": visual_metrics, "cases": cases}
    tag = "_" + args.validation_tag if getattr(args, "validation_tag", "") else ""
    filename = "host_validation_" + args.validation_backend + tag + ".json"
    save_json(args.output / filename, report)
    return {"report": filename, "passed": True,
            "cases": [value["name"] for value in cases], "decode_steps": args.decode_steps}
