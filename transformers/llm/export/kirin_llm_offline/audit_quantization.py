"""Held-out teacher-forced W4 audit and separate FP16-vs-FP32 KV checks.

CPU fake-quant/dequantized execution is an export reference, not OMC execution.
NLL values below describe this small fixed corpus, not a benchmark accuracy.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment import metrics, save_json, sha256
from external_weights import arguments, prepare
from kv_validation import validate_cache
from w4a16 import CachedQuantLinear, W4Experiment, image_inputs, static_generate


@torch.no_grad()
def audit(options):
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
    source = json.loads((options.source_build / "manifest.json").read_text(encoding="utf-8"))
    args = arguments(source, options.output, options.model, options.ddk,
                     layers=source["profile"]["layers_per_chunk"], kv_dtype="fp16",
                     activation_range_multiplier=options.activation_range_multiplier,
                     output_range_multiplier=options.output_range_multiplier)
    args.threads = options.threads
    args.decode_steps = 8
    args.kv_long_tokens = 1800
    args.kv_quality_tokens = 0
    args.strict_greedy = False
    args.quality_tokens = options.generation_tokens
    args.image = options.image
    args.validation_backend = "torch"
    torch.set_num_threads(args.threads)
    experiment = W4Experiment(args)
    prepare(options.source_build, experiment)
    quantized = experiment.load_model()
    original = Qwen3VLForConditionalGeneration.from_pretrained(
        options.model, dtype=torch.float32, attn_implementation="eager", local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    controls = [(q, q.bypass) for module in quantized.modules() if isinstance(module, CachedQuantLinear)
                for q in (module.input_quantizer, module.output_quantizer) if hasattr(q, "bypass")]
    if options.image_only:
        images = list(dict.fromkeys(p for p in [options.image, *options.quality_images] if p))
        if not images:
            raise ValueError("Image audit requires --image or --quality-images")
        calibration_hashes = {s.get("image_sha256") for s in source["calibration"]["samples"] if "image" in s}
        cases = []
        for path in images:
            if sha256(path) in calibration_hashes:
                raise ValueError("Audit image was used for calibration: " + str(path))
            prompt = "请描述这张图片的内容。"
            inputs = image_inputs(original, tokenizer, path, args, prompt)
            messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = text.replace("<|image_pad|>", "<|image_pad|>" * (args.vision_tokens // 4))
            inputs.update(tokenizer(text, add_special_tokens=False, return_tensors="pt"))
            inputs["mm_token_type_ids"] = (inputs["input_ids"] == original.config.image_token_id).long()
            reference = original(**inputs, logits_to_keep=1, return_dict=True).logits.float()
            candidate = quantized(**inputs, logits_to_keep=1, return_dict=True).logits.float()
            try:
                for q, _ in controls:
                    q.bypass = True
                weight_only = quantized(**inputs, logits_to_keep=1, return_dict=True).logits.float()
            finally:
                for q, bypass in controls:
                    q.bypass = bypass
            case = {"image": str(path), "image_sha256": sha256(path), "prompt": prompt,
                    "first_logits": metrics(reference.numpy(), candidate.numpy()),
                    "weight_only_first_logits": metrics(reference.numpy(), weight_only.numpy()),
                    "first_token_agrees": bool(reference.argmax(-1) == candidate.argmax(-1))}
            if not options.logits_only:
                generated = original.generate(**inputs, do_sample=False, max_new_tokens=options.generation_tokens,
                                              use_cache=True)[0, inputs["input_ids"].shape[1]:]
                quantized_tokens = static_generate(quantized, inputs, args)
                case.update(reference_answer=tokenizer.decode(generated, skip_special_tokens=True),
                            w4_fp16_kv_answer=tokenizer.decode(quantized_tokens, skip_special_tokens=True))
            cases.append(case)
            print(json.dumps(case, ensure_ascii=False), flush=True)
            save_json(options.output / ("image_logit_audit.json" if options.logits_only else "image_quantization_audit.json"), {
                "complete": len(cases) == len(images), "device_tested": False,
                "scope": "CPU reference; images excluded from calibration; no OMC accuracy claim",
                "activation_range_multiplier": options.activation_range_multiplier,
                "output_range_multiplier": options.output_range_multiplier, "cases": cases})
        return
    specifications = [
        ("arithmetic", "请只回答结果：8 乘以 7 等于多少？", "56"),
        ("translation", "把“桌上有一本蓝色的书。”翻译成英语。", "There is a blue book on the table."),
        ("retrieval", "已知暗号是“松树”。请只重复暗号，不要解释。", "松树"),
        ("explanation", "用一句话说明为什么白天能看见月亮。", "月亮反射太阳光，当它白天位于地平线上方时也能被看见。"),
        ("python", "写一个 Python 函数，返回列表中最大的数。", "def largest(values):\n    return max(values)"),
        ("arithmetic_unseen", "请只回答：9 乘以 6 的结果。", "54"),
        ("translation_unseen", "将“我们下周一在图书馆见面。”翻译成英语。", "We will meet at the library next Monday."),
        ("retrieval_unseen", "名单里甲对应红色、乙对应绿色。乙对应哪种颜色？只答颜色。", "绿色"),
        ("summary_unseen", "用一句话概括：小林忘带雨伞，下班时下雨，同事把备用伞借给他。", "小林下班遇雨，同事把备用伞借给了他。"),
        ("python_unseen", "写一个 Python 函数，把两个数字相加。", "def add(a, b):\n    return a + b"),
    ]
    cases = []
    for name, prompt, target in specifications:
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        labels = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        ids = torch.cat((prefix, labels[None]), dim=1)
        start = prefix.shape[1] - 1
        expected = original(input_ids=ids, use_cache=False, return_dict=True).logits[0, start:-1].float()
        quant = quantized(input_ids=ids, use_cache=False, return_dict=True).logits[0, start:-1].float()
        try:
            for q, _ in controls:
                q.bypass = True
            weight_only = quantized(input_ids=ids, use_cache=False, return_dict=True).logits[0, start:-1].float()
        finally:
            for q, bypass in controls:
                q.bypass = bypass
        ref_logp = F.log_softmax(expected, dim=-1)
        ref_nll = float(F.nll_loss(ref_logp, labels))
        scores = {}
        for label, values in (("weight_only", weight_only), ("sdk_w4a16", quant)):
            logp = F.log_softmax(values, dim=-1)
            nll = float(F.nll_loss(logp, labels))
            scores[label] = {**metrics(expected.numpy(), values.numpy()), "nll": nll,
                             "nll_delta": nll - ref_nll, "perplexity_ratio": math.exp(nll - ref_nll),
                             "teacher_forced_top1_agreement": float((expected.argmax(-1) == values.argmax(-1)).float().mean()),
                             "kl_reference_to_candidate": float((ref_logp.exp() * (ref_logp - logp)).sum(-1).mean())}
        hf_tokens = original.generate(input_ids=prefix, max_new_tokens=options.generation_tokens,
                                      do_sample=False, use_cache=True)[0, prefix.shape[1]:]
        candidate_tokens = static_generate(quantized, {"input_ids": prefix}, args)
        case = {"name": name, "prompt": prompt, "teacher_forced_target": target, "target_tokens": len(labels),
                "reference_nll": ref_nll, "scores": scores,
                "reference_answer": tokenizer.decode(hf_tokens, skip_special_tokens=True),
                "w4_fp16_kv_answer": tokenizer.decode(candidate_tokens, skip_special_tokens=True)}
        cases.append(case)
        print(json.dumps(case, ensure_ascii=False), flush=True)
        save_json(options.output / "quantization_audit.json", {"complete": False, "device_tested": False, "cases": cases})
    report = {"complete": True, "device_tested": False,
              "scope": "CPU dequantized/fake-quant reference; no native OMC accuracy claim",
              "corpus": "ten fixed prompts with system template; *_unseen were not used to choose the range guard",
              "source_manifest_sha256": sha256(options.source_build / "manifest.json"),
              "calibration_checkpoint_sha256": source["calibration"]["checkpoint_sha256"], "cases": cases}
    report["activation_range_multiplier"] = options.activation_range_multiplier
    report["output_range_multiplier"] = options.output_range_multiplier
    save_json(options.output / "quantization_audit.json", report)
    del original
    validate_cache(quantized, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-build", "model", "ddk", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--quality-images", type=Path, nargs="*", default=[])
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--logits-only", action="store_true", help="With --image-only, compare activation/weight effects without generation")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--activation-range-multiplier", type=float, default=1)
    parser.add_argument("--output-range-multiplier", type=float, default=1)
    options = parser.parse_args()
    audit(options)


if __name__ == "__main__":
    main()
