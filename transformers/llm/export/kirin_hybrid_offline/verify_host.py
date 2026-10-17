"""Numerical bridge validation: production C++ + HF inputs + CPU ONNX vision.

OMC/HiAI is not executed on this host. MNN language graph and weights are real.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLVisionModel, Qwen3VLVisionRotaryEmbedding
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'kirin_llm_offline'))
from experiment import ONNXGraphs


def verify(a):
    a.output.mkdir(parents=True, exist_ok=True)
    tag = '<img>' + str(a.image.resolve()) + '</img>'
    prompt = '<|im_start|>user\n请描述这张图片：' + tag*a.copies + '<|im_end|>\n<|im_start|>assistant\n'
    (a.output / 'prompt.txt').write_text(prompt)

    def driver(mode, allowed=(0,)):
        with (a.output / (mode + '.log')).open('w') as f:
            r = subprocess.run([str(a.driver), str(a.package / 'config.json'), str(a.output), mode], stdout=f, stderr=f)
        if r.returncode not in allowed:
            raise RuntimeError(f'C++ {mode} failed ({r.returncode}): {a.output / (mode + ".log")}')

    # First call captures production preprocessor inputs; absent feature outputs
    # deliberately stop before CPU language generation. This failure is expected.
    driver('prepare', (0, 4))
    cfg = AutoConfig.from_pretrained(a.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    grid = torch.tensor([[1, 38, 16]])
    with torch.device('meta'):
        vision = Qwen3VLVisionModel(cfg.vision_config)
        model = Qwen3VLModel(cfg)
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=vision.num_grid_per_side, mode=vision.interpolation_mode,
        align_corners=vision.interpolation_align_corners, spatial_merge_size=2)
    positions = get_vision_position_ids(grid, 2)
    rotary = Qwen3VLVisionRotaryEmbedding(32)
    angles = rotary(positions)
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().numpy().reshape(1, 1, 608, 64), angles.sin().numpy().reshape(1, 1, 608, 64)
    processor = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2,
                                        image_mean=[.5]*3, image_std=[.5]*3, do_resize=False)
    image = Image.open(a.image).convert('RGB').resize((256, 608), Image.Resampling.BICUBIC)
    patches = np.asarray(processor(images=image, return_tensors='np')['pixel_values'], dtype=np.float32)
    expected = [patches, indices.numpy().reshape(608, 4).astype(np.int32),
                weights.numpy().reshape(608, 4).astype(np.float32), cos, sin,
                np.zeros((1, 1, 608, 608), dtype=np.float32)]
    checks = {}
    for i, e in enumerate(expected):
        actual = np.fromfile(a.output / f'vision_input_{i}.bin', dtype=e.dtype).reshape(e.shape)
        # Same Pillow-compatible 8-bit resize; allow at most one RGB level only
        # for floating boundary roundoff. Constants/rotary are strictly aligned.
        tolerance = 2 / 255 + 1e-6 if i == 0 else 1e-6
        np.testing.assert_allclose(actual, e, atol=tolerance, rtol=0)
        checks[f'vision_input_{i}_max_abs'] = float(np.max(np.abs(actual - e)))
    # ONNX receives the actual production preprocessor bytes, not a different
    # host image pipeline, so CPU language checks isolate the boundary transfer.
    arrays = [np.fromfile(a.output / f'vision_input_{i}.bin', dtype=e.dtype).reshape(e.shape) for i, e in enumerate(expected)]
    build_hash = hashlib.sha256((a.build / 'manifest.json').read_bytes()).hexdigest()
    if a.vision_reference:
        saved = json.loads((a.vision_reference / 'validation.json').read_text())
        assert saved['build_manifest_sha256'] == build_hash
        for i, e in enumerate(arrays):
            assert (a.vision_reference / f'vision_input_{i}.bin').read_bytes() == e.tobytes()
        outputs = []
        for i, shape in enumerate(((1,152,2048),(3,1,152,2048))):
            raw = (a.vision_reference / f'feature_{i}.bin').read_bytes()
            assert hashlib.sha256(raw).hexdigest() == saved['visual_output_sha256'][i]
            outputs.append(np.frombuffer(raw, np.float32).reshape(shape))
    else:
        outputs = ONNXGraphs(a.build).run('vision_full', arrays)
    for i, f in enumerate(outputs):
        np.asarray(f, dtype=np.float32).tofile(a.output / f'feature_{i}.bin')
    driver('prepare')
    ids = np.fromfile(a.output / 'tokens.bin', np.int32)
    rendered = prompt.replace('<img>' + str(a.image.resolve()) + '</img>',
                              '<|vision_start|>' + '<|image_pad|>'*152 + '<|vision_end|>')
    hf_ids = tokenizer.encode(rendered, add_special_tokens=False)
    np.testing.assert_array_equal(ids, hf_ids)
    input_ids = torch.tensor([hf_ids])
    hf_pos, delta = model.get_rope_index(input_ids, mm_token_type_ids=(input_ids == cfg.image_token_id).int(), image_grid_thw=grid.repeat(a.copies,1), attention_mask=torch.ones_like(input_ids))
    hf_pos = hf_pos[:, 0].numpy().astype(np.int32)
    actual_pos = np.fromfile(a.output / 'positions.bin', np.int32).reshape(3, -1)
    np.testing.assert_array_equal(actual_pos, hf_pos)
    emb = np.fromfile(a.output / 'text_embeddings.bin', np.float32).reshape(len(ids), 2048)
    deep = np.zeros((3, len(ids), 2048), np.float32)
    where = np.flatnonzero(ids == cfg.image_token_id)
    emb[where] = np.tile(outputs[0].reshape(152,2048), (a.copies,1))
    deep[:, where] = np.tile(outputs[1].reshape(3,152,2048), (1,a.copies,1))
    actual_emb = np.fromfile(a.output / 'embeddings.bin', np.float32).reshape(emb.shape)
    np.testing.assert_array_equal(actual_emb, emb)
    emb.tofile(a.output / 'expected_embeddings.bin')
    deep.tofile(a.output / 'expected_deepstack.bin')
    hf_pos.tofile(a.output / 'expected_positions.bin')
    driver('reference')
    driver('run')
    actual = np.fromfile(a.output / 'logits.bin', np.float32)
    reference = np.fromfile(a.output / 'reference_logits.bin', np.float32)
    np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-5)
    generated = np.fromfile(a.output / 'generated.bin', np.int32)
    assert len(generated) == 1 and generated[0] == int(reference.reshape(-1, cfg.text_config.vocab_size)[-1].argmax())
    checks.update(tokens=len(ids), image_tokens=len(where), mrope='HF exact', embeddings='exact',
                  cpu_prefill_logits_max_abs=float(np.max(np.abs(actual-reference))),
                  cpu_prefill_logits_cosine=float(np.dot(actual, reference)/(np.linalg.norm(actual)*np.linalg.norm(reference))),
                  first_token=int(generated[0]), first_piece=tokenizer.decode(generated),
                  retry_and_decode='passed; see run.log', omc_executed=False)
    checks['build_manifest_sha256'] = build_hash
    checks['visual_output_sha256'] = [hashlib.sha256(np.asarray(f,dtype=np.float32).tobytes()).hexdigest() for f in outputs]
    (a.output / 'validation.json').write_text(json.dumps(checks, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('driver','package','build','model','image','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--copies',type=int,choices=(1,2),default=1,help='Repeat the same image to validate multiple image spans')
    p.add_argument('--vision-reference',type=Path,help='Reuse previously verified CPU vision outputs only after byte/hash checks')
    verify(p.parse_args())
