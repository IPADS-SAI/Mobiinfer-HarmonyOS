import unittest

import numpy as np
import torch
from transformers import Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLVisionModel

from static_kv import (KVCache, StaticDecoderChunk, StaticSession, StaticVisionChunk,
                       StaticVisionPost, StaticVisionPre, StaticVisionFull, attention_mask,
                       stable_rms_norm, stable_attention_scores)


class StaticKVTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.config = Qwen3VLTextConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            vocab_size=64, rms_norm_eps=1e-6,
            rope_parameters={"rope_type": "default", "rope_theta": 10000,
                             "mrope_section": [1, 1, 2]},
        )
        self.config._attn_implementation = "eager"
        self.model = Qwen3VLTextModel(self.config).eval()

    def session(self, capacity=16, width=4):
        chunks = []
        for start, end in [(0, 2), (2, 3)]:
            module = StaticDecoderChunk(self.model.layers[start:end], self.config).eval()
            def callback(_decode, inputs, module=module):
                with torch.no_grad():
                    return [tensor.numpy() for tensor in module(*[torch.from_numpy(v) for v in inputs])]
            chunks.append((start, end - start, callback))
        return StaticSession(chunks, self.config, capacity, width)

    def inputs(self, count, offset=0):
        hidden = torch.randn(1, count, self.config.hidden_size)
        # Nonuniform temporal/height/width coordinates catch treating M-RoPE
        # as a single text position, particularly across a prefill boundary.
        base = torch.arange(offset, offset + count)
        positions = torch.stack((base, base // 2, base % 3))[:, None]
        cos, sin = self.model.rotary_emb(hidden, positions)
        return hidden, positions, cos[:, None], sin[:, None]

    def compare(self, reference, actual):
        with torch.no_grad():
            normalized = self.model.norm(torch.from_numpy(actual))
        torch.testing.assert_close(normalized, reference.last_hidden_state, rtol=2e-5, atol=2e-5)

    def test_rms_norm_survives_native_fp16_residual_outliers(self):
        from types import SimpleNamespace
        torch.manual_seed(5)
        # The phone residual range exceeded 28k. A direct FP16 square is not
        # representable even though RMSNorm's final result is small and finite.
        x = (torch.randn(4, 2048) * 3000).half()
        x[:, 0] = 28768
        norm = SimpleNamespace(weight=torch.ones(2048, dtype=torch.float16), variance_epsilon=1e-6)
        self.assertFalse(torch.isfinite(x.square()).all())
        expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        actual = stable_rms_norm(x, norm)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), expected, rtol=.003, atol=.003)
        torch.testing.assert_close(stable_rms_norm(x.float(), norm).float(), expected, rtol=2e-6, atol=2e-6)

    def test_attention_prescaling_avoids_unscaled_fp16_dot_overflow(self):
        q = torch.full((1, 1, 1, 128), 32., dtype=torch.float16)
        k = torch.full((1, 1, 2, 128), 32., dtype=torch.float16)
        self.assertFalse(torch.isfinite(torch.matmul(q, k.transpose(-1, -2))).all())
        actual = stable_attention_scores(q, k, 128)
        expected = torch.matmul(q.float(), k.float().transpose(-1, -2)) / np.sqrt(128)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), expected, rtol=.002, atol=1)

    def test_chunked_prefill_tail_padding_and_repeated_decode_match_hf(self):
        session = self.session()
        hidden, positions, cos, sin = self.inputs(9)
        with torch.no_grad():
            reference = self.model(inputs_embeds=hidden, position_ids=positions, use_cache=True)
        actual = session.forward(hidden.numpy(), cos.numpy(), sin.numpy())
        self.compare(reference, actual)
        cache = reference.past_key_values
        for offset in range(9, 12):
            hidden, positions, cos, sin = self.inputs(1, offset)
            with torch.no_grad():
                reference = self.model(inputs_embeds=hidden, position_ids=positions,
                                       past_key_values=cache, use_cache=True)
            actual = session.forward(hidden.numpy(), cos.numpy(), sin.numpy(), decode=True)
            self.compare(reference, actual)
            self.assertEqual(session.cache.length, offset + 1)
            for layer in range(self.config.num_hidden_layers):
                np.testing.assert_allclose(session.cache.key[layer][:, :, :offset + 1],
                                           cache.layers[layer].keys.numpy(), rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(session.cache.value[layer][:, :, :offset + 1],
                                           cache.layers[layer].values.numpy(), rtol=2e-5, atol=2e-5)

    def test_deepstack_features_at_visual_positions_match_hf(self):
        session = self.session()
        hidden, positions, cos, sin = self.inputs(9)
        visual_mask = torch.tensor([[False, True, True, False, True, False, True, True, False]])
        visual_features = [torch.randn(int(visual_mask.sum()), self.config.hidden_size) for _ in range(3)]
        dense = torch.zeros(3, *hidden.shape)
        for layer, features in enumerate(visual_features):
            dense[layer][visual_mask] = features
        with torch.no_grad():
            reference = self.model(inputs_embeds=hidden, position_ids=positions, use_cache=True,
                                   visual_pos_masks=visual_mask, deepstack_visual_embeds=visual_features)
        actual = session.forward(hidden.numpy(), cos.numpy(), sin.numpy(), dense.numpy())
        self.compare(reference, actual)

    def test_mask_hides_unused_cache_and_padded_new_keys(self):
        mask = attention_mask(16, 5, 2, 4)[0, 0]
        np.testing.assert_array_equal(mask[:, :5], 0)
        np.testing.assert_array_equal(mask[:, 5:16], -10000)
        np.testing.assert_array_equal(mask[:, 18:], -10000)
        self.assertEqual(mask[0, 16], 0)
        self.assertEqual(mask[0, 17], -10000)
        self.assertEqual(mask[1, 17], 0)
        self.assertTrue(np.isfinite(mask).all())

    def test_capacity_and_decode_shape_are_rejected_before_cache_changes(self):
        session = self.session(capacity=8)
        hidden, _, cos, sin = self.inputs(9)
        with self.assertRaisesRegex(ValueError, "capacity"):
            session.forward(hidden.numpy(), cos.numpy(), sin.numpy())
        self.assertEqual(session.cache.length, 0)
        with self.assertRaisesRegex(ValueError, "one token"):
            session.forward(hidden.numpy(), cos.numpy(), sin.numpy(), decode=True)

    def test_failed_chunk_does_not_commit_length_and_can_be_retried(self):
        session = self.session()
        original = session.chunks[1]
        def fail(_decode, _inputs):
            raise RuntimeError("device failure")
        session.chunks[1] = (*original[:2], fail)
        hidden, positions, cos, sin = self.inputs(3)
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            session.forward(hidden.numpy(), cos.numpy(), sin.numpy())
        self.assertEqual(session.cache.length, 0)
        session.chunks[1] = original
        actual = session.forward(hidden.numpy(), cos.numpy(), sin.numpy())
        with torch.no_grad():
            reference = self.model(inputs_embeds=hidden, position_ids=positions, use_cache=True)
        self.compare(reference, actual)

    def test_reset_discards_previous_conversation(self):
        cache = KVCache(2, 2, 8, 4)
        cache.append_chunk(0, [np.ones((1, 2, 3, 4))] * 4, 3)
        cache.commit(3)
        cache.reset()
        self.assertEqual(cache.length, 0)
        self.assertTrue(all(not tensor.any() for tensor in cache.key + cache.value))

    def test_failure_in_later_prefill_block_rolls_back_complete_call(self):
        session = self.session()
        original = session.chunks[1]
        calls = []
        def fail_second(decode, inputs):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("second block failed")
            return original[2](decode, inputs)
        session.chunks[1] = (*original[:2], fail_second)
        hidden, positions, cos, sin = self.inputs(9)
        with self.assertRaisesRegex(RuntimeError, "second block"):
            session.forward(hidden.numpy(), cos.numpy(), sin.numpy())
        self.assertEqual(session.cache.length, 0)
        session.chunks[1] = original
        actual = session.forward(hidden.numpy(), cos.numpy(), sin.numpy())
        with torch.no_grad():
            reference = self.model(inputs_embeds=hidden, position_ids=positions, use_cache=True)
        self.compare(reference, actual)

    def test_chunks_must_cover_each_layer_once(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            StaticSession([(0, 2, None), (1, 2, None)], self.config, 16, 4)

    def test_static_vision_pre_blocks_post_match_original_hf_vision(self):
        from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
        config = Qwen3VLVisionConfig(depth=3, hidden_size=32, intermediate_size=64, num_heads=4,
                                    out_hidden_size=32, patch_size=2, temporal_patch_size=2,
                                    spatial_merge_size=2, num_position_embeddings=16,
                                    deepstack_visual_indexes=[0, 1, 2])
        config._attn_implementation = "eager"
        visual = Qwen3VLVisionModel(config).eval()
        patches = torch.randn(16, 24)
        grid = torch.tensor([[1, 4, 4]])
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=visual.num_grid_per_side, mode="bilinear",
            align_corners=True, spatial_merge_size=2)
        position_ids = get_vision_position_ids(grid, 2)
        with torch.no_grad():
            reference = visual(patches, grid_thw=grid, return_dict=True)
            hidden = StaticVisionPre(visual)(patches, indices.int(), weights)
            rotary = visual.rotary_pos_emb(position_ids).reshape(16, -1)
            rotary = torch.cat((rotary, rotary), -1)
            outputs = StaticVisionChunk(visual.blocks, 0, [0, 1, 2])(
                hidden, rotary.cos()[None, None], rotary.sin()[None, None], torch.zeros(1, 1, 16, 16))
            image, deepstack = StaticVisionPost(visual)(*outputs)
            fused_image, fused_deepstack = StaticVisionFull(visual)(patches, indices.int(), weights,
                rotary.cos()[None, None], rotary.sin()[None, None], torch.zeros(1, 1, 16, 16))
        torch.testing.assert_close(fused_image, image, rtol=0, atol=0)
        torch.testing.assert_close(fused_deepstack, deepstack, rtol=0, atol=0)
        torch.testing.assert_close(image[0], reference.pooler_output, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(deepstack[:, 0], torch.stack(reference.deepstack_features), rtol=2e-5, atol=2e-5)
        # A tap at the final block must remain present even when it aliases
        # the final hidden tensor. Merger count/order must never be truncated.
        for bad in ([2, 1, 0], [0, 0, 2], [-1, 1, 2], [0, 1, 3], [0, 1]):
            visual.deepstack_visual_indexes = bad
            with self.assertRaisesRegex(ValueError, "DeepStack"):
                StaticVisionFull(visual)


if __name__ == "__main__":
    unittest.main()
