"""KV codec, packed graph inputs, and complete static-session regressions."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from static_kv import (KVCache, KV_DTYPES, StaticDecoderChunk, StaticSession,
                       attention_mask, cache_memory, quantize_kv)


class QuantizedKVTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.config = Qwen3VLTextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128,
            rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [2, 2, 4]})
        self.config._attn_implementation = "eager"
        self.model = Qwen3VLTextModel(self.config).eval()

    def session(self, dtype):
        chunks = []
        for start, end in [(0, 2), (2, 3)]:
            module = StaticDecoderChunk(self.model.layers[start:end], self.config, dtype,
                                         kv_residual_tokens=4, kv_sink_tokens=2).eval()
            def callback(_decode, inputs, module=module):
                if dtype == "int8":
                    self.assertEqual(inputs[5].dtype, np.int8)
                    self.assertEqual(inputs[7].dtype, np.float32)
                with torch.no_grad():
                    return [value.numpy() for value in module(*[torch.from_numpy(value) for value in inputs])]
            chunks.append((start, end - start, callback))
        return StaticSession(chunks, self.config, 32, 4, kv_dtype=dtype,
                             kv_residual_tokens=4, kv_sink_tokens=2)

    def inputs(self, count, offset=0):
        hidden = torch.randn(1, count, self.config.hidden_size)
        base = torch.arange(offset, offset + count)
        positions = torch.stack((base, base // 2, base % 3))[:, None]
        cos, sin = self.model.rotary_emb(hidden, positions)
        return hidden.numpy(), cos[:, None].numpy(), sin[:, None].numpy()

    def test_int8_error_bound_and_scale_independence(self):
        rng = np.random.default_rng(42)
        value = rng.normal(size=(1, 2, 7, 16)).astype(np.float32)
        value[:, 0, 0] = 0
        value[:, 0, 1] *= 1e-30
        value[:, 1, 1] *= 1e25
        encoded, scale = quantize_kv(value)
        self.assertEqual(encoded.dtype, np.int8)
        self.assertEqual(scale.dtype, np.float32)
        self.assertEqual(scale.shape, (1, 2, 7, 1))
        self.assertTrue(np.isfinite(scale).all() and (scale > 0).all())
        self.assertEqual(scale[0, 0, 0, 0], 1)
        self.assertGreaterEqual(encoded.min(), -127)
        self.assertLessEqual(encoded.max(), 127)
        decoded = encoded.astype(np.float32) * scale
        error = np.abs(decoded.astype(np.float64) - value.astype(np.float64))
        self.assertTrue((error <= scale * 0.501 + 1e-37).all())
        individual, individual_scale = quantize_kv(value[:, 0:1, 2:3])
        np.testing.assert_array_equal(individual, encoded[:, 0:1, 2:3])
        np.testing.assert_array_equal(individual_scale, scale[:, 0:1, 2:3])

    def test_subhead_groups_preserve_small_channels_beside_outliers(self):
        value = np.random.default_rng(1).normal(size=(1, 2, 3, 128)).astype(np.float32)
        value[..., 0] = 10000
        block, scales = quantize_kv(value, 32)
        decoded = (block.reshape(-1, 32).astype(np.float32) * scales.reshape(-1, 1)).reshape(value.shape)
        whole, whole_scale = quantize_kv(value, 128)
        whole_decoded = whole.astype(np.float32) * whole_scale
        error = np.linalg.norm(decoded[..., 32:] - value[..., 32:]) / np.linalg.norm(value[..., 32:])
        whole_error = np.linalg.norm(whole_decoded[..., 32:] - value[..., 32:]) / np.linalg.norm(value[..., 32:])
        self.assertLess(error, 0.01)
        self.assertGreater(whole_error, 0.9)

    def test_append_tail_preserves_old_rows_and_resets_scales(self):
        cache = KVCache(1, 2, 16, 16, dtype=np.int8, residual_tokens=0, sink_tokens=0)
        first = np.random.default_rng(1).normal(size=(1, 2, 4, 16)).astype(np.float32)
        first[:, :, 3:] = np.nan  # padded rows are never encoded or committed
        cache.append_chunk(0, [first, first * 2], 3)
        cache.commit(3)
        old_data, old_scale = cache.key[0][:, :, :3].copy(), cache.key_scale[0][:, :, :3].copy()
        cache.append_chunk(0, [np.full((1, 2, 1, 16), 1e6, dtype=np.float32)] * 2, 1)
        cache.commit(1)
        np.testing.assert_array_equal(cache.key[0][:, :, :3], old_data)
        np.testing.assert_array_equal(cache.key_scale[0][:, :, :3], old_scale)
        self.assertEqual(cache.nbytes, cache_memory(1, 2, 16, 16, "int8", residual_tokens=0, sink_tokens=0)["total_bytes"])
        cache.reset()
        self.assertEqual(cache.length, 0)
        self.assertFalse(cache.key[0].any() or cache.value[0].any())
        np.testing.assert_array_equal(cache.key_scale[0], 1)
        np.testing.assert_array_equal(cache.value_scale[0], 1)

    def test_invalid_delta_rejected_before_any_chunk_write(self):
        cache = KVCache(1, 2, 16, 16, dtype=np.int8)
        good = np.ones((1, 2, 1, 16), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            cache.append_chunk(0, [good, good * np.nan], 1)
        self.assertFalse(cache.key[0].any())
        with self.assertRaisesRegex(ValueError, "overflows"):
            KVCache(1, 2, 16, 16, dtype=np.float16).append_chunk(0, [good * 1e10] * 2, 1)

    def test_multiblock_prefill_and_decode_against_float_cache(self):
        for dtype, tolerance in [("fp16", 0.001), ("int8", 0.015)]:
            baseline, candidate = self.session("fp32"), self.session(dtype)
            for count, offset in [(9, 0), (1, 9), (1, 10), (1, 11)]:
                inputs = self.inputs(count, offset)
                expected = baseline.forward(*inputs, decode=count == 1)
                actual = candidate.forward(*inputs, decode=count == 1)
                error = np.linalg.norm(expected - actual) / np.linalg.norm(expected)
                self.assertLess(error, tolerance)
            self.assertEqual(candidate.cache.length, 12)
            self.assertLess(candidate.cache.nbytes, baseline.cache.nbytes)
            for i in range(3):
                for expected, actual in zip(baseline.cache.dequantized(i), candidate.cache.dequantized(i)):
                    self.assertLess(np.linalg.norm(expected - actual) / np.linalg.norm(expected), tolerance)

    def test_failure_in_later_block_can_retry_with_dirty_int8_slots(self):
        candidate = self.session("int8")
        clean = self.session("int8")
        original = candidate.chunks[1]
        calls = []
        def fail_second(decode, inputs):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("late failure")
            return original[2](decode, inputs)
        candidate.chunks[1] = (*original[:2], fail_second)
        inputs = self.inputs(9)
        with self.assertRaisesRegex(RuntimeError, "late failure"):
            candidate.forward(*inputs)
        self.assertEqual(candidate.cache.length, 0)
        candidate.chunks[1] = original
        np.testing.assert_array_equal(candidate.forward(*inputs), clean.forward(*inputs))
        for a, b in zip(candidate.cache.inputs(0, 3), clean.cache.inputs(0, 3)):
            np.testing.assert_array_equal(a[:, :, :9], b[:, :, :9])

    def test_ring_rollback_restores_already_committed_recent_tokens(self):
        candidate, clean = self.session("int8"), self.session("int8")
        prefix = self.inputs(9)
        candidate.forward(*prefix)
        clean.forward(*prefix)
        old = [tuple(v.copy() for v in candidate.cache.dequantized(i)) for i in range(3)]
        original = candidate.chunks[1]
        calls = []
        def fail_second(decode, inputs):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("ring overwritten")
            return original[2](decode, inputs)
        candidate.chunks[1] = (*original[:2], fail_second)
        extension = self.inputs(7, 9)
        with self.assertRaisesRegex(RuntimeError, "ring overwritten"):
            candidate.forward(*extension)
        self.assertEqual(candidate.cache.length, 9)
        for i in range(3):
            for expected, actual in zip(old[i], candidate.cache.dequantized(i)):
                np.testing.assert_array_equal(expected, actual)
        candidate.chunks[1] = original
        np.testing.assert_array_equal(candidate.forward(*extension), clean.forward(*extension))

    def test_hybrid_mask_exposes_each_historical_position_exactly_once(self):
        for recent, sinks in [(4, 2), (0, 2), (4, 0)]:
            for length in [0, 1, 2, 3, 5, 9, 15]:
                mask = attention_mask(16, length, 1, 1, residual_tokens=recent, sink_tokens=sinks)[0, 0, 0]
                visible = list(np.flatnonzero(mask[:16] == 0))
                visible += list(np.flatnonzero(mask[16:16 + sinks] == 0))
                ring = np.full(recent, -1)
                for position in range(length):
                    if recent:
                        ring[position % recent] = position
                if recent:
                    visible += list(ring[np.flatnonzero(mask[16 + sinks:16 + sinks + recent] == 0)])
                self.assertEqual(sorted(visible), list(range(length)))
                self.assertEqual(mask[-1], 0)

    def test_actual_onnx_keeps_int8_inputs_and_dequantizes_in_graph(self):
        import onnxruntime as ort
        for tokens, recent, sinks in [(1, 0, 0), (4, 0, 0), (1, 4, 2), (4, 4, 2)]:
            module = StaticDecoderChunk(self.model.layers[:1], self.config, "int8", kv_group_size=8,
                                         kv_residual_tokens=recent, kv_sink_tokens=sinks).eval()
            cache = KVCache(1, 2, 16, 16, dtype=np.int8, group_size=8, residual_tokens=recent, sink_tokens=sinks)
            values = np.random.default_rng(2).normal(size=(1, 2, 5, 16)).astype(np.float32)
            cache.append_chunk(0, [values, values * 2], 5)
            cache.commit(5)
            hidden, cos, sin = self.inputs(tokens, 5)
            inputs = [hidden, cos, sin, attention_mask(16, 5, tokens, tokens, residual_tokens=recent, sink_tokens=sinks),
                      np.zeros((1, 1, tokens, 64), dtype=np.float32), *cache.inputs(0, 1)]
            names = ["hidden", "cos", "sin", "mask", "deepstack", "key", "value", "key_scale", "value_scale"]
            if sinks:
                names.extend(["sink_key", "sink_value"])
            if recent:
                names.extend(["recent_key", "recent_value"])
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.onnx"
                with torch.no_grad():
                    torch.onnx.export(module, tuple(torch.from_numpy(v) for v in inputs), str(path),
                        opset_version=12, dynamo=False, input_names=names, output_names=["output", "dk", "dv"])
                    expected = module(*[torch.from_numpy(v) for v in inputs])
                graph = onnx.load(str(path))
                types = {v.name: v.type.tensor_type.elem_type for v in graph.graph.input}
                self.assertEqual(types["key"], onnx.TensorProto.INT8)
                self.assertEqual(types["value"], onnx.TensorProto.INT8)
                self.assertEqual(types["key_scale"], onnx.TensorProto.FLOAT)
                self.assertTrue(any(v.op_type == "Cast" and v.input[0] == "key" for v in graph.graph.node))
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                actual = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(None, dict(zip(names, inputs)))
                for a, b in zip(expected, actual):
                    np.testing.assert_allclose(a.numpy(), b, rtol=2e-5, atol=2e-5)

    def test_real_model_cache_size_includes_all_scales(self):
        self.assertEqual(cache_memory(28, 8, 2048, 128, "fp32")["total_bytes"], 448 * 2**20)
        self.assertEqual(cache_memory(28, 8, 2048, 128, "fp16")["total_bytes"], 224 * 2**20)
        self.assertEqual(cache_memory(28, 8, 2048, 128, "int8", 128, 0, 0)["total_bytes"], int(115.5 * 2**20))
        self.assertEqual(cache_memory(28, 8, 2048, 128, "int8", 32, 0, 0)["total_bytes"], 126 * 2**20)
        self.assertEqual(cache_memory(28, 8, 2048, 128, "int8")["total_bytes"], int(140.4375 * 2**20))


if __name__ == "__main__":
    unittest.main()
