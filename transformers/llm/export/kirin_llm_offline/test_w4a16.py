import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import torch

from experiment import ONNXGraphs, parse_args
from static_kv import StaticDecoderChunk, StaticHead, StaticVisionChunk, StaticVisionPost, StaticVisionFull, attention_mask
from w4a16 import W4Experiment, cache_quantized_weights


@unittest.skipUnless(os.environ.get("KIRIN_TEST_DDK"), "Set KIRIN_TEST_DDK to run official SDK quantization tests")
class W4Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.temporary = tempfile.TemporaryDirectory(prefix="kirin_w4_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        (source / "config.json").write_text(json.dumps({
            "model_type": "qwen3_vl", "text_config": {
                "num_hidden_layers": 1, "num_key_value_heads": 2, "head_dim": 8}}))
        self.args = parse_args(["--model", str(source), "--output", str(self.root / "result"),
                               "--precision", "w4a16", "--ddk", os.environ["KIRIN_TEST_DDK"], "--steps", "export"])
        self.experiment = W4Experiment(self.args)
        route, sdk = self.experiment.sdk()
        config = self.root / "quant.json"
        config.write_text(json.dumps({"layer_strategy": {"projection": {
            "type": str(torch.nn.Linear), "quant_strategy": "Quant_lm_head",
            "weight": {"bit": 4, "group_size": 64},
            "input": {"bit": 16, "unsigned_quant": False, "input_algo": "min_max"},
            "output": {"bit": 16, "per_channel": True, "input_algo": "min_max"}}}}))
        self.quantized = sdk.optimize_model(StaticHead(torch.nn.Identity(), torch.nn.Linear(128, 32)), str(config)).eval()
        sdk.set_quant_state(self.quantized, True, True)
        sdk.set_calibrate_state(self.quantized, True)
        with torch.no_grad():
            self.quantized(torch.randn(1, 8, 128))
            self.quantized(torch.randn(1, 1, 128) * 3)
        sdk.set_calibrate_state(self.quantized, False)
        self.reference = cache_quantized_weights(copy.deepcopy(self.quantized)).eval()

    def test_cached_forward_exactly_matches_official_fake_quant(self):
        for scale in [0.01, 1, 10]:
            value = torch.randn(1, 5, 128) * scale
            with torch.no_grad():
                torch.testing.assert_close(self.reference(value), self.quantized(value), rtol=0, atol=0)

    def test_native_pair_reference_loading_and_shared_parameters(self):
        class TinyFull(torch.nn.Module):
            def __init__(self, head):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.norm = head.norm
                self.model.visual = torch.nn.Module()
                self.lm_head = head.projection
        self.experiment.sdk_model = TinyFull(self.quantized)
        self.experiment.manifest["calibration"] = {"checkpoint_sha256": "test-calibration"}
        value = torch.randn(1, 1, 128)
        for name in ["head_prefill", "head_decode"]:
            self.experiment.export_graph(name, self.reference, [value], ["hidden"], ["logits"], component="head")
        first, second = self.experiment.manifest["graphs"]
        self.assertEqual(first["quantization"], second["quantization"])
        self.assertTrue(first["quantization"]["native_initializers_verified"])
        compiled_graph = onnx.load(str(self.args.output / first["onnx"]))
        reference_graph = onnx.load(str(self.args.output / first["reference"]["onnx"]))
        self.assertFalse(any(node.op_type == "Round" for node in compiled_graph.graph.node))
        self.assertTrue(any(node.op_type == "Round" for node in reference_graph.graph.node))
        actual = ONNXGraphs(self.args.output).run("head_prefill", [value.numpy()])[0]
        with torch.no_grad():
            np.testing.assert_allclose(actual, self.reference(value).numpy(), rtol=1e-4, atol=1e-4)

    def test_manual_attention_scopes_match_sdk_parameter_names(self):
        from transformers import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
        config = Qwen3VLTextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=64,
            rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [4, 4, 8]})
        config._attn_implementation = "eager"
        module = StaticDecoderChunk(Qwen3VLTextModel(config).eval().layers, config)
        path = self.root / "decoder_config.json"
        path.write_text(json.dumps({"layer_strategy": {name: {
            "type": str(torch.nn.Linear), "quant_strategy": "Quant_act_weight_eco",
            "weight": {"bit": 4, "group_size": 64},
            "input": {"bit": 16, "unsigned_quant": False, "input_algo": "min_max"},
            "output": {"bit": 16, "per_channel": True, "input_algo": "min_max"}}
            for name, child in module.named_modules() if type(child) is torch.nn.Linear}}))
        _route, sdk = self.experiment.sdk()
        module = sdk.optimize_model(module, str(path)).eval()
        sdk.set_quant_state(module, True, True)
        sdk.set_calibrate_state(module, True)
        values = [torch.randn(1, 1, 128), torch.ones(1, 1, 1, 32), torch.zeros(1, 1, 1, 32),
                  torch.from_numpy(attention_mask(16, 0, 1, 1)), torch.zeros(1, 1, 1, 128),
                  torch.zeros(1, 2, 16, 32), torch.zeros(1, 2, 16, 32)]
        with torch.no_grad():
            module(*values)
        sdk.set_calibrate_state(module, False)
        reference = cache_quantized_weights(copy.deepcopy(module))
        self.experiment.manifest["calibration"] = {"checkpoint_sha256": "test-decoder"}
        with mock.patch.object(self.experiment, "sdk_component", return_value=("llm_0", module)):
            self.experiment.export_graph("llm_decode_0", reference, values,
                ["hidden", "cos", "sin", "mask", "deepstack", "past_key", "past_value"],
                ["output", "key_delta", "value_delta"], component="llm", mode="decode", layer_start=0, layer_end=1)
        entry = self.experiment.manifest["graphs"][-1]
        names = {node.name for node in onnx.load(str(self.args.output / entry["onnx"])).graph.node}
        self.assertIn("layers.0.layer.self_attn.q_proj", names)
        self.assertIn("layers.0.layer.mlp.down_proj", names)

    def test_vision_flat_scopes_and_biased_merger_gemm(self):
        from transformers import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock, Qwen3VLVisionPatchMerger
        config = Qwen3VLVisionConfig(hidden_size=128, intermediate_size=256, num_heads=4,
                                     depth=1, out_hidden_size=128, spatial_merge_size=2)
        visual = torch.nn.Module()
        visual.merger = Qwen3VLVisionPatchMerger(config)
        visual.deepstack_merger_list = torch.nn.ModuleList([Qwen3VLVisionPatchMerger(config, True)])
        hidden = torch.randn(1, 4, 128)
        cases = [
            ("vision_blocks", StaticVisionChunk([Qwen3VLVisionBlock(config)], 0, [0]),
             [hidden, torch.ones(1, 1, 4, 32), torch.zeros(1, 1, 4, 32), torch.zeros(1, 1, 4, 4)],
             ["hidden", "cos", "sin", "mask"]),
            ("vision_post", StaticVisionPost(visual), [hidden, hidden * 2], ["hidden", "deepstack"]),
        ]
        _route, sdk = self.experiment.sdk()
        self.experiment.manifest["calibration"] = {"checkpoint_sha256": "test-vision"}
        for component, module, values, inputs in cases:
            path = self.root / (component + ".json")
            path.write_text(json.dumps({"layer_strategy": {name: {
                "type": str(torch.nn.Linear), "quant_strategy": "Quant_act_weight_eco",
                "weight": {"bit": 4, "group_size": 64},
                "input": {"bit": 16, "unsigned_quant": False, "input_algo": "min_max"},
                "output": {"bit": 16, "per_channel": True, "input_algo": "min_max"}}
                for name, child in module.named_modules() if type(child) is torch.nn.Linear}}))
            module = sdk.optimize_model(module.eval(), str(path)).eval()
            sdk.set_quant_state(module, True, True)
            sdk.set_calibrate_state(module, True)
            with torch.no_grad():
                module(*values)
            sdk.set_calibrate_state(module, False)
            reference = cache_quantized_weights(copy.deepcopy(module))
            with mock.patch.object(self.experiment, "sdk_component", return_value=(component, module)):
                self.experiment.export_graph(component, reference, values, inputs,
                                             ["output", "features"], component=component)
            entry = self.experiment.manifest["graphs"][-1]
            graph = onnx.load(str(self.args.output / entry["onnx"]))
            names = {node.name for node in graph.graph.node}
            bindings = entry["quantization"]["node_name_bindings"]
            self.assertEqual(len(bindings), 4)
            self.assertTrue(all(value["sdk_name"] in names for value in bindings))
            self.assertFalse(any(node.op_type == "Gemm" for node in graph.graph.node))
            onnx.checker.check_model(graph)

    def test_fused_vision_pairs_every_block_and_individual_deepstack_merger(self):
        from transformers import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        config = Qwen3VLVisionConfig(hidden_size=128, intermediate_size=256, num_heads=4, depth=3,
            out_hidden_size=128, patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
            num_position_embeddings=16, deepstack_visual_indexes=[0, 1, 2])
        config._attn_implementation = "eager"
        module = StaticVisionFull(Qwen3VLVisionModel(config).eval())
        path = self.root / "fused_quant.json"
        # The original patch projection remains floating point; the 4 block
        # matrices and 2 matrices in each separate merger stay W4.
        path.write_text(json.dumps({"layer_strategy": {name: {
            "type": str(torch.nn.Linear), "quant_strategy": "Quant_act_weight_eco",
            "weight": {"bit": 4, "group_size": 64},
            "input": {"bit": 16, "unsigned_quant": False, "input_algo": "min_max"},
            "output": {"bit": 16, "per_channel": True, "input_algo": "min_max"}}
            for name, child in module.named_modules() if type(child) is torch.nn.Linear}}))
        _route, sdk = self.experiment.sdk()
        module = sdk.optimize_model(module, str(path)).eval()
        values = [torch.randn(16, 24), torch.arange(16, dtype=torch.int32)[:, None].expand(-1, 4).clone(),
            torch.full((16, 4), .25), torch.ones(1, 1, 16, 32), torch.zeros(1, 1, 16, 32), torch.zeros(1, 1, 16, 16)]
        sdk.set_quant_state(module, True, True); sdk.set_calibrate_state(module, True)
        with torch.no_grad(): module(*values)
        sdk.set_calibrate_state(module, False)
        reference = cache_quantized_weights(copy.deepcopy(module))
        self.experiment.manifest["calibration"] = {"checkpoint_sha256": "test-fused-vision"}
        with mock.patch.object(self.experiment, "sdk_component", return_value=("vision_full", module)):
            self.experiment.export_graph("vision_full", reference, values,
                ["patches", "position_indices", "position_weights", "rope_cos", "rope_sin", "attention_mask"],
                ["image_embeddings", "deepstack_features"], component="vision_full", layer_start=0, layer_end=3,
                deepstack_layers=[0, 1, 2])
        entry = self.experiment.manifest["graphs"][-1]
        self.assertEqual(len(entry["quantization"]["node_name_bindings"]), 3 * 4 + 4 * 2)
        with torch.no_grad(): expected = reference(*values)
        actual = ONNXGraphs(self.args.output).run("vision_full", [v.numpy() for v in values])
        for gold, value in zip(expected, actual):
            np.testing.assert_allclose(value, gold.numpy(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
