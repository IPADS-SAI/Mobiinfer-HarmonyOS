"""Regression coverage for the actual ONNX simplification/export boundary."""
import json
import locale
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np
import onnx
import torch

from experiment import Experiment, ONNXGraphs, parse_args, save_json, sha256


class ExportTest(unittest.TestCase):
    def test_graph_only_rebuild_reuses_verified_independent_frozen_encoding(self):
        from external_weights import prepare
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "source"; source.mkdir()
            (source / "config.json").write_text("{}")
            (source / "checkpoint.pth").write_bytes(b"frozen")
            directory = Path("quantization/llm_0")
            quant = source / directory / "quant_output"; quant.mkdir(parents=True)
            (quant / "quant_params_file").write_bytes(b"paired-parameters")
            (quant / "fake_quant_weight.pth").write_bytes(b"paired-indices")
            checkpoint = sha256(source / "checkpoint.pth")
            save_json(source / directory / "encoding.json", {"calibrated_sha256": checkpoint,
                "fake_quant_weight_sha256": sha256(quant / "fake_quant_weight.pth")})
            record = {"profile": {"precision": "w4a16", "head_bit": 4, "layers_per_chunk": 28,
                                  "output_range_multiplier": 4},
                "calibration": {"config": "config.json", "config_sha256": sha256(source / "config.json"),
                                "checkpoint": "checkpoint.pth", "checkpoint_sha256": checkpoint},
                "graphs": [{"name": "llm_prefill_0", "component": "llm", "quantization": {
                    "parameters": str(directory / "quant_output/quant_params_file"),
                    "parameters_sha256": sha256(quant / "quant_params_file")}}]}
            save_json(source / "manifest.json", record)
            def experiment(name, factor=4):
                return SimpleNamespace(output=root / name, manifest={"graphs": []}, save=lambda: None,
                    args=SimpleNamespace(components={"llm"}, layers_per_chunk=28,
                                         activation_range_multiplier=1, output_range_multiplier=factor))
            candidate = experiment("same_ranges"); prepare(source, candidate)
            copied = candidate.output / directory / "quant_output/quant_params_file"
            self.assertNotEqual(copied.stat().st_ino, (quant / "quant_params_file").stat().st_ino)
            copied.write_bytes(b"later-sdk-write")
            self.assertEqual((quant / "quant_params_file").read_bytes(), b"paired-parameters")
            changed = experiment("different_ranges", 2); prepare(source, changed)
            self.assertFalse((changed.output / directory).exists())
            (quant / "fake_quant_weight.pth").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "encoding pair changed"):
                prepare(source, experiment("tampered"))

    def test_failed_kv_acceptance_is_saved_before_command_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            save_json(source / "config.json", {"model_type": "qwen3_vl", "text_config": {
                "num_hidden_layers": 1, "num_key_value_heads": 2, "head_dim": 8}})
            args = parse_args(["--model", str(source), "--output", str(root / "build"), "--steps", "kv_validate"])
            experiment = Experiment(args)
            result = {"report": "kv_validation_torch.json", "passed": False, "cases": ["long_text"]}
            with mock.patch.object(experiment, "load_model", return_value=None), \
                    mock.patch("kv_validation.validate_cache", return_value=result):
                with self.assertRaisesRegex(ValueError, "acceptance criteria"):
                    experiment.kv_validate()
            manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["kv_validation"]["torch"]["passed"])

    def test_kv_profile_requires_new_build_and_reuse_copies_independent_weights(self):
        with tempfile.TemporaryDirectory(prefix="kirin_reuse_test_") as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            save_json(source / "config.json", {"model_type": "qwen3_vl", "text_config": {
                "num_hidden_layers": 1, "num_key_value_heads": 2, "head_dim": 8}})
            baseline_args = parse_args(["--model", str(source), "--output", str(root / "fp32"), "--steps", "export"])
            baseline = Experiment(baseline_args)
            directory = baseline.output / "quantization"
            directory.mkdir()
            (directory / "calibrated.pth").write_bytes(b"frozen-checkpoint")
            save_json(directory / "config.json", {"frozen": True})
            baseline.manifest["calibration"] = {"checkpoint": "quantization/calibrated.pth",
                "checkpoint_sha256": sha256(directory / "calibrated.pth"),
                "config": "quantization/config.json", "config_sha256": sha256(directory / "config.json")}
            graph = baseline.output / "graphs/embedding/model.onnx"
            graph.parent.mkdir(parents=True)
            graph.write_bytes(b"unchanged-graph")
            baseline.manifest["graphs"] = [{"name": "embedding", "component": "embedding",
                "onnx": "graphs/embedding/model.onnx", "onnx_sha256": sha256(graph), "external_data_sha256": {}},
                {"name": "llm_decode_0", "component": "llm"}]
            baseline.save()
            changed = parse_args(["--model", str(source), "--output", str(baseline.output),
                                  "--kv-dtype", "int8", "--steps", "export"])
            with self.assertRaisesRegex(ValueError, "profile changed"):
                Experiment(changed)
            args = parse_args(["--model", str(source), "--output", str(root / "int8"), "--kv-dtype", "int8",
                               "--reuse-build", str(baseline.output), "--steps", "reuse"])
            candidate = Experiment(args)
            candidate.reuse()
            self.assertEqual([v["name"] for v in candidate.manifest["graphs"]], ["embedding"])
            copied = candidate.output / "quantization/calibrated.pth"
            original = directory / "calibrated.pth"
            self.assertNotEqual(copied.stat().st_ino, original.stat().st_ino)
            copied.write_bytes(b"later-new-checkpoint")
            self.assertEqual(original.read_bytes(), b"frozen-checkpoint")
            self.assertEqual((candidate.output / "graphs/embedding/model.onnx").read_bytes(), b"unchanged-graph")

    def test_utf8_metadata_survives_vendor_locale_changes(self):
        previous = locale.setlocale(locale.LC_CTYPE)
        try:
            locale.setlocale(locale.LC_CTYPE, "C")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "report.json"
                save_json(path, {"text": "量化校准图片"})
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"text": "量化校准图片"})
        finally:
            locale.setlocale(locale.LC_CTYPE, previous)

    def test_reuse_preserves_external_graph_weight_pair_and_compile_log(self):
        with tempfile.TemporaryDirectory(prefix="kirin_external_reuse_") as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            save_json(model / "config.json", {"model_type": "qwen3_vl", "text_config": {
                "num_hidden_layers": 1, "num_key_value_heads": 2, "head_dim": 8}})
            baseline = Experiment(parse_args(["--model", str(model), "--output", str(root / "source"),
                                               "--steps", "export"]))
            graph = baseline.output / "graphs/embedding/model.onnx"
            graph.parent.mkdir(parents=True)
            graph.write_bytes(b"graph")
            artifact = graph.parent / "external/external.omc"
            artifact.parent.mkdir()
            artifact.write_bytes(b"compiled graph")
            weight = artifact.parent / "model.weight"
            weight.write_bytes(b"external weights")
            compile_log = graph.parent / "compile_external.log"
            compile_log.write_text("successful external compile")
            relative = lambda path: str(path.relative_to(baseline.output))
            baseline.manifest["graphs"] = [{"name": "embedding", "component": "embedding",
                "onnx": relative(graph), "onnx_sha256": sha256(graph), "external_data_sha256": {},
                "omc": relative(artifact), "omc_sha256": sha256(artifact),
                "external_weights": [{"file": relative(weight), "sha256": sha256(weight),
                                      "bytes": weight.stat().st_size}]}]
            baseline.save()

            def candidate(directory):
                return Experiment(parse_args(["--model", str(model), "--output", str(root / directory),
                    "--kv-dtype", "int8", "--reuse-build", str(baseline.output), "--steps", "reuse"]))

            reused = candidate("reused")
            reused.reuse()
            for path in (artifact, weight, compile_log):
                copied = reused.output / relative(path)
                self.assertEqual(copied.read_bytes(), path.read_bytes())
                self.assertNotEqual(copied.stat().st_ino, path.stat().st_ino)
            (reused.output / relative(weight)).write_bytes(b"independent copy")
            self.assertEqual(weight.read_bytes(), b"external weights")
            weight.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "external weight changed"):
                candidate("tampered").reuse()

    def test_external_weights_ir_and_integrity(self):
        with tempfile.TemporaryDirectory(prefix="kirin_export_test_") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps({
                "model_type": "qwen3_vl",
                "text_config": {"num_hidden_layers": 1, "num_key_value_heads": 2, "head_dim": 8},
            }))
            args = parse_args(["--model", str(source), "--output", str(root / "result"), "--steps", "export"])
            experiment = Experiment(args)
            torch.manual_seed(42)
            module = torch.nn.Linear(32, 32).eval()
            inputs = [torch.randn(1, 4, 32)]
            experiment.export_graph("tiny", module, inputs, ["hidden"], ["projected"], component="head")
            entry = experiment.manifest["graphs"][0]
            graph = onnx.load(str(args.output / entry["onnx"]), load_external_data=False)
            # ONNX 1.22 defaults to IR 13, while the installed ORT accepts
            # only <= 11. The opset-12 Torch exporter actually emits IR 7.
            self.assertEqual(graph.ir_version, 7)
            self.assertTrue(entry["external_data_sha256"])
            self.assertEqual(entry["output_info"], [{"name": "projected", "shape": [1, 4, 32], "dtype": "float32"}])
            with torch.no_grad():
                expected = module(*inputs).numpy()
            actual = ONNXGraphs(args.output).run("tiny", [inputs[0].numpy()])[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            external = args.output / Path(entry["onnx"]).parent / next(iter(entry["external_data_sha256"]))
            with external.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "graph changed"):
                ONNXGraphs(args.output).run("tiny", [inputs[0].numpy()])


if __name__ == "__main__":
    unittest.main()
