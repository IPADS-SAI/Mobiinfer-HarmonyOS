import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiment import Experiment, parse_args
from package_runtime import package
from w4a16 import W4Experiment


class NativeConverterTest(unittest.TestCase):
    def test_native_arguments_and_exit_status_are_preserved(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from utils import mnn_converter
        with tempfile.TemporaryDirectory(prefix="mnn converter ") as temp:
            root = Path(temp)
            converter = root / "converter with spaces"
            converter.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                                 'print(json.dumps(sys.argv[1:]))\n'
                                 'sys.exit(int(os.environ.get("MOBI_TEST_CONVERT_EXIT", "0")))\n')
            converter.chmod(0o755)
            obj = mnn_converter.MNNConverter(SimpleNamespace(args=SimpleNamespace(mnnconvert=str(converter))))
            arguments = ["converter", "--modelFile", "input with spaces; literal.onnx"]
            original = sys.argv
            with mock.patch.object(mnn_converter, "EXPORT_LOG", str(root / "convert.log")):
                obj.convert(arguments)
                self.assertEqual(json.loads((root / "convert.log").read_text()), arguments[1:])
                self.assertIs(sys.argv, original)
                with mock.patch.dict(os.environ, {"MOBI_TEST_CONVERT_EXIT": "7"}):
                    with self.assertRaises(subprocess.CalledProcessError):
                        obj.convert(arguments)
                self.assertIs(sys.argv, original)


class CalibrationImagesTest(unittest.TestCase):
    def test_symlinked_heldout_is_excluded_without_reordering_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images"
            images.mkdir()
            heldout, a, z = (root / name for name in ("heldout.png", "a.png", "z.png"))
            for path in (heldout, a, z):
                path.write_bytes(path.name.encode())
            for name, target in (("00.png", heldout), ("01.png", z), ("02.png", a)):
                (images / name).symlink_to(target)
            args = SimpleNamespace(calibration_images=images, image=heldout, quality_images=[],
                                   calibration_texts=None, calibration_samples=4, prompt="Describe this image")
            rows = W4Experiment.calibration_rows(SimpleNamespace(args=args))
            self.assertEqual([row["image"] for row in rows if "image" in row], [str(z), str(a)])


class ExternalCompileStagingTest(unittest.TestCase):
    def test_only_compiler_paths_are_staged_and_weight_pairs_are_copied_back(self):
        from external_weights import compile_external
        from experiment import sha256
        with tempfile.TemporaryDirectory(prefix="mobi_compile- 路径.") as temp:
            root = Path(temp)
            graph = root / "graphs/model.onnx"
            graph.parent.mkdir()
            graph.write_bytes(b"native ONNX fixture")
            (graph.parent / "native.pb").write_bytes(b"paired native weights")
            params = root / "quant/quant_params_file"
            params.parent.mkdir()
            params.write_bytes(b"paired compression parameters")
            entry = {"name": "vision_full", "onnx": "graphs/model.onnx", "onnx_sha256": sha256(graph),
                     "external_data_sha256": {"native.pb": sha256(graph.parent / "native.pb")},
                     "inputs": [{"name": "patches", "shape": [1, 32]}],
                     "quantization": {"parameters": "quant/quant_params_file", "parameters_sha256": sha256(params)}}
            def compiler(cmd, cwd, env, stdout, stderr):
                flags = dict(part[2:].split("=", 1) for part in cmd if part.startswith("--"))
                model, output, parameter = map(Path, [flags["model"], flags["output"], flags["compress_conf"]])
                self.assertRegex(str(model.parent), r"^/[A-Za-z0-9_/]+$")
                self.assertEqual(model.read_bytes(), graph.read_bytes())
                self.assertEqual((model.parent / "native.pb").read_bytes(), (graph.parent / "native.pb").read_bytes())
                self.assertEqual(parameter.read_bytes(), params.read_bytes())
                (output / "external.omc").write_bytes(b"compiled graph")
                (output / "SubGraph_0.weight").write_bytes(b"compiled external weights")
                stdout.write("partition type NPU:1, CPU:0\nSaveCompiledModelToFile SUCCESS\nOMG generate offline model success\n")
                return SimpleNamespace(returncode=0)
            experiment = SimpleNamespace(output=root, manifest={"graphs": [entry]}, save=lambda: None,
                args=SimpleNamespace(ddk=root / "DDK", platform="kirin9030", compiler_work_dir=Path("/tmp")))
            with mock.patch("external_weights.subprocess.run", side_effect=compiler):
                compile_external(experiment)
            self.assertTrue(entry["pure_npu"])
            self.assertEqual((root / entry["omc"]).read_bytes(), b"compiled graph")
            self.assertEqual((root / entry["external_weights"][0]["file"]).read_bytes(), b"compiled external weights")


class VisionPackageTest(unittest.TestCase):
    def test_vision_only_packaging_does_not_require_or_claim_language_graphs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, build = root / "model", root / "build"
            model.mkdir(); build.mkdir()
            source = {"model_type": "qwen3_vl", "image_token_id": 31, "pad_token_id": 0,
                      "text_config": {"num_hidden_layers": 1, "hidden_size": 32, "num_key_value_heads": 2,
                                      "head_dim": 8, "rope_theta": 10000, "rope_scaling": {"mrope_section": [1, 1, 2]}},
                      "vision_config": {"depth": 1, "patch_size": 16, "spatial_merge_size": 2,
                                        "temporal_patch_size": 2, "hidden_size": 128, "num_heads": 4,
                                        "num_position_embeddings": 16, "deepstack_visual_indexes": [0]}}
            (model / "config.json").write_text(json.dumps(source))
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            (build / "vision.omc").write_bytes(b"fixture graph")
            (build / "SubGraph_0.weight").write_bytes(b"fixture weight")
            profile = {"precision": "w4a16", "group_size": 64, "weight_method": "gptq", "head_bit": 4,
                       "kv_dtype": "fp16", "context": 8, "prefill_tokens": 2, "layers_per_chunk": 1,
                       "vision_tokens": 4, "vision_layers_per_chunk": 1, "fused_vision": True,
                       "source_config_sha256": digest(model / "config.json"), "platform": "kirin9030"}
            graph = {"name": "vision_full", "omc": "vision.omc", "omc_sha256": digest(build / "vision.omc"),
                     "pure_npu": True, "inputs": [], "output_info": [], "layer_start": 0, "layer_end": 1,
                     "deepstack_layers": [0], "external_weights": [{"file": "SubGraph_0.weight", "bytes": 14,
                                                                   "sha256": digest(build / "SubGraph_0.weight")}]}
            (build / "manifest.json").write_text(json.dumps({"profile": profile, "graphs": [graph], "cache": {}}))
            token = root / "tokenizer.mtok"
            token.write_bytes(b"fixture tokenizer")
            # Default full-NPU packaging retains its completeness check.
            with self.assertRaisesRegex(ValueError, "complete VLM"):
                package(build, model, token, root / "full", 32, 32, 16)
            package(build, model, token, root / "vision", 32, 32, 16, vision_only=True)
            cfg = json.loads((root / "vision/config.json").read_text())
            self.assertEqual(cfg["format"], "kirin_fused_vision_staging_v1")
            self.assertEqual([g["name"] for g in cfg["graphs"]], ["vision_full"])
            self.assertEqual(cfg["quantization"]["weight_method"], "gptq")
            release = json.loads((root / "vision/release_manifest.json").read_text())
            for record in release["files"]:
                self.assertEqual(digest(root / "vision" / record["path"]), record["sha256"])


@unittest.skipUnless(os.environ.get("KIRIN_TEST_DDK"), "Official SDK is required")
class VisionCalibrationTest(unittest.TestCase):
    def test_actual_sdk_calibration_leaves_cpu_language_floating(self):
        from transformers import Qwen3VLConfig, Qwen3VLVisionConfig, Qwen3VLTextConfig, Qwen3VLForConditionalGeneration
        torch.set_num_threads(2)
        torch.manual_seed(17)
        text = Qwen3VLTextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                                num_attention_heads=4, num_key_value_heads=4, head_dim=32, vocab_size=512,
                                rope_parameters={"rope_type": "default", "rope_theta": 10000,
                                                 "mrope_section": [4, 6, 6]})
        vision = Qwen3VLVisionConfig(hidden_size=128, intermediate_size=256, num_heads=4, depth=1,
                                    patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
                                    out_hidden_size=128, num_position_embeddings=16, deepstack_visual_indexes=[0])
        cfg = Qwen3VLConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                           image_token_id=511, vision_start_token_id=510, vision_end_token_id=509, pad_token_id=0)
        cfg._attn_implementation = "eager"
        model = Qwen3VLForConditionalGeneration(cfg).eval()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg.save_pretrained(root / "model")
            images = root / "images"
            images.mkdir()
            (images / "fixture.png").write_bytes(b"fixture image; image_inputs is isolated below")
            args = parse_args(["--model", str(root / "model"), "--output", str(root / "build"),
                               "--ddk", os.environ["KIRIN_TEST_DDK"], "--precision", "w4a16",
                               "--calibration-components", "vision", "--components", "vision",
                               "--calibration-images", str(images), "--calibration-samples", "2",
                               "--calibration-decode-steps", "1", "--context", "64", "--prefill-tokens", "4",
                               "--vision-tokens", "4", "--image-height", "32", "--image-width", "32",
                               "--steps", "calibrate"])
            image_inputs = {"input_ids": torch.tensor([[510, 511, 1]]),
                            "pixel_values": torch.randn(4, 1536), "image_grid_thw": torch.tensor([[1, 2, 2]]),
                            "mm_token_type_ids": torch.tensor([[0, 1, 0]])}
            tokenizer = mock.Mock()
            tokenizer.apply_chat_template.return_value = "fixture text"
            tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
            experiment = W4Experiment(args)
            with mock.patch.object(Experiment, "load_model", return_value=model), \
                    mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                    mock.patch("w4a16.image_inputs", return_value=image_inputs):
                experiment.calibrate()
            strategy = json.loads((root / "build/quantization/config.json").read_text())["layer_strategy"]
            self.assertTrue(any(v["quant_strategy"] != "float" for k, v in strategy.items() if k.startswith("model.visual.")))
            self.assertTrue(all(v["quant_strategy"] == "float" for k, v in strategy.items() if not k.startswith("model.visual.")))
            self.assertEqual(experiment.manifest["calibration"]["components"], "vision")
            self.assertTrue((root / "build/quantization/calibrated.pth").is_file())
            experiment.args.components = {"llm"}
            with self.assertRaisesRegex(ValueError, "Vision-only calibration"):
                experiment.export()


if __name__ == "__main__":
    unittest.main()
