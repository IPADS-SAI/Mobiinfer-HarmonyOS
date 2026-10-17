"""Validate pipeline inputs, shell argument plumbing and integrity-based resume without ML dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("model_pipeline", SCRIPTS / "model_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class ModelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mobi-model-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "original model"
        self.gptq = self.root / "GPTQ model"
        self.images = self.root / "calibration images"
        self.image = self.root / "heldout.png"
        cfg = {"model_type": "qwen3_vl", "image_token_id": 100, "vision_start_token_id": 101,
               "text_config": {"hidden_size": 2048, "num_hidden_layers": 28, "num_key_value_heads": 8,
                               "head_dim": 128, "vocab_size": 128},
               "vision_config": {"depth": 24, "patch_size": 16, "spatial_merge_size": 2,
                                 "temporal_patch_size": 2, "deepstack_visual_indexes": [5, 11, 17]}}
        for root in (self.model, self.gptq):
            root.mkdir()
            (root / "config.json").write_text(json.dumps(cfg))
            (root / "model.safetensors").write_bytes(b"weight fixture")
            (root / "tokenizer.json").write_text('{}')
        (self.gptq / "quantization_config.json").write_text(json.dumps({"bits": 8, "group_size": 128, "sym": True}))
        self.images.mkdir()
        (self.images / "01.png").write_bytes(b"image fixture 1")
        (self.images / "02.png").write_bytes(b"image fixture 2")
        self.image.write_bytes(b"held-out fixture")
        self.env = dict(os.environ)
        for key in ("MOBI_MODEL_CONFIG", "MOBI_SDK_CONFIG", "MOBI_HF_MODEL_ID", "MOBI_GPTQ_MODEL_ID",
                    "MOBI_HF_MODEL_REVISION", "MOBI_GPTQ_MODEL_REVISION"):
            self.env.pop(key, None)
        self.env.update(MOBI_ROOT=str(SCRIPTS.parents[1]), MOBI_PYTHON=sys.executable,
                        MOBI_MODEL_PYTHON=sys.executable, MOBI_QUANT_PYTHON=sys.executable,
                        MOBI_HF_MODEL=str(self.model), MOBI_GPTQ_MODEL=str(self.gptq),
                        MOBI_CALIB_IMAGES=str(self.images), MOBI_VALIDATION_IMAGE=str(self.image),
                        MOBI_CALIB_TEXTS="", MOBI_MODEL_WORK=str(self.root / "work"),
                        MOBI_HOST_BUILD=str(self.root / "work/host"), MOBI_MNNCONVERT=str(self.root / "work/host/MNNConvert"),
                        MOBI_MNN_EXPORT=str(self.root / "work/mnn"), MOBI_VISION_BUILD=str(self.root / "work/vision"),
                        MOBI_MODEL_PACKAGE=str(self.root / "work/package"), KIRIN_DDK=str(self.root / "ddk"),
                        MOBI_CALIB_SAMPLES="4", MOBI_CALIB_DECODE_STEPS="4", MOBI_MODEL_THREADS="2",
                        MOBI_VISION_WEIGHT_METHOD="gptq", MOBI_QUANT_DEVICE="cuda:0", MOBI_REUSE_MNN="", MOBI_FROZEN_W4="")
        self.env["MOBI_COMPILER_WORK_DIR"] = "/tmp"
        self.env["MOBI_MODEL_INPUT_ROOT"] = str(self.root / "downloads")
        for name in ("tools/tools_omg/omg", "tools/tools_omg/master/omg",
                     "tools/tools_ascendc/set_ascendc_env.sh", "tools/tools_dopt/dopt_pytorch_py3/dopt_so.py",
                     "tools/platform/kirin9030/config/kirin9030.ini"):
            target = self.root / "ddk" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"SDK fixture")

    def command(self, *args, success=True):
        result = subprocess.run(["bash", str(SCRIPTS / "build_model.sh"), *args], env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=self.root)
        self.assertEqual(result.returncode == 0, success, result.stdout)
        return result.stdout

    def test_dry_run_describes_full_pipeline_without_writing(self):
        output = self.command("--dry-run")
        for value in ("--gptq_path", "--visual_gptq_path", "--calibration-components vision",
                      "--calibration-images", "--vision-only", "--with-mnn-vision", "verify_hybrid_vision.py",
                      "--steps prepare,export", "--steps compile"):
            self.assertIn(value, output)
        self.assertFalse((self.root / "work").exists())
        line = next(line for line in output.splitlines() if "[mobi-model] calibrate:" in line)
        self.assertNotIn(str(self.gptq), line)
        self.assertIn("original model", line)

    def test_custom_texts_are_passed_as_one_argument(self):
        texts = self.root / "calibration text.json"
        texts.write_text('[{"text":"Describe this UI"}]')
        self.env["MOBI_CALIB_TEXTS"] = str(texts)
        output = self.command("--stages", "calibrate", "--dry-run")
        self.assertIn("--calibration-texts", output)
        self.assertIn(str(texts), output)

    def test_unconfigured_models_plan_default_modelscope_download_without_writing(self):
        self.env.pop("MOBI_HF_MODEL")
        self.env.pop("MOBI_GPTQ_MODEL")
        output = self.command("--dry-run")
        for name in ("YYDH21/mobi0402_2B_halfimage_rl@master", "YYDH21/mobi0402_2B_halfimage_rl-w8g128@master"):
            self.assertIn("Would download " + name, output)
        self.assertIn("validation will run after download", output)
        self.assertFalse((self.root / "downloads").exists())
        self.assertFalse((self.root / "work").exists())

    def test_invalid_custom_local_model_is_not_replaced_by_default_mobi(self):
        self.env["MOBI_HF_MODEL"] = str(self.root / "custom model which is absent")
        output = self.command("--dry-run", success=False)
        self.assertIn("Local original model is missing", output)
        self.assertNotIn("Would download", output)

    def test_missing_sdk_fails_before_model_download(self):
        sdk_file = self.root / "ddk/tools/tools_omg/omg"
        sdk_file.unlink()
        with mock.patch.dict(os.environ, self.env, clear=True):
            obj = pipeline.Pipeline()
            with mock.patch.object(obj, "prepare_sources") as download:
                with self.assertRaisesRegex(ValueError, "SDK is missing or incomplete"):
                    obj.run(["compile"])
                download.assert_not_called()

    def test_download_bootstrap_runs_model_python_and_reuses_completed_sources(self):
        sdk = self.root / "fake SDK"
        sdk.mkdir()
        cfg = (self.model / "config.json").read_text()
        (sdk / "modelscope.py").write_text(
            'import json\nfrom pathlib import Path\n'
            'def snapshot_download(*, model_id, revision, local_dir):\n'
            '    p=Path(local_dir)\n'
            '    p.mkdir(parents=True,exist_ok=True)\n'
            '    (p/"config.json").write_text(' + repr(cfg) + ')\n'
            '    (p/"tokenizer.json").write_text("{}")\n'
            '    if model_id.endswith("GPTQ"):\n'
            '        (p/"quantization_config.json").write_text(\'{"bits":8,"group_size":128,"sym":true}\')\n'
            '    (p/"model.safetensors").write_bytes(b"weight fixture")\n'
            '    with (Path(__file__).parent/"calls.jsonl").open("a") as f:\n'
            '        f.write(json.dumps([model_id,revision,local_dir])+"\\n")\n'
            '    return local_dir\n')
        self.env.update(MOBI_HF_MODEL=str(self.root / "downloaded original"),
                        MOBI_GPTQ_MODEL=str(self.root / "downloaded GPTQ"),
                        MOBI_HF_MODEL_ID="owner/original", MOBI_GPTQ_MODEL_ID="owner/GPTQ",
                        PYTHONPATH=str(sdk))
        with mock.patch.dict(os.environ, self.env, clear=True):
            obj = pipeline.Pipeline()
            self.assertTrue(obj.prepare_sources())
            obj.preflight()
            with mock.patch("model_pipeline.subprocess.run") as download:
                self.assertTrue(obj.prepare_sources())
                download.assert_not_called()
        calls = [json.loads(line) for line in (sdk / "calls.jsonl").read_text().splitlines()]
        self.assertEqual([row[0] for row in calls], ["owner/original", "owner/GPTQ"])
        self.assertEqual([row[2] for row in calls], [self.env["MOBI_HF_MODEL"], self.env["MOBI_GPTQ_MODEL"]])

    def test_validation_content_is_excluded_even_if_renamed(self):
        (self.images / "renamed.png").write_bytes(self.image.read_bytes())
        self.assertIn("duplicated", self.command("--dry-run", success=False))

    def test_symlink_images_keep_calibration_filename_order_and_exclude_heldout(self):
        for path in self.images.iterdir():
            path.unlink()
        a, z = self.root / "a.png", self.root / "z.png"
        a.write_bytes(b"image a")
        z.write_bytes(b"image z")
        (self.images / "00.png").symlink_to(self.image)
        (self.images / "01.png").symlink_to(z)
        (self.images / "02.png").symlink_to(a)
        with mock.patch.dict(os.environ, self.env, clear=True):
            obj = pipeline.Pipeline(dry_run=True)
            obj.preflight()
            self.assertEqual(obj.selected_images, [z, a])

    def test_unsupported_gptq_format_is_not_silently_requantized(self):
        (self.gptq / "quantization_config.json").write_text('{"bits":4,"group_size":64,"sym":true}')
        self.assertIn("W8/group128", self.command("--dry-run", success=False))

    def test_input_cannot_be_a_build_output(self):
        self.env["MOBI_MODEL_WORK"] = str(self.model)
        self.assertIn("overlaps", self.command("--dry-run", success=False))

    def test_runtime_only_mnn_reuse_is_rejected_before_packaging(self):
        self.env["MOBI_REUSE_MNN"] = str(self.root / "runtime package")
        self.assertIn("complete MNN export", self.command("--dry-run", success=False))
        self.assertFalse((self.root / "work").exists())

    def test_resume_verifies_completed_artifacts_and_source_weights(self):
        artifact = self.root / "result.bin"
        artifact.write_bytes(b"completed result")
        with mock.patch.dict(os.environ, self.env, clear=True):
            obj = pipeline.Pipeline()
            with mock.patch.object(obj, "run_stage", return_value=[artifact]) as run:
                obj.run(["mnn"])
                run.assert_called_once()
            obj = pipeline.Pipeline()
            with mock.patch.object(obj, "run_stage") as run:
                obj.run(["mnn"])
                run.assert_not_called()
            artifact.write_bytes(b"changed result")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                pipeline.Pipeline().run(["mnn"])
            (self.gptq / "model.safetensors").write_bytes(b"new source weights")
            with self.assertRaisesRegex(ValueError, "settings changed"):
                pipeline.Pipeline().run(["mnn"])

    def test_failed_stage_is_not_marked_complete(self):
        with mock.patch.dict(os.environ, self.env, clear=True):
            obj = pipeline.Pipeline()
            with mock.patch.object(obj, "run_stage", side_effect=RuntimeError("compiler failed")):
                with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                    obj.run(["mnn"])
            state = json.loads((self.root / "work/pipeline.json").read_text())
            self.assertNotIn("mnn", state["completed"])


if __name__ == "__main__":
    unittest.main()
