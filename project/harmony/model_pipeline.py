#!/usr/bin/env python3
"""Build the supported Mobi Qwen3-VL hybrid runtime from local or ModelScope inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from model_sources import source_ready

STAGES = ("tools", "mnn", "calibrate", "vision", "verify", "compile", "package")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def log(value):
    print("[mobi-model] " + value, flush=True)


class Pipeline:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        def path(name, optional=False):
            value = os.environ.get(name, "")
            if not value and optional:
                return None
            if not value:
                raise ValueError("Set {} in model_config.sh".format(name))
            return Path(value).expanduser().resolve()
        self.repo = path("MOBI_ROOT")
        self.model = path("MOBI_HF_MODEL")
        self.gptq = path("MOBI_GPTQ_MODEL")
        self.images = path("MOBI_CALIB_IMAGES", True)
        self.image = path("MOBI_VALIDATION_IMAGE")
        self.texts = path("MOBI_CALIB_TEXTS", True)
        self.work = path("MOBI_MODEL_WORK")
        self.host = path("MOBI_HOST_BUILD")
        self.converter = path("MOBI_MNNCONVERT")
        self.reuse_mnn = path("MOBI_REUSE_MNN", True)
        self.mnn = self.reuse_mnn or path("MOBI_MNN_EXPORT")
        self.reuse_frozen = path("MOBI_FROZEN_W4", True)
        self.frozen = self.reuse_frozen or self.work / "frozen_w4"
        self.vision = path("MOBI_VISION_BUILD")
        self.output = path("MOBI_MODEL_PACKAGE")
        self.staging = self.work / "vision_runtime"
        self.cpu_python = os.environ["MOBI_MODEL_PYTHON"]
        self.quant_python = os.environ["MOBI_QUANT_PYTHON"]
        self.samples = int(os.environ["MOBI_CALIB_SAMPLES"])
        self.decode = int(os.environ["MOBI_CALIB_DECODE_STEPS"])
        self.threads = int(os.environ["MOBI_MODEL_THREADS"])
        self.method = os.environ["MOBI_VISION_WEIGHT_METHOD"]
        self.device = os.environ["MOBI_QUANT_DEVICE"]
        self.export = self.repo / "transformers/llm/export"
        self.offline = self.export / "kirin_llm_offline"
        self.ddk = path("KIRIN_DDK")
        self.compiler_work = path("MOBI_COMPILER_WORK_DIR")
        self.state_path = self.work / "pipeline.json"
        self.state = {"completed": {}}

    def check_sdk(self, stages):
        if not set(stages) & {"calibrate", "vision", "verify", "compile"}:
            return
        required = ("tools/tools_omg/omg", "tools/tools_omg/master/omg",
                    "tools/tools_ascendc/set_ascendc_env.sh", "tools/tools_dopt/dopt_pytorch_py3/dopt_so.py",
                    "tools/platform/kirin9030/config/kirin9030.ini")
        missing = [str(self.ddk / name) for name in required if not (self.ddk / name).is_file()]
        if missing:
            raise ValueError("DDK/Kirin9030 SDK is missing or incomplete: {}. Configure sdk_config.sh and run "
                             "bash project/harmony/setup.sh offline-tools before building models. Missing: {}"
                             .format(self.ddk, ", ".join(missing)))

    def check_paths(self):
        # Check before any download so a bad output setting cannot alter source data.
        for generated in [self.work, self.output, self.vision, self.host] + ([] if self.reuse_mnn else [self.mnn]):
            for source in [self.model, self.gptq, self.ddk, self.image] + ([self.images] if self.images else []):
                if generated == source or generated in source.parents or source in generated.parents:
                    raise ValueError("Output overlaps an input directory: {} / {}".format(generated, source))

    def prepare_sources(self):
        ready = True
        for directory, prefix, kind in ((self.model, "MOBI_HF_MODEL", "original"),
                                        (self.gptq, "MOBI_GPTQ_MODEL", "gptq")):
            model_id = os.environ.get(prefix + "_ID", "")
            revision = os.environ.get(prefix + "_REVISION", "master")
            if source_ready(directory, model_id, revision, kind):
                log("Using local {} model: {}".format(kind, directory))
                continue
            if self.dry_run:
                log("Would download {}@{} -> {}".format(model_id, revision, directory))
                ready = False
                continue
            subprocess.run([self.cpu_python, str(Path(__file__).with_name("model_sources.py")),
                            "--directory", str(directory), "--model-id", model_id, "--revision", revision,
                            "--kind", kind], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        return ready

    def preflight(self):
        self.check_paths()
        cfg = json.loads((self.model / "config.json").read_text(encoding="utf-8"))
        if cfg.get("quantization_config") or (self.model / "quantization_config.json").exists():
            raise ValueError("MOBI_HF_MODEL must point to the original unquantized model")
        if not list(self.model.glob("*.safetensors")):
            raise ValueError("Original model directory must include its safetensors weight files")
        text, vision = cfg.get("text_config", {}), cfg.get("vision_config", {})
        if (cfg.get("model_type"), text.get("hidden_size"), text.get("num_hidden_layers"),
            text.get("num_key_value_heads"), text.get("head_dim"), vision.get("depth"),
            vision.get("patch_size"), vision.get("spatial_merge_size"), vision.get("temporal_patch_size"),
            vision.get("deepstack_visual_indexes")) != ("qwen3_vl", 2048, 28, 8, 128, 24, 16, 2, 2, [5, 11, 17]):
            raise ValueError("This pipeline supports the existing Mobi Qwen3-VL 2B architecture only")
        quant = json.loads((self.gptq / "quantization_config.json").read_text(encoding="utf-8"))
        if (quant.get("bits"), quant.get("group_size"), quant.get("sym")) != (8, 128, True):
            raise ValueError("The validated MNN import/audit requires symmetric W8/group128 GPTQ input")
        gc = json.loads((self.gptq / "config.json").read_text(encoding="utf-8"))
        if gc.get("text_config") is None or any(gc.get(k) != cfg.get(k) for k in ("image_token_id", "vision_start_token_id")):
            raise ValueError("GPTQ source and original model multimodal token IDs differ")
        if any(gc["text_config"].get(k) != text.get(k) for k in ("hidden_size", "num_hidden_layers", "vocab_size")):
            raise ValueError("GPTQ source does not match original language architecture")
        if not (self.gptq / "model.safetensors").is_file():
            raise ValueError("The current GPTQ audit requires single-file model.safetensors")
        if self.samples < 2 or self.decode < 1 or self.threads < 1 or self.method not in ("gptq", "rtn"):
            raise ValueError("Invalid calibration samples/decode/threads/weight method")
        if any(c not in "/_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in str(self.compiler_work)):
            raise ValueError("MOBI_COMPILER_WORK_DIR must use only ASCII letters, digits, slashes and underscores")
        if not self.image.is_file():
            raise ValueError("Provide a separate held-out MOBI_VALIDATION_IMAGE: {}".format(self.image))
        for token_file in ("tokenizer.json", "vocab.json", "merges.txt"):
            a, b = self.model / token_file, self.gptq / token_file
            if a.is_file() and b.is_file() and sha(a) != sha(b):
                raise ValueError("Source model/GPTQ tokenizer differs: " + token_file)
        self.image_hash = sha(self.image)
        self.selected_images = []
        if not self.reuse_frozen:
            if not self.images or not self.images.is_dir():
                raise ValueError("Provide MOBI_CALIB_IMAGES for new NPU calibration")
            images = [p.resolve() for p in sorted(self.images.iterdir())
                      if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                      and p.resolve() != self.image]
            for image in images:
                if sha(image) == self.image_hash:
                    raise ValueError("Validation image is duplicated in calibration data: {}".format(image))
            if not images:
                raise ValueError("No calibration images remain after excluding the validation image")
            self.selected_images = list(dict.fromkeys(images[:(self.samples + 1) // 2]))
            log("Calibration: {} rows ({} image rows, {} text rows); {} distinct image files".format(
                self.samples, (self.samples + 1) // 2, self.samples // 2, len(self.selected_images)))
        else:
            frozen = json.loads((self.frozen / "manifest.json").read_text(encoding="utf-8"))
            if (frozen["profile"].get("source_config_sha256") != sha(self.model / "config.json")
                    or Path(frozen["profile"]["source_model"]).resolve() != self.model):
                raise ValueError("Frozen checkpoint must belong to this same original model")
            for key in ("checkpoint", "config"):
                if sha(self.frozen / frozen["calibration"][key]) != frozen["calibration"][key + "_sha256"]:
                    raise ValueError("Frozen calibration changed: " + key)
            if self.image_hash in {s.get("image_sha256") for s in frozen["calibration"]["samples"]}:
                raise ValueError("Validation image was used by the frozen calibration")
        if self.texts:
            rows = json.loads(self.texts.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) or not r.get("text") for r in rows):
                raise ValueError("MOBI_CALIB_TEXTS must be a JSON list of {text: prompt}")
        if self.reuse_mnn:
            self.mnn_artifacts()

    def identity(self):
        paths = [self.image, self.model / "config.json", self.gptq / "quantization_config.json"]
        for root in (self.model, self.gptq):
            paths += sorted(root.glob("*.safetensors")) + sorted(root.glob("*.json"))
        paths += self.selected_images + ([self.texts] if self.texts else [])
        # Track implementation changes as well as weight/data changes before accepting resume.
        paths += [Path(__file__).resolve(), Path(__file__).with_name("model_sources.py"),
                  self.offline / "w4a16.py", self.offline / "external_weights.py",
                  self.offline / "package_runtime.py", self.offline / "verify_hybrid_vision.py",
                  self.export / "llmexport.py", self.export / "utils/mnn_converter.py"]
        if self.reuse_frozen:
            paths += [self.frozen / "manifest.json"]
        record = {"inputs": {str(p): sha(p) for p in sorted(set(paths))},
                  "settings": {k: os.environ.get(k) for k in (
                      "MOBI_HF_MODEL", "MOBI_GPTQ_MODEL", "MOBI_CALIB_IMAGES", "MOBI_CALIB_TEXTS",
                      "MOBI_VALIDATION_IMAGE", "MOBI_MODEL_WORK", "MOBI_MODEL_PACKAGE", "MOBI_MNN_EXPORT",
                      "MOBI_HOST_BUILD", "MOBI_MNNCONVERT", "MOBI_VISION_BUILD", "MOBI_MODEL_PYTHON",
                      "MOBI_QUANT_PYTHON", "MOBI_QUANT_DEVICE", "MOBI_VISION_WEIGHT_METHOD",
                      "MOBI_CALIB_SAMPLES", "MOBI_CALIB_DECODE_STEPS", "MOBI_MODEL_THREADS",
                      "MOBI_FROZEN_W4", "MOBI_REUSE_MNN", "MOBI_COMPILER_WORK_DIR",
                      "MOBI_HF_MODEL_ID", "MOBI_GPTQ_MODEL_ID", "MOBI_HF_MODEL_REVISION", "MOBI_GPTQ_MODEL_REVISION")},
                  "ddk": str(self.ddk)}
        return record

    def command(self, command, stage):
        command = list(map(str, command))
        log("{}: {}".format(stage, shlex.join(command)))
        if self.dry_run:
            return
        directory = self.work / "logs"
        directory.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env["LD_LIBRARY_PATH"] = str(self.converter.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        with (directory / (stage + ".log")).open("a", encoding="utf-8") as stream:
            stream.write(shlex.join(command) + "\n"); stream.flush()
            try:
                subprocess.run(command, cwd=self.work, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError("{} failed (exit {}); inspect {}".format(stage, error.returncode, stream.name)) from error

    def mnn_artifacts(self):
        names = ["config.json", "llm_config.json", "tokenizer.mtok", "llm.mnn", "llm.mnn.json",
                 "llm.mnn.weight", "embeddings_bf16.bin"]
        for name in ["visual_pre", "visual_post"] + ["visual_blocks_npu_" + str(i) for i in range(6)]:
            names += [name + ".mnn", name + ".mnn.weight"]
            if name.startswith("visual_blocks"):
                names.append(name + ".mnn.json")
        paths = [self.mnn / name for name in names]
        for path in paths:
            if not path.is_file():
                raise ValueError("Expected complete MNN export (including audit JSON), missing: {}".format(path))
        return paths

    def quant_args(self):
        return ["--model", self.model, "--ddk", self.ddk, "--precision", "w4a16", "--weight-method", self.method,
                "--group-size", "64", "--head-bit", "4", "--kv-dtype", "fp16", "--layers-per-chunk", "28",
                "--vision-tokens", "608", "--image-height", "608", "--image-width", "256",
                "--fuse-vision", "--components", "vision", "--image", self.image, "--threads", str(self.threads),
                "--compiler-work-dir", self.compiler_work]

    @staticmethod
    def verify_release(directory):
        manifest = json.loads((directory / "release_manifest.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            if sha(directory / item["path"]) != item["sha256"]:
                raise ValueError("Packaged file changed: " + str(directory / item["path"]))

    def stage_vision(self):
        cmd = [self.cpu_python, self.offline / "package_runtime.py", "--build", self.vision,
               "--model", self.model, "--tokenizer", self.mnn / "tokenizer.mtok", "--vision-only"]
        if self.dry_run:
            self.command(cmd + ["--output", self.staging], "vision_package")
            return
        if self.staging.exists():
            self.verify_release(self.staging)
            cfg = json.loads((self.staging / "config.json").read_text(encoding="utf-8"))
            source = json.loads((self.vision / "manifest.json").read_text(encoding="utf-8"))
            expected = next(g["omc_sha256"] for g in source["graphs"] if g["name"] == "vision_full")
            if (cfg["graphs"][0]["sha256"] != expected
                    or sha(self.staging / "tokenizer.mtok") != sha(self.mnn / "tokenizer.mtok")):
                raise ValueError("Staged vision graph/tokenizer changed; use a new work directory")
            return
        with tempfile.TemporaryDirectory(prefix=".vision-stage-", dir=self.work) as temp:
            dest = Path(temp) / "vision"
            self.command(cmd + ["--output", dest], "vision_package")
            dest.rename(self.staging)

    def require_vision_report(self):
        report = json.loads((self.vision / "hybrid_vision_validation.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.vision / "manifest.json").read_text(encoding="utf-8"))
        graph = next(g for g in manifest["graphs"] if g["name"] == "vision_full")
        expected = {"checkpoint_sha256": manifest["calibration"]["checkpoint_sha256"],
                    "reference_onnx_sha256": graph["reference"]["onnx_sha256"],
                    "reference_external_data_sha256": graph["reference"]["external_data_sha256"],
                    "native_onnx_sha256": graph["onnx_sha256"],
                    "native_external_data_sha256": graph["external_data_sha256"],
                    "parameters_sha256": graph["quantization"]["parameters_sha256"],
                    "image_sha256": self.image_hash}
        if not report.get("passed") or any(report.get(k) != v for k, v in expected.items()):
            raise ValueError("CPU vision validation does not belong to these weights/graphs/image; rerun verify")

    def run_stage(self, stage):
        cpu, tools = self.cpu_python, self.offline
        if stage == "tools":
            if self.reuse_mnn or self.converter.is_file():
                log("Using existing MNN export/converter; host-tool build skipped")
                return []
            self.command(["cmake", "-S", self.repo, "-B", self.host, "-DCMAKE_BUILD_TYPE=Release",
                          "-DMNN_BUILD_CONVERTER=ON", "-DMNN_BUILD_TOOLS=ON", "-DMNN_BUILD_LLM=ON",
                          "-DMNN_BUILD_LLM_OMNI=ON", "-DMNN_LOW_MEMORY=ON", "-DMNN_SUPPORT_TRANSFORMER_FUSE=ON",
                          "-DMOBIINFER_MNN=OFF"], "tools_configure")
            self.command(["cmake", "--build", self.host, "--target", "MNNConvert", "llm_demo", "--parallel",
                          os.environ.get("MOBI_BUILD_JOBS", "8")], "tools_build")
            return [self.converter]
        if stage == "mnn":
            if not self.reuse_mnn:
                self.command([cpu, self.export / "llmexport.py", "--path", self.model, "--export", "mnn",
                              "--gptq_path", self.gptq, "--visual_gptq_path", self.gptq,
                              "--quant_bit", "8", "--quant_block", "128", "--visual_quant_bit", "8",
                              "--visual_quant_block", "128", "--lm_quant_bit", "16", "--seperate_embed",
                              "--visual_split", "--visual_npu_chunks", "6", "--visual_chunk_backends",
                              "npu,npu,npu,npu,npu,npu", "--mnnconvert", self.converter, "--dst_path", self.mnn], stage)
            else:
                log("Using explicitly configured MOBI_REUSE_MNN: {}".format(self.mnn))
            return [] if self.dry_run else self.mnn_artifacts()
        if stage == "calibrate":
            if self.reuse_frozen:
                log("Using explicitly configured frozen W4 checkpoint: {}".format(self.frozen))
            else:
                args = [self.quant_python, tools / "experiment.py", *self.quant_args(), "--output", self.frozen,
                        "--calibration-components", "vision", "--calibration-images", self.images,
                        "--calibration-samples", str(self.samples), "--calibration-decode-steps", str(self.decode),
                        "--quant-device", self.device, "--steps", "calibrate"]
                if self.texts:
                    args += ["--calibration-texts", self.texts]
                self.command(args, stage)
            return [self.frozen / "manifest.json", self.frozen / "quantization/config.json",
                    self.frozen / "quantization/calibrated.pth"]
        if stage in ("vision", "compile"):
            self.command([cpu, tools / "external_weights.py", "--source-build", self.frozen,
                          "--model", self.model, "--output", self.vision, "--ddk", self.ddk,
                          "--layers-per-chunk", "28", "--kv-dtype", "fp16", "--components", "vision", "--fuse-vision",
                          "--activation-range-multiplier", "1", "--output-range-multiplier", "4",
                          "--compiler-work-dir", self.compiler_work,
                          "--steps", "prepare,export" if stage == "vision" else "compile"], stage)
            if self.dry_run:
                return []
            manifest = json.loads((self.vision / "manifest.json").read_text(encoding="utf-8"))
            graph = next(g for g in manifest["graphs"] if g["name"] == "vision_full")
            paths = [self.vision / "manifest.json", self.vision / graph["onnx"],
                     self.vision / graph["reference"]["onnx"], self.vision / graph["quantization"]["parameters"]]
            for entry in [graph, graph["reference"]]:
                paths += [self.vision / Path(entry["onnx"]).parent / name for name in entry["external_data_sha256"]]
            if stage == "compile":
                paths += [self.vision / graph["omc"]] + [self.vision / w["file"] for w in graph["external_weights"]]
            return paths
        if stage == "verify":
            self.command([cpu, tools / "verify_hybrid_vision.py", "--build", self.vision, "--model", self.model,
                          "--ddk", self.ddk, "--image", self.image, "--threads", str(self.threads)], stage)
            return [self.vision / "hybrid_vision_validation.json"]
        if stage == "package":
            if not self.dry_run:
                self.require_vision_report()
            self.stage_vision()
            if self.dry_run:
                dest = self.output
                self.command([cpu, self.export / "kirin_hybrid_offline/package_runtime.py", "--cpu", self.mnn,
                              "--full-npu", self.staging, "--output", dest, "--with-mnn-vision"], stage)
                return []
            if self.output.exists():
                raise ValueError("Final package already exists; select a new MOBI_MODEL_PACKAGE")
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".mobi-model-", dir=self.output.parent) as temp:
                dest = Path(temp) / "package"
                self.command([cpu, self.export / "kirin_hybrid_offline/package_runtime.py", "--cpu", self.mnn,
                              "--full-npu", self.staging, "--output", dest, "--with-mnn-vision"], stage)
                self.command([cpu, self.export / "kirin_hybrid_offline/audit_gptq_weights.py", "--source", self.gptq,
                              "--exported", self.mnn, "--package", dest, "--output", dest / "validation/gptq_import.json"], "audit_gptq")
                shutil.copy2(self.vision / "hybrid_vision_validation.json", dest / "validation/hybrid_vision.json")
                save(dest / "build_provenance.json", {"input_identity": self.state["identity"],
                     "source_model": str(self.model), "gptq_source": str(self.gptq),
                     "vision_build": str(self.vision), "device_tested": False, "cpu_llm": "source W8/group128 GPTQ"})
                manifest_path = dest / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(vision_source=str(self.vision), vision_source_kind="static_build",
                                npu_vision_recalibrated=not bool(self.reuse_frozen), source_gptq_requantized=False)
                save(manifest_path, manifest)
                (dest / "README.md").write_text(
                    "# Mobi Qwen3-VL hybrid NPU model\n\n"
                    "Load config.json in ClawMate. Offline NPU: one external-weight W4A16 ViT and MNN CPU LLM.\n"
                    "Online NPU/CPU: complete MNN W8/group128 vision and the same CPU LLM.\n"
                    "Language/online vision import the supplied GPTQ weights; see build_provenance.json.\n"
                    "CPU reference and held-out visual checks are in validation/. They do not execute OMC, "
                    "measure device performance or establish end-to-end answer quality.\n", encoding="utf-8")
                self.command([cpu, "-c", "import sys;sys.path.insert(0,sys.argv[1]);from package_runtime import seal_release;"
                              "seal_release(__import__('pathlib').Path(sys.argv[2]))", tools, dest], "seal")
                dest.rename(self.output)
            return [self.output / "release_manifest.json", self.output / "config.json", self.output / "build_provenance.json"]
        raise ValueError("Unknown stage: " + stage)

    def run(self, stages):
        self.check_paths()
        self.check_sdk(stages)
        if self.prepare_sources():
            self.preflight()
        else:
            log("Model input validation will run after download; dry-run does not download files")
        if not self.dry_run:
            identity = self.identity()
            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
                if self.state.get("identity") != identity:
                    raise ValueError("Model/weights/calibration/settings changed; use a new MOBI_MODEL_WORK")
            else:
                self.state["identity"] = identity
                save(self.state_path, self.state)
            for stage in stages:
                if (stage == "calibrate" and not self.reuse_frozen and self.method == "gptq"
                        and stage not in self.state["completed"]):
                    subprocess.run([self.quant_python, "-c", "import torch;assert torch.cuda.is_available(),"
                                    "'Fresh GPTQ calibration requires CUDA Torch; set MOBI_QUANT_PYTHON'"], check=True)
        for stage in stages:
            previous = self.state["completed"].get(stage)
            if previous:
                for name, digest in previous.items():
                    if sha(name) != digest:
                        raise ValueError("Completed stage artifact changed: " + name)
                if stage == "package":
                    self.verify_release(self.output)
                log("{} already verified; skipping".format(stage))
                continue
            outputs = self.run_stage(stage)
            if not self.dry_run:
                self.state["completed"][stage] = {str(p): sha(p) for p in outputs}
                if stage == "compile" and "vision" in self.state["completed"]:
                    # Compilation appends OMC records to this otherwise unchanged export manifest.
                    self.state["completed"]["vision"][str(self.vision / "manifest.json")] = sha(self.vision / "manifest.json")
                save(self.state_path, self.state)
        if self.dry_run:
            log("Plan complete")
        elif "package" in stages:
            log("Stages complete; final model entry: {}".format(self.output / "config.json"))
        else:
            log("Selected stages complete; logs: {}".format(self.work / "logs"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default=",".join(STAGES), help=",".join(STAGES))
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without writing/building")
    args = parser.parse_args()
    stages = args.stages.split(",")
    if not stages or any(stage not in STAGES for stage in stages):
        parser.error("Unknown stages; choose from " + ",".join(STAGES))
    try:
        Pipeline(args.dry_run).run(stages)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, "[mobi-model] ERROR: {}\n".format(error))


if __name__ == "__main__":
    main()
