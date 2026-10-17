"""Rebuild frozen W4 graphs with one decoder chunk and external OMC weights.

The source calibration is immutable. A new build/release is required; this does
not rewrite the previously published embedded-weight model package.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from experiment import parse_args, save_json, sha256


def arguments(source, output, model, ddk, layers=None, kv_group=None, kv_dtype="fp16",
              activation_range_multiplier=1, output_range_multiplier=1, components=None, fuse_vision=False,
              compiler_work_dir=None):
    p = source["profile"]
    values = ["--model", str(model), "--output", str(output), "--ddk", str(ddk),
              "--precision", "w4a16", "--kv-dtype", kv_dtype, "--components",
              components or ("llm,head,vision" if activation_range_multiplier != 1 or output_range_multiplier != 1 else "llm"), "--steps", "export",
              "--activation-range-multiplier", str(activation_range_multiplier),
              "--output-range-multiplier", str(output_range_multiplier),
              "--threads", "8"]
    fields = {"platform": p["platform"], "context": p["context"], "prefill-tokens": p["prefill_tokens"],
              "layers-per-chunk": layers or json.loads((model / "config.json").read_text())["text_config"]["num_hidden_layers"],
              "vision-tokens": p["vision_tokens"], "vision-layers-per-chunk": p["vision_layers_per_chunk"],
              "group-size": p["group_size"], "head-bit": p["head_bit"], "weight-method": p.get("weight_method", "rtn"),
              "gptq-damping": p.get("gptq_damping", .01), "gptq-block-size": p.get("gptq_block_size", 128),
              "kv-group-size": kv_group or p.get("kv_group_size", 32), "kv-residual-tokens": p.get("kv_residual_tokens", 0),
              "kv-sink-tokens": p.get("kv_sink_tokens", 0)}
    for key, value in fields.items():
        values.extend(["--" + key, str(value)])
    if fuse_vision or p.get("fused_vision"):
        values.append("--fuse-vision")
    if compiler_work_dir is not None:
        values.extend(["--compiler-work-dir", str(compiler_work_dir)])
    return parse_args(values)


def prepare(source_dir, experiment):
    source = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if experiment.manifest.get("calibration"):
        if experiment.manifest.get("external_weight_rebuild", {}).get("source_manifest_sha256") != sha256(source_dir / "manifest.json"):
            raise ValueError("Resume source manifest differs")
        for field in ("config", "checkpoint"):
            calibration = experiment.manifest["calibration"]
            if sha256(experiment.output / calibration[field]) != calibration[field + "_sha256"]:
                raise ValueError("Resume calibration changed: " + field)
        return
    if source["profile"].get("precision") != "w4a16" or source["profile"].get("head_bit") != 4:
        raise ValueError("A frozen W4 build with a W4 head is required")
    # Copy metadata; immutable binary checkpoints/graphs can use hard links.
    # Never link a manifest or quantization JSON that an exporter may rewrite.
    def copy_file(src, dst):
        src, dst = Path(src), Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if sha256(src) != sha256(dst):
                raise ValueError("Resume file differs: " + str(dst))
            return str(dst)
        if src.suffix == ".json":
            shutil.copyfile(src, dst)
        else:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
    for field in ("config", "checkpoint"):
        relative = source["calibration"][field]
        src = source_dir / relative
        if sha256(src) != source["calibration"][field + "_sha256"]:
            raise ValueError("Frozen calibration changed: " + field)
        copy_file(src, experiment.output / relative)
    for entry in source["graphs"]:
        component = "vision" if entry["component"].startswith("vision") else entry["component"]
        if component in experiment.args.components:
            # A graph-only math change does not require re-encoding billions
            # of frozen W4 indices. Reuse only a verified parameter/weight pair
            # with identical calibration ranges and chunk grouping, and make
            # independent copies so a later SDK write cannot alter the source.
            q = entry.get("quantization")
            same_ranges = (source["profile"].get("activation_range_multiplier", 1) == experiment.args.activation_range_multiplier and
                           source["profile"].get("output_range_multiplier", 1) == experiment.args.output_range_multiplier)
            same_vision = source["profile"].get("fused_vision", False) == getattr(experiment.args, "fuse_vision", False)
            if q and same_ranges and source["profile"]["layers_per_chunk"] == experiment.args.layers_per_chunk and (component != "vision" or same_vision):
                directory = Path(q["parameters"]).parent.parent
                audit = json.loads((source_dir / directory / "encoding.json").read_text())
                if audit.get("calibrated_sha256") != source["calibration"]["checkpoint_sha256"]:
                    raise ValueError("Frozen encoding calibration differs")
                for name, expected in (("quant_output/quant_params_file", q["parameters_sha256"]),
                                       ("quant_output/fake_quant_weight.pth", audit["fake_quant_weight_sha256"])):
                    src = source_dir / directory / name
                    if sha256(src) != expected:
                        raise ValueError("Frozen encoding pair changed: " + str(src))
                    dst = experiment.output / directory / name
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
                    elif sha256(dst) != expected:
                        raise ValueError("Resume encoding pair changed: " + str(dst))
                dst = experiment.output / directory / "encoding.json"
                if not dst.exists(): shutil.copyfile(source_dir / directory / "encoding.json", dst)
            continue
        directory = Path(entry["onnx"]).parent
        shutil.copytree(source_dir / directory, experiment.output / directory, copy_function=copy_file, dirs_exist_ok=True)
        if entry.get("quantization"):
            directory = Path(entry["quantization"]["parameters"]).parent.parent
            dst = experiment.output / directory
            shutil.copytree(source_dir / directory, dst, copy_function=copy_file, dirs_exist_ok=True)
        experiment.manifest["graphs"].append(copy.deepcopy(entry))
    experiment.manifest["calibration"] = copy.deepcopy(source["calibration"])
    experiment.manifest["external_weight_rebuild"] = {
        "source_manifest_sha256": sha256(source_dir / "manifest.json"),
        "frozen_checkpoint_sha256": source["calibration"]["checkpoint_sha256"],
        "device_tested": False, "quantization_recalibrated": False,
        "activation_range_multiplier": experiment.args.activation_range_multiplier,
        "output_range_multiplier": experiment.args.output_range_multiplier,
    }
    experiment.save()


def compile_external(experiment):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugin_quant_visual_matmul_route_v1"))
    from build_kirin_offline import validate_compile_log
    ddk = experiment.args.ddk
    env = dict(os.environ, SOC_VERSION=experiment.args.platform, PYTHONDONTWRITEBYTECODE="1")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    for entry in experiment.manifest["graphs"]:
        if entry.get("external_weights"):
            if sha256(experiment.output / entry["omc"]) != entry["omc_sha256"] or any(
                    sha256(experiment.output / w["file"]) != w["sha256"] for w in entry["external_weights"]):
                raise ValueError("Compiled external graph/weight pair changed: " + entry["name"])
            continue
        graph = experiment.output / entry["onnx"]
        if sha256(graph) != entry["onnx_sha256"] or any(
                sha256(graph.parent / name) != value for name, value in entry["external_data_sha256"].items()):
            raise ValueError("ONNX changed: " + entry["name"])
        output = graph.parent / "external"
        output.mkdir(exist_ok=True)
        shape = ";".join(v["name"] + ":" + ",".join(map(str, v["shape"])) for v in entry["inputs"])
        parameters = None
        if entry.get("quantization"):
            q = entry["quantization"]
            parameters = experiment.output / q["parameters"]
            if sha256(parameters) != q["parameters_sha256"]:
                raise ValueError("Quantization parameters changed")
        print("compile external " + entry["name"], flush=True)
        log = graph.parent / "compile_external.log"
        # Match the regular compiler path: OMG rejects dots, whitespace and
        # hyphens in input/output directories even when the filesystem accepts them.
        staging_root = getattr(experiment.args, "compiler_work_dir", Path("/tmp"))
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="kirin_external_", dir=staging_root) as directory:
            staging = Path(directory)
            staged_graph = staging / "onnx/model.onnx"
            staged_graph.parent.mkdir()
            shutil.copy2(graph, staged_graph)
            for name in entry["external_data_sha256"]:
                target = staged_graph.parent / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(graph.parent / name, target)
            staged_output = staging / "external"
            staged_output.mkdir()
            cmd = ["bash", "-c", 'source "$1" && { shift; exec "$@"; }', "kirin-external",
                   str(ddk / "tools/tools_ascendc/set_ascendc_env.sh"), str(ddk / "tools/tools_omg/omg"),
                   "--model=" + str(staged_graph), "--framework=5", "--output=" + str(staged_output),
                   "--input_shape=" + shape, "--target=omc", "--platform=" + experiment.args.platform,
                   "--save_weights_as_external_data=true", "--weight_data_type=FP16"]
            if parameters:
                staged_parameters = staging / "quant_params_file"
                shutil.copy2(parameters, staged_parameters)
                cmd.append("--compress_conf=" + str(staged_parameters))
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(cmd, cwd=staged_output, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError("External OMC compilation failed: " + str(log))
            validate_compile_log(log.read_text(errors="replace"))
            files = [staged_output / "external.omc", *sorted(staged_output.glob("*.weight"))]
            if len(files) < 2 or any(not p.is_file() or not p.stat().st_size for p in files):
                raise RuntimeError("Compiler did not create graph/weight pair: " + entry["name"])
            for path in files:
                shutil.copy2(path, output / path.name)
        artifact = output / "external.omc"
        weights = sorted(output.glob("*.weight"))
        if not artifact.is_file() or not artifact.stat().st_size or not weights:
            raise RuntimeError("Compiler did not create graph/weight pair: " + entry["name"])
        entry.update(omc=str(artifact.relative_to(experiment.output)), omc_sha256=sha256(artifact),
                     omc_bytes=artifact.stat().st_size, pure_npu=True,
                     compiled_onnx_sha256=entry["onnx_sha256"],
                     external_weights=[{"file": str(p.relative_to(experiment.output)), "sha256": sha256(p),
                                        "bytes": p.stat().st_size} for p in weights])
        experiment.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-build", "output", "model", "ddk"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layers-per-chunk", type=int)
    parser.add_argument("--components", help="Components to rebuild; unselected compiled graphs are copied unchanged")
    parser.add_argument("--fuse-vision", action="store_true")
    parser.add_argument("--kv-group", type=int)
    parser.add_argument("--kv-dtype", choices=("fp16", "int8"), default="fp16")
    parser.add_argument("--activation-range-multiplier", type=float, default=1)
    parser.add_argument("--output-range-multiplier", type=float, default=1)
    parser.add_argument("--steps", default="prepare,export,compile")
    parser.add_argument("--compiler-work-dir", type=Path, default=Path("/tmp"))
    options = parser.parse_args()
    source_dir, output, model, ddk = [getattr(options, name).resolve() for name in ("source_build", "output", "model", "ddk")]
    if output == source_dir or source_dir in output.parents or output in source_dir.parents:
        raise ValueError("Rebuild must use an independent directory")
    source = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    from w4a16 import W4Experiment
    import torch
    experiment = W4Experiment(arguments(source, output, model, ddk, options.layers_per_chunk, options.kv_group,
                                       options.kv_dtype, options.activation_range_multiplier, options.output_range_multiplier,
                                       options.components, options.fuse_vision, options.compiler_work_dir))
    torch.set_num_threads(experiment.args.threads)
    for step in options.steps.split(","):
        if step == "prepare":
            prepare(source_dir, experiment)
        elif step == "export":
            experiment.export()
        elif step == "compile":
            compile_external(experiment)
        else:
            raise ValueError("Unknown step: " + step)


if __name__ == "__main__":
    main()
