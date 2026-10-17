"""Exercise shell entry points in isolated workspaces; no device or vendor installer is used."""

import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile, ZipInfo


SCRIPTS = Path(__file__).resolve().parents[1]


def elf(machine=183):
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", header, 16, 3, machine)
    return bytes(header)


def make_zip(path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        for name, value in files.items():
            data, mode = value if isinstance(value, tuple) else (value, stat.S_IFREG | 0o644)
            info = ZipInfo(name)
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, data)


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mobi setup ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "source tree"
        self.app = self.base / "App tree"
        self.sdk = self.base / "SDK tree"
        self.build = self.base / "build output"
        self.env = dict(os.environ)
        self.env.pop("MOBI_SDK_CONFIG", None)
        self.env.update(MOBI_ROOT=str(self.repo), MOBI_APP_ROOT=str(self.app), MOBI_SDK_ROOT=str(self.sdk),
                        MOBI_DOWNLOAD_DIR=str(self.sdk / "downloads"),
                        HARMONY_HOME=str(self.sdk / "harmony/openharmony"),
                        CANN_KIT_ROOT=str(self.sdk / "CANN"), CANN_KIT_ZIP=str(self.sdk / "CANN.zip"),
                        KIRIN_DDK=str(self.sdk / "DDK"), KIRIN_DDK_ZIP=str(self.sdk / "DDK.zip"),
                        KIRIN_PLUGIN_ROOT=str(self.sdk / "plugin"), KIRIN_PLUGIN_ZIP=str(self.sdk / "plugin.zip"),
                        MOBI_BUILD_DIR=str(self.build), MOBI_BUILD_JOBS="2", MOBI_PYTHON=sys.executable,
                        PYTHONDONTWRITEBYTECODE="1")
        for root, relative, data in [
                (self.repo, "CMakeLists.txt", "project(fixture)\n"),
                (self.repo, "include/MNN/Interpreter.hpp", "fixture MNN header\n"),
                (self.repo, "transformers/llm/engine/include/llm/llm.hpp", "fixture LLM header\n"),
                (self.app, "entry/src/main/cpp/CMakeLists.txt", "project(fixture)\n"),
                (self.app, "entry/src/main/cpp/offline_npu2/OfflineNpu2.cpp", "// fixture bridge\n"),
                (self.app, "entry/keep.txt", "keep unrelated files\n")]:
            self.write(root / relative, data)
        self.cann_files = {"ddk/ai_ddk_lib/include/hiai_ir_build.h": b"fixture HiAI header\n"}
        for name in ["libhiai.so", "libhiai_ir.so", "libhiai_ir_build.so", "libhiai_ir_build_aipp.so"]:
            self.cann_files["ddk/ai_ddk_lib/lib64/" + name] = elf()
        make_zip(Path(self.env["CANN_KIT_ZIP"]), self.cann_files)

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())

    def run_script(self, name, *args, ok=True):
        result = subprocess.run(["bash", str(SCRIPTS / name), *args], cwd=self.base,
                                env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def offline_archives(self, installer=None):
        # The real vendor install.sh does not support whitespace in its DDK path.
        ddk_temp = tempfile.TemporaryDirectory(prefix="mobi-ddk-", dir="/tmp")
        self.addCleanup(ddk_temp.cleanup)
        self.env["KIRIN_DDK"] = str(Path(ddk_temp.name) / "DDK")
        install = installer or (
            '#!/bin/bash\nmkdir -p package/python\n'
            'printf "# fixture module\\n" > package/python/te_fusion.py\n'
            'printf "initialized\\n" >> install-count\n')
        prefix = "DDK-tools-test/"
        make_zip(Path(self.env["KIRIN_DDK_ZIP"]), {
            prefix + "tools/tools_omg/omg": b"#!/bin/bash\n",
            prefix + "tools/tools_omg/master/omg": b"fixture compiler",
            prefix + "tools/tools_ascendc/install.sh": install,
            prefix + "tools/tools_ascendc/set_ascendc_env.sh":
                'export PYTHONPATH="$PWD/package/python${PYTHONPATH:+:$PYTHONPATH}"\n',
        })
        prefix = "image/ddk_platform_plugin/kirin9030/"
        make_zip(Path(self.env["KIRIN_PLUGIN_ZIP"]), {
            prefix + "config/kirin9030.ini": b"fixture platform",
            prefix + "simulator/real.so": (b"fixture simulator", stat.S_IFREG | 0o755),
            prefix + "simulator/alias.so": (b"real.so", stat.S_IFLNK | 0o777),
        })

    def test_prepare_and_app_copy_are_repeatable(self):
        self.run_script("01_prepare_hiai.sh")
        self.write(self.build / "libMNN.so", elf())
        self.run_script("04_sync_app.sh")
        target = self.app / "entry/libs/arm64-v8a"
        self.assertEqual((target / "libMNN.so").read_bytes(), elf())
        self.assertEqual(len(list(target.glob("*.so"))), 5)
        self.assertEqual((self.app / "entry/src/main/cpp/include/MNN/Interpreter.hpp").read_text(),
                         "fixture MNN header\n")
        before = (target / "libMNN.so").stat().st_mtime_ns
        Path(self.env["CANN_KIT_ZIP"]).unlink()  # Reuse an already installed SDK without the ZIP.
        self.run_script("01_prepare_hiai.sh")
        self.run_script("04_sync_app.sh")
        self.assertEqual(before, (target / "libMNN.so").stat().st_mtime_ns)
        self.assertEqual((self.app / "entry/keep.txt").read_text(), "keep unrelated files\n")

    def test_x86_library_is_rejected_before_app_copy(self):
        self.run_script("01_prepare_hiai.sh")
        self.write(self.build / "libMNN.so", elf(machine=62))
        target = self.app / "entry/libs/arm64-v8a/libMNN.so"
        self.write(target, b"existing app library")
        result = self.run_script("04_sync_app.sh", ok=False)
        self.assertIn("ARM64", result.stdout)
        self.assertEqual(target.read_bytes(), b"existing app library")
        self.assertFalse((self.app / "entry/src/main/cpp/include").exists())

    def test_missing_library_is_rejected_before_repo_copy(self):
        del self.cann_files["ddk/ai_ddk_lib/lib64/libhiai_ir.so"]
        make_zip(Path(self.env["CANN_KIT_ZIP"]), self.cann_files)
        self.run_script("01_prepare_hiai.sh", ok=False)
        self.assertFalse((self.repo / "source/backend/hiai/3rdParty").exists())

    def test_incomplete_sdk_is_preserved(self):
        root = Path(self.env["CANN_KIT_ROOT"])
        self.write(root / "existing.txt", b"do not overwrite")
        result = self.run_script("01_prepare_hiai.sh", ok=False)
        self.assertIn("Incomplete SDK", result.stdout)
        self.assertEqual((root / "existing.txt").read_bytes(), b"do not overwrite")

    def test_zip_escape_is_rejected(self):
        for name, value in [("../outside.txt", b"escape"),
                            ("ddk/escape", (b"../../outside", stat.S_IFLNK | 0o777))]:
            with self.subTest(name=name):
                make_zip(Path(self.env["CANN_KIT_ZIP"]), dict(self.cann_files, **{name: value}))
                self.run_script("01_prepare_hiai.sh", ok=False)
                self.assertFalse(Path(self.env["CANN_KIT_ROOT"]).exists())

    def test_offline_tools_preserve_symlinks_and_initialize_once(self):
        self.offline_archives()
        self.run_script("setup.sh", "offline-tools")
        ddk = Path(self.env["KIRIN_DDK"])
        plugin = ddk / "tools/platform/kirin9030"
        self.assertTrue((plugin / "config/kirin9030.ini").is_file())
        self.assertTrue((plugin / "simulator/alias.so").is_symlink())
        self.assertEqual(os.readlink(plugin / "simulator/alias.so"), "real.so")
        self.assertTrue((plugin / "simulator/real.so").stat().st_mode & stat.S_IXUSR)
        self.assertTrue((ddk / "tools/tools_omg/master/omg").stat().st_mode & stat.S_IXUSR)
        self.run_script("02_prepare_offline_tools.sh")
        self.assertEqual((ddk / "tools/tools_ascendc/install-count").read_text().splitlines(), ["initialized"])

    def test_failed_vendor_install_is_reported_without_success_stamp(self):
        self.offline_archives(installer="#!/bin/bash\nexit 17\n")
        self.run_script("02_prepare_offline_tools.sh", ok=False)
        self.assertFalse((Path(self.env["KIRIN_DDK"]) / "tools/tools_ascendc/.mobi-setup.json").exists())

    def test_extract_only_does_not_run_installer(self):
        self.offline_archives(installer="#!/bin/bash\nexit 17\n")
        result = self.run_script("02_prepare_offline_tools.sh", "--extract-only")
        self.assertIn("skipped", result.stdout)

    def test_vendor_path_restriction_is_reported_before_install(self):
        self.offline_archives()
        self.env["KIRIN_DDK"] = str(self.sdk / "DDK with spaces")
        result = self.run_script("02_prepare_offline_tools.sh", ok=False)
        self.assertIn("without whitespace", result.stdout)
        self.assertFalse(Path(self.env["KIRIN_DDK"]).exists())

    def mock_cmake(self):
        tool = self.base / "bin/cmake"
        self.write(tool, '#!' + sys.executable + '\n' +
                   'import json, os, pathlib, sys\n'
                   'with open(os.environ["MOBI_CMAKE_LOG"], "a") as out: out.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                   'if "--build" not in sys.argv and os.environ.get("MOBI_CMAKE_FAIL"): sys.exit(23)\n'
                   'if "--build" in sys.argv:\n'
                   ' pathlib.Path(sys.argv[2], "libMNN.so").write_bytes(bytes.fromhex("' + elf().hex() + '"))\n')
        tool.chmod(0o755)
        self.env["PATH"] = str(tool.parent) + os.pathsep + self.env["PATH"]
        self.env["MOBI_CMAKE_LOG"] = str(self.base / "cmake.jsonl")
        self.write(Path(self.env["HARMONY_HOME"]) / "native/build/cmake/ohos.toolchain.cmake", "# fixture")

    def test_app_workflow_preserves_cmake_arguments_with_spaces(self):
        self.mock_cmake()
        self.run_script("setup.sh", "app", "-DMOBI_CUSTOM=value with spaces")
        calls = [json.loads(line) for line in Path(self.env["MOBI_CMAKE_LOG"]).read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertIn(str(self.repo), calls[0])
        self.assertIn("-DCMAKE_TOOLCHAIN_FILE=" + self.env["HARMONY_HOME"] +
                      "/native/build/cmake/ohos.toolchain.cmake", calls[0])
        self.assertIn("-DMOBI_CUSTOM=value with spaces", calls[0])
        self.assertEqual(calls[1], ["--build", str(self.build), "--parallel", "2"])
        self.assertTrue((self.app / "entry/libs/arm64-v8a/libMNN.so").is_file())

    def test_cmake_failure_stops_build_and_app_sync(self):
        self.mock_cmake()
        self.env["MOBI_CMAKE_FAIL"] = "1"
        target = self.app / "entry/libs/arm64-v8a/libMNN.so"
        self.write(target, b"existing app library")
        self.run_script("setup.sh", "app", ok=False)
        calls = Path(self.env["MOBI_CMAKE_LOG"]).read_text().splitlines()
        self.assertEqual(len(calls), 1)
        self.assertEqual(target.read_bytes(), b"existing app library")


if __name__ == "__main__":
    unittest.main()
