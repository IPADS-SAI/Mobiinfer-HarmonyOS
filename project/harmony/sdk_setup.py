#!/usr/bin/env python3
"""Standard-library helpers for the Harmony build/setup shell entry points."""

import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from zipfile import BadZipFile, ZipFile


HIAI_LIBS = ("libhiai.so", "libhiai_ir.so", "libhiai_ir_build.so")


def log(message):
    print("[mobi-sdk] " + message, flush=True)


def config_path(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError("Set {} in sdk_config.sh".format(name))
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("{} must be an absolute path: {}".format(name, value))
    return path.resolve()


def require_files(root, names):
    for name in names:
        if not (root / name).is_file():
            raise ValueError("Missing file: {}".format(root / name))


def require_arm64(path):
    with path.open("rb") as stream:
        header = stream.read(20)
    if (len(header) != 20 or header[:6] != b"\x7fELF\x02\x01"
            or struct.unpack_from("<HH", header, 16) != (3, 183)):
        raise ValueError("Expected an ARM64 little-endian ELF shared library: {}".format(path))


def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def unpack_sdk(archive_path, destination, required):
    """Reuse complete SDKs; otherwise extract safely into a new directory, then rename."""
    if destination.is_dir() and all((destination / name).is_file() for name in required):
        log("Using existing SDK: {}".format(destination))
        return
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Incomplete SDK directory: {}. Select a new empty directory or repair it."
                         .format(destination))
    if not archive_path.is_file():
        raise ValueError("SDK archive not found: {}. Download it or configure an installed SDK."
                         .format(archive_path))
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        # Check all names before writing, including entries outside a possible wrapper directory.
        for info in infos:
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                raise ValueError("Unsafe archive path: {}".format(info.filename))
        prefixes = {info.filename[:-len(required[0])] for info in infos
                    if info.filename == required[0] or info.filename.endswith("/" + required[0])}
        if len(prefixes) != 1:
            raise ValueError("Cannot locate SDK layout {} in {}".format(required[0], archive_path))
        prefix = prefixes.pop()
        names = {info.filename for info in infos}
        if any(prefix + name not in names for name in required):
            raise ValueError("SDK archive is missing required files: {}".format(archive_path))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".mobi-sdk-", dir=str(destination.parent)) as temp:
            root = Path(temp) / "sdk"
            root.mkdir()
            links = []
            for info in infos:
                if not info.filename.startswith(prefix):
                    continue
                relative = info.filename[len(prefix):]
                if not relative:
                    continue
                target = root / relative
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    link = archive.read(info).decode("utf-8")
                    if Path(link).is_absolute() or not within((target.parent / link).resolve(), root):
                        raise ValueError("Unsafe archive symlink: {} -> {}".format(relative, link))
                    links.append((target, link))
                elif info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    if mode & 0o777:
                        target.chmod(mode & 0o777)
            # Create symlinks last so an archive cannot redirect another member's writes.
            for target, link in links:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(link)
            # Resolve again to catch chains of links that would leave the extraction directory.
            for target, _ in links:
                if not within(target.resolve(), root):
                    raise ValueError("Archive symlink leaves SDK: {}".format(target))
            require_files(root, required)
            if destination.exists():
                destination.rmdir()  # Only the empty directory accepted above.
            root.rename(destination)
    log("Extracted {} -> {}".format(archive_path.name, destination))


def copy_file(source, target):
    if target.is_symlink():
        raise ValueError("Refusing to overwrite symlink: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or not filecmp.cmp(str(source), str(target), shallow=False):
        shutil.copy2(source, target)


def copy_tree(source, target):
    if target.is_symlink():
        raise ValueError("Refusing to copy into symlink: {}".format(target))
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        output = target / item.name
        if item.is_symlink():
            link = os.readlink(item)
            if output.is_symlink() and os.readlink(output) == link:
                continue
            if output.exists() or output.is_symlink():
                raise ValueError("Conflicting plugin symlink destination: {}".format(output))
            output.symlink_to(link)
        elif item.is_dir():
            copy_tree(item, output)
        else:
            copy_file(item, output)


def hiai_libraries():
    root = config_path("CANN_KIT_ROOT") / "ddk/ai_ddk_lib"
    require_files(root, ["include/hiai_ir_build.h"] + ["lib64/" + name for name in HIAI_LIBS])
    libraries = sorted((root / "lib64").glob("libhiai*.so"))
    for path in libraries:
        require_arm64(path)
    return root, libraries


def prepare_hiai():
    required = ["ddk/ai_ddk_lib/include/hiai_ir_build.h"]
    required += ["ddk/ai_ddk_lib/lib64/" + name for name in HIAI_LIBS]
    unpack_sdk(config_path("CANN_KIT_ZIP"), config_path("CANN_KIT_ROOT"), required)
    root, libraries = hiai_libraries()  # Validate all libraries before copying anything into the repo.
    target = config_path("MOBI_ROOT") / "source/backend/hiai/3rdParty"
    copy_tree(root / "include", target / "include")
    for path in libraries:
        copy_file(path, target / "arm64-v8a" / path.name)
    log("Prepared HiAI headers and {} ARM64 libraries: {}".format(len(libraries), target))


def check_build():
    require_files(config_path("HARMONY_HOME"), ["native/build/cmake/ohos.toolchain.cmake"])
    root = config_path("MOBI_ROOT")
    require_files(root, ["CMakeLists.txt", "source/backend/hiai/3rdParty/include/hiai_ir_build.h"])
    for name in HIAI_LIBS:
        require_arm64(root / "source/backend/hiai/3rdParty/arm64-v8a" / name)


def check_app():
    root = config_path("MOBI_APP_ROOT")
    require_files(root, ["entry/src/main/cpp/CMakeLists.txt", "entry/src/main/cpp/offline_npu2/OfflineNpu2.cpp"])


def sync_app():
    check_app()
    root = config_path("MOBI_ROOT")
    require_files(root, ["include/MNN/Interpreter.hpp", "transformers/llm/engine/include/llm/llm.hpp"])
    library = config_path("MOBI_BUILD_DIR") / "libMNN.so"
    require_arm64(library)
    _, libraries = hiai_libraries()
    target = config_path("MOBI_APP_ROOT") / "entry"
    for path in [library] + libraries:
        copy_file(path, target / "libs/arm64-v8a" / path.name)
    copy_tree(root / "include/MNN", target / "src/main/cpp/include/MNN")
    copy_tree(root / "transformers/llm/engine/include/llm", target / "src/main/cpp/include/llm")
    log("Synced libMNN.so, {} HiAI libraries and public headers: {}".format(len(libraries), target))
    log("Build the App in DevEco Studio to generate libentry.so and the HAP.")


def prepare_offline(extract_only):
    ddk = config_path("KIRIN_DDK")
    # Huawei's install.sh expands several internal paths without quoting.
    if not extract_only and any(character.isspace() for character in str(ddk)):
        raise ValueError("AscendC's vendor installer needs KIRIN_DDK without whitespace: {}".format(ddk))
    unpack_sdk(config_path("KIRIN_DDK_ZIP"), ddk,
               ["tools/tools_omg/omg", "tools/tools_omg/master/omg",
                "tools/tools_ascendc/install.sh", "tools/tools_ascendc/set_ascendc_env.sh"])
    plugin = config_path("KIRIN_PLUGIN_ROOT")
    unpack_sdk(config_path("KIRIN_PLUGIN_ZIP"), plugin,
               ["image/ddk_platform_plugin/kirin9030/config/kirin9030.ini"])
    copy_tree(plugin / "image/ddk_platform_plugin/kirin9030", ddk / "tools/platform/kirin9030")
    for name in ["tools/tools_omg/omg", "tools/tools_omg/master/omg"]:
        path = ddk / name
        path.chmod(path.stat().st_mode | 0o111)
    log("Installed Kirin9030 plugin: {}".format(ddk / "tools/platform/kirin9030"))
    if extract_only:
        log("Extraction complete; AscendC initialization was skipped (--extract-only).")
        return
    ascendc = ddk / "tools/tools_ascendc"
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    check = ["bash", "-c", 'source "$1" && "$2" -c "import te_fusion"', "mobi-ascendc-check",
             str(ascendc / "set_ascendc_env.sh"), sys.executable]
    stamp = ascendc / ".mobi-setup.json"
    state = {"python": sys.executable, "installer_sha256": hashlib.sha256((ascendc / "install.sh").read_bytes()).hexdigest()}
    try:
        previous = json.loads(stamp.read_text())
    except (OSError, ValueError):
        previous = None
    if previous == state and subprocess.run(check, cwd=ascendc, env=env, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL).returncode == 0:
        log("AscendC already initialized and te_fusion import passed.")
        return
    log("Initializing AscendC with {} (vendor install.sh may install Python dependencies).".format(sys.executable))
    subprocess.run(["bash", "-c", 'source "$1"', "mobi-ascendc-init", str(ascendc / "install.sh")],
                   cwd=ascendc, env=env, check=True)
    subprocess.run(check, cwd=ascendc, env=env, check=True)
    stamp.write_text(json.dumps(state, indent=2) + "\n")
    log("AscendC initialized and te_fusion import passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare-hiai", "check-build", "check-app", "sync-app"]:
        commands.add_parser(name)
    offline = commands.add_parser("prepare-offline")
    offline.add_argument("--extract-only", action="store_true", help="Skip vendor AscendC/Python installation")
    args = parser.parse_args()
    try:
        if args.command == "prepare-offline":
            prepare_offline(args.extract_only)
        else:
            {"prepare-hiai": prepare_hiai, "check-build": check_build,
             "check-app": check_app, "sync-app": sync_app}[args.command]()
    except (OSError, ValueError, BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, "[mobi-sdk] ERROR: {}\n".format(error))


if __name__ == "__main__":
    main()
