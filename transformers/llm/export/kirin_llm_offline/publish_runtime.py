"""Publish an independently audited runtime package to a dedicated ModelScope repo.

Credentials are read in-process and are never placed in command-line arguments.
Existing different remote content is rejected rather than overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish(args):
    from modelscope_hub import HubApi
    import requests
    credentials = args.credential_file.read_text(encoding="utf-8")
    match = re.search(r"(?:token|令牌)\s*[:：=]\s*([^\s]+)", credentials, re.I)
    if not match:
        raise ValueError("No ModelScope credential found")
    token = match.group(1).strip("`\"'")
    for name in ("", "modelscope", "modelscope_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    status = {"repo": args.repo, "started_utc": datetime.now(timezone.utc).isoformat(), "status": "verifying_local"}

    def save():
        args.status.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    def public_files(session):
        response = session.get(args.endpoint + "/api/v1/models/" + args.repo + "/repo/files",
                               params={"Revision": "master", "Recursive": "1"}, timeout=(15, 60))
        response.raise_for_status()
        body = response.json()
        if body.get("Success") is False:
            raise RuntimeError("Public ModelScope file-list API failed")
        data = body.get("Data", body)
        values = data.get("Files", []) if isinstance(data, dict) else data
        return {v["Path"]: v for v in values if v.get("Type") != "tree"}

    try:
        manifest = json.loads((args.folder / "release_manifest.json").read_text(encoding="utf-8"))
        known = {}
        for record in manifest["files"]:
            path = args.folder / record["path"]
            if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
                raise ValueError("Local package mismatch: " + record["path"])
            known[record["path"]] = record
        for name in ("SHA256SUMS", "release_manifest.json"):
            path = args.folder / name
            known[name] = {"path": name, "size": path.stat().st_size, "sha256": digest(path)}
        actual = {str(p.relative_to(args.folder)) for p in args.folder.rglob("*") if p.is_file()}
        if actual != set(known):
            raise ValueError("Unexpected or missing release files")
        config = json.loads((args.folder / "config.json").read_text(encoding="utf-8"))
        kv_dtype = config["runtime"].get("kv_dtype", "int8")
        external = any(g.get("external_weights") for g in config["graphs"])
        description = ("Experimental complete Kirin9030 VLM W4A16, " + kv_dtype + " KV, " +
                       ("external weights" if external else "embedded weights") + "; updated App required")
        api = HubApi(token=token, endpoint=args.endpoint)
        owner = args.repo.split("/")[0]
        if api.whoami().username != owner:
            raise ValueError("Authenticated account differs from requested repository owner")
        with requests.Session() as session:
            if api.repo_exists(args.repo, "model"):
                existing = public_files(session)
                for name, record in existing.items():
                    if name == ".gitattributes":
                        continue
                    if name not in known or record.get("Sha256") != known[name]["sha256"]:
                        raise ValueError("Different existing remote content: " + name)
            else:
                api.create_repo(args.repo, "model", visibility="public", license="Apache License 2.0",
                                description=description)
            status.update(status="uploading", files=len(known), total_bytes=sum(v["size"] for v in known.values()))
            save()
            print(json.dumps(status), flush=True)

            def progress(event):
                print(json.dumps({k: event.get(k) for k in ("event", "committed_files", "total_files", "committed_bytes", "total_bytes")}), flush=True)

            api.upload_folder(args.repo, "model", args.folder, path_in_repo="", revision="master",
                              allow_patterns=sorted(known), max_workers=4, disable_tqdm=True, use_cache=True,
                              tracker_path=args.status.with_suffix(".tracker.json"),
                              commit_message="Publish experimental Kirin9030 W4A16 VLM with " + kv_dtype + " KV" +
                                             (" and external weights" if external else ""),
                              sync_remote_repo=False, progress_callback=progress)
            remote = public_files(session)
            for name, expected in known.items():
                record = remote.get(name, {})
                if record.get("Sha256") != expected["sha256"] or record.get("Size") != expected["size"]:
                    raise ValueError("Remote size/hash verification failed: " + name)
            status.update(status="complete", verified_files=len(known), finished_utc=datetime.now(timezone.utc).isoformat(),
                          url="https://www.modelscope.cn/models/" + args.repo)
            save()
            print(json.dumps(status), flush=True)
    except Exception as error:
        safe = str(error).replace(token, "[REDACTED]")
        safe = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", safe)
        status.update(status="failed", error=safe)
        save()
        raise RuntimeError(safe) from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://modelscope.cn")
    publish(parser.parse_args())
