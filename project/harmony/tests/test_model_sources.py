"""Exercise source selection, interrupted downloads and protection of local inputs."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_sources import MARKER, complete_model, prepare_source, source_ready


class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mobi model download ")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "model"

    def snapshot(self, *, model_id, revision, local_dir):
        directory = Path(local_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "tokenizer.json", "quantization_config.json"):
            (directory / name).write_text('{}')
        (directory / "model.safetensors").write_bytes(b"fake model weight")
        return str(directory)

    def test_download_uses_configured_id_revision_and_local_directory(self):
        download = mock.Mock(side_effect=self.snapshot)
        with mock.patch.dict(sys.modules, {"modelscope": SimpleNamespace(snapshot_download=download)}):
            prepare_source(self.directory, "owner/model", "v1", "gptq")
        download.assert_called_once_with(model_id="owner/model", revision="v1", local_dir=str(self.directory))
        self.assertTrue(source_ready(self.directory, "owner/model", "v1", "gptq"))
        self.assertTrue(json.loads((self.directory / MARKER).read_text())["completed"])

    def test_complete_existing_local_model_works_without_modelscope(self):
        self.snapshot(model_id="unused", revision="master", local_dir=str(self.directory))
        with mock.patch.dict(sys.modules, {"modelscope": None}):
            prepare_source(self.directory, "", "master", "original")
        self.assertFalse((self.directory / MARKER).exists())

    def test_custom_missing_local_directory_requires_an_explicit_remote_id(self):
        with self.assertRaisesRegex(ValueError, "missing or incomplete"):
            prepare_source(self.directory, "", "master", "original")
        self.assertFalse(self.directory.exists())

    def test_incomplete_unmanaged_directory_is_not_overwritten(self):
        self.directory.mkdir()
        config = self.directory / "config.json"
        config.write_text('{"custom":true}')
        with self.assertRaisesRegex(ValueError, "will not be overwritten"):
            prepare_source(self.directory, "owner/model", "master", "original")
        self.assertEqual(config.read_text(), '{"custom":true}')

    def test_interrupted_snapshot_is_resumed_even_after_weight_file_arrives(self):
        def interrupted(**kwargs):
            self.snapshot(**kwargs)
            raise RuntimeError("connection interrupted")
        download = mock.Mock(side_effect=interrupted)
        sdk = SimpleNamespace(snapshot_download=download)
        with mock.patch.dict(sys.modules, {"modelscope": sdk}):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                prepare_source(self.directory, "owner/model", "master", "original")
            self.assertFalse(source_ready(self.directory, "owner/model", "master", "original"))
            download.side_effect = self.snapshot
            prepare_source(self.directory, "owner/model", "master", "original")
            prepare_source(self.directory, "owner/model", "master", "original")
        self.assertEqual(download.call_count, 2)

    def test_changing_remote_model_cannot_reuse_another_download(self):
        sdk = SimpleNamespace(snapshot_download=self.snapshot)
        with mock.patch.dict(sys.modules, {"modelscope": sdk}):
            prepare_source(self.directory, "owner/model", "master", "gptq")
        with self.assertRaisesRegex(ValueError, "source changed"):
            source_ready(self.directory, "owner/different", "master", "gptq")
        with self.assertRaisesRegex(ValueError, "source changed"):
            source_ready(self.directory, "owner/model", "v2", "gptq")

    def test_sharded_model_requires_every_weight_shard(self):
        self.snapshot(model_id="unused", revision="master", local_dir=str(self.directory))
        (self.directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight_a": "part1.safetensors", "weight_b": "part2.safetensors"}}))
        (self.directory / "part1.safetensors").write_bytes(b"part 1")
        self.assertFalse(complete_model(self.directory, "original"))
        (self.directory / "part2.safetensors").write_bytes(b"part 2")
        self.assertTrue(complete_model(self.directory, "original"))

    def test_failed_or_incomplete_download_is_never_marked_complete(self):
        sdk = SimpleNamespace(snapshot_download=lambda **kwargs: None)
        with mock.patch.dict(sys.modules, {"modelscope": sdk}):
            with self.assertRaisesRegex(ValueError, "missing required"):
                prepare_source(self.directory, "owner/model", "master", "gptq")
        self.assertFalse(json.loads((self.directory / MARKER).read_text())["completed"])


if __name__ == "__main__":
    unittest.main()
