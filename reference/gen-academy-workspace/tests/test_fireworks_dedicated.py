"""Offline lifecycle tests: no Fireworks credentials, GPU, or network required."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dedicated", ROOT / "scripts/fireworks_dedicated.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeAPI:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.methods = []

    def call(self, method, resource, payload=None):
        self.methods.append(method)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class DedicatedTests(unittest.TestCase):
    def test_default_plan_is_offline_and_split_is_frozen(self):
        with patch.object(runner.API, "__init__", side_effect=AssertionError("No API")):
            plan = runner.local_plan(runner.read_config())
        self.assertEqual(plan["optimizer_steps"], 279)
        self.assertEqual(plan["rows"], {"train": 372, "valid": 94, "test": 117})
        cfg = copy.deepcopy(runner.read_config())
        cfg["splits"]["train"]["sha256"] = "changed"
        with self.assertRaises(ValueError):
            runner.local_plan(cfg)

    def test_local_tokenizer_resolves_from_project_and_rejects_tampering(self):
        cfg = runner.read_config()
        self.assertTrue(Path(runner.local_tokenizer(cfg)).is_absolute())
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "tokenizer.json"
            target.write_text("changed")
            cfg["tokenizer_local_dir"] = folder
            cfg["tokenizer_local_sha256"] = {
                "tokenizer.json": runner.read_config()["tokenizer_local_sha256"]["tokenizer.json"]}
            with self.assertRaisesRegex(ValueError, "Local tokenizer changed"):
                runner.local_tokenizer(cfg)

    def test_shape_change_to_more_gpus_is_rejected(self):
        cfg = runner.read_config()
        shape = {"name": cfg["training_shape_version"], "snapshot": {
            "baseModel": cfg["base_model"], "trainerMode": "LORA_TRAINER",
            "acceleratorType": "NVIDIA_H200_141GB", "acceleratorCount": 2,
            "nodeCount": 1, "maxSupportedContextLength": 131072}}
        with self.assertRaises(ValueError):
            runner.verify_shape(shape, cfg)
        shape["snapshot"]["acceleratorCount"] = 1
        runner.verify_shape(shape, cfg)

    def test_delete_acknowledgment_does_not_mean_stopped(self):
        api = FakeAPI([(200, {"state": "JOB_STATE_RUNNING"}), (200, {}),
                       (200, {"state": "JOB_STATE_DELETING"}), (200, {}),
                       (404, {})])
        self.assertTrue(runner.cleanup(api, "our-job", attempts=3, sleep=lambda _: None))
        self.assertEqual(api.methods, ["GET", "DELETE", "GET", "DELETE", "GET"])

    def test_cleanup_failure_is_reported_not_silently_accepted(self):
        api = FakeAPI([(403, {}), OSError("offline")])
        self.assertFalse(runner.cleanup(api, "our-job", attempts=2, sleep=lambda _: None))

    def test_timeout_stops_real_local_worker(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                start_new_session=True)
        try:
            with self.assertRaises(TimeoutError):
                runner.wait_worker(proc, 0.05)
        finally:
            runner.stop_worker(proc)
        self.assertIsNotNone(proc.poll())

    def test_existing_job_blocks_creation(self):
        api = FakeAPI([(200, {"state": "JOB_STATE_RUNNING"})])
        with patch.object(runner, "check_runtime"), patch.object(runner, "load_key", return_value="fake"), \
             patch.object(runner, "API", return_value=api):
            with self.assertRaises(ValueError):
                runner.launch(runner.read_config())
        self.assertEqual(api.methods, ["GET"])

    def test_uncertain_submission_is_not_retried_and_cleanup_runs(self):
        cfg = runner.read_config()
        shape = {"name": cfg["training_shape_version"], "snapshot": {
            "baseModel": cfg["base_model"], "trainerMode": "LORA_TRAINER",
            "acceleratorType": "NVIDIA_H200_141GB", "acceleratorCount": 1,
            "nodeCount": 1, "maxSupportedContextLength": 131072}}
        api = FakeAPI([(404, {}), (404, {}), (200, shape), TimeoutError("Lost response")])
        with tempfile.TemporaryDirectory() as folder:
            cfg["run_dir"] = folder + "/run"
            with patch.object(runner, "check_runtime"), patch.object(runner, "load_key", return_value="fake"), \
                 patch.object(runner, "API", return_value=api), \
                 patch.object(runner, "cleanup", return_value=True) as cleanup:
                with self.assertRaises(TimeoutError):
                    runner.launch(cfg)
            cleanup.assert_called_once()
            self.assertEqual(api.methods.count("POST"), 1)
            state = json.loads((Path(cfg["run_dir"]) / "supervisor.json").read_text())
            self.assertTrue(state["cleanup_confirmed"])

    def test_redacts_key_and_signed_urls(self):
        value = runner.scrub("secret123 https://storage.example/file?signature=abc", "secret123")
        self.assertNotIn("secret123", value)
        self.assertNotIn("signature", value)


if __name__ == "__main__":
    unittest.main()
