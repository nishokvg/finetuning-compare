#!/usr/bin/env python3
"""Review-first Fireworks SFT runner. Default: local validation only, no API calls."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fireworks_llama_dedicated.json"
STOPPED = {"JOB_STATE_STOPPED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
           "JOB_STATE_COMPLETED", "JOB_STATE_DELETED"}


def read_config():
    return json.loads(CONFIG.read_text())


def local_plan(cfg):
    """Check the frozen split before anything can allocate compute."""
    counts = {}
    for split, spec in cfg["splits"].items():
        data = (ROOT / spec["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"{split} file changed; review the split before running")
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        if len(rows) != spec["rows"]:
            raise ValueError(f"Wrong {split} row count")
        for row in rows:
            messages = row.get("messages", [])
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError(f"Missing assistant label in {split}")
            if any(not isinstance(m.get("content"), str) for m in messages):
                raise ValueError(f"Expected text-only messages in {split}")
        counts[split] = len(rows)
    if cfg["runtime_limit_seconds"] > 1800 or cfg["runtime_limit_seconds"] <= 0:
        raise ValueError("Review required for runtime outside (0, 1800] seconds")
    if cfg["replicas"] != 1 or cfg["gpu_hour_usd"] != 8:
        raise ValueError("Review required for changed hardware or price")
    steps = math.ceil(counts["train"] / cfg["batch_size"]) * cfg["epochs"]
    return {"mode": "PREPARED; NOT LAUNCHED", "model": cfg["base_model"],
            "rows": counts, "optimizer_steps": steps,
            "runtime_limit_minutes": cfg["runtime_limit_seconds"] / 60,
            "training_time_cost_reference_usd": cfg["runtime_limit_seconds"] / 3600 * cfg["gpu_hour_usd"],
            "budget_is_not_a_provider_enforced_cap": True,
            "deployment_created": False, "test_used_for_training": False}


def load_key():
    key = os.environ.get("FIREWORKS_API_KEY", "")
    for raw in (ROOT / ".env").read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if "=" in line:
            name, value = line.split("=", 1)
            if name.strip() == "FIREWORKS_API_KEY" and not key:
                key = value.strip().strip("\"'")
    if not key:
        raise ValueError("FIREWORKS_API_KEY is missing")
    return key


def scrub(text, key=""):
    if key:
        text = text.replace(key, "[REDACTED]")
    text = re.sub(r"https?://[^\s\"'<>]+", "[URL]", text)
    return re.sub(r"\bfw_[A-Za-z0-9_-]+", "[REDACTED]", text)


class SafeStream:
    def __init__(self, stream, key):
        self.stream, self.key = stream, key

    def write(self, text):
        return self.stream.write(scrub(text, self.key))

    def flush(self):
        self.stream.flush()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class API:
    def __init__(self, key):
        self.key = key
        self.opener = urllib.request.build_opener(NoRedirect)

    def call(self, method, resource, payload=None):
        req = urllib.request.Request(
            "https://api.fireworks.ai/v1/" + resource,
            method=method,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.key,
                     "Accept": "application/json", "Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=20) as response:
                data = response.read()
                return response.status, json.loads(data) if data else {}
        except urllib.error.HTTPError as error:
            return error.code, {"message": scrub(error.read().decode(), self.key)[:1000]}


def job_resource(cfg):
    return f"accounts/{cfg['account']}/rlorTrainerJobs/{cfg['job_id']}"


def verify_shape(shape, cfg):
    snapshot = shape.get("snapshot", {})
    expected = {"baseModel": cfg["base_model"], "trainerMode": "LORA_TRAINER",
                "acceleratorType": "NVIDIA_H200_141GB", "acceleratorCount": 1, "nodeCount": 1}
    if shape.get("name") != cfg["training_shape_version"]:
        raise ValueError("Unexpected training shape version")
    for name, value in expected.items():
        if snapshot.get(name) != value:
            raise ValueError(f"Training shape {name} differs from reviewed setting")
    if snapshot.get("maxSupportedContextLength", 0) < cfg["max_seq_len"]:
        raise ValueError("Shape context is too short")


def create_request(cfg):
    # Matches fireworks-ai 1.2.11 TrainerJobManager's validated shape path.
    query = urllib.parse.urlencode({"trainingShape": cfg["training_shape_version"],
                                   "rlorTrainerJobId": cfg["job_id"]})
    resource = f"accounts/{cfg['account']}/rlorTrainerJobs?{query}"
    payload = {"serviceMode": True, "keepAlive": False, "dataset": "",
               "trainingConfig": {"baseModel": cfg["base_model"],
                                  "loraRank": cfg["lora_rank"],
                                  "learningRate": cfg["learning_rate"]},
               "trainerReplicaCount": 1, "useReservation": False,
               "inactivityTimeout": f"{cfg['inactivity_timeout_seconds']}s",
               "displayName": "Support router Llama3B dedicated LoRA"}
    return resource, payload


def cleanup(api, resource, attempts=6, sleep=time.sleep, absence_checks=1):
    """A DELETE acknowledgment alone does not establish that billing stopped."""
    absent = 0
    for _ in range(attempts):
        try:
            code, job = api.call("GET", resource)
            if code == 404:
                absent += 1
                if absent >= absence_checks:
                    return True
            else:
                absent = 0
            if code == 200 and job.get("state") in STOPPED:
                return True
            if code == 200:
                api.call("DELETE", resource)
        except Exception:
            pass
        sleep(5)
    return False


def stop_worker(proc):
    if proc is None:
        return
    # Kill the entire group, including descendants that could keep heartbeats alive.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


def wait_worker(proc, seconds):
    try:
        return proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        raise TimeoutError("30-minute run deadline reached; trainer cleanup required") from None


def local_tokenizer(cfg):
    """Use checksum-verified local artifacts; never fall back to the gated Hub."""
    directory = (ROOT / cfg["tokenizer_local_dir"]).resolve()
    for name, expected in cfg["tokenizer_local_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Local tokenizer changed: {name}; reconvert and review")
    return str(directory)


def runtime_recipe(cfg):
    """Import only in explicit runtime-check/launch modes, never default plan mode."""
    if importlib.metadata.version("fireworks-ai") != cfg["sdk_version"]:
        raise ValueError("Install the pinned Fireworks SDK in the separate environment")
    dist = importlib.metadata.distribution("fireworks-training-cookbook")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != cfg["cookbook_commit"]:
        raise ValueError("Cookbook must be installed from the reviewed commit")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["FIREWORKS_BASE_URL"] = "https://api.fireworks.ai"
    os.environ.pop("FIREWORKS_API_EXTRA_HEADERS", None)
    from training.recipes import sft_loop
    from training.utils import TrainerConfig, WandBConfig
    options = sft_loop.Config(
        log_path=str(ROOT / cfg["run_dir"] / "recipe"),
        base_model=cfg["base_model"], dataset=str(ROOT / cfg["splits"]["train"]["file"]),
        evaluation_dataset=str(ROOT / cfg["splits"]["valid"]["file"]),
        tokenizer_model=local_tokenizer(cfg), tokenizer_revision=None,
        tokenizer_trust_remote_code=False, renderer_name=cfg["renderer_name"],
        train_on_what="all_assistant_messages",
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], max_seq_len=cfg["max_seq_len"],
        learning_rate=cfg["learning_rate"], warmup_steps=0,
        lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        weight_decay=0.0, adam_beta2=0.999, seed=42,
        pipeline_depth=1, render_workers=1, eval_auto_carveout=False,
        dcp_save_interval=93, sampler_save_interval=93, save_final_checkpoint=True,
        output_model_id=cfg["output_model_id"], serverless=False,
        step_timeout=120, wandb=WandBConfig(),
        trainer=TrainerConfig(job_id=cfg["job_id"],
                              training_shape_id=cfg["training_shape_version"],
                              use_reservation=False, replica_count=1,
                              timeout_s=300, pending_timeout_s=300,
                              inactivity_timeout=f"{cfg['inactivity_timeout_seconds']}s"))
    return sft_loop, options


def check_runtime(cfg):
    """Render all train/valid rows with the actual pinned recipe before GPU creation."""
    recipe, options = runtime_recipe(cfg)
    renderer = recipe._resolved_renderer_name(
        tokenizer_model=local_tokenizer(cfg), renderer_name=cfg["renderer_name"],
        thinking_trace_history_mode="")
    recipe._init_render_worker(local_tokenizer(cfg), renderer,
                               "all_assistant_messages", cfg["max_seq_len"],
                               tokenizer_revision=None,
                               tokenizer_trust_remote_code=False)
    for split in ("train", "valid"):
        path = ROOT / cfg["splits"][split]["file"]
        for number, line in enumerate(path.read_text().splitlines(), 1):
            datum = recipe._render_one_worker(json.loads(line))
            if datum is None or (isinstance(datum, list) and len(datum) != 1):
                raise ValueError(f"{split} row {number}: dropped/split by renderer; review required")
    print("Pinned recipe imports and train/validation rendering passed; no GPU created.")


def launch(cfg):
    local_plan(cfg)
    check_runtime(cfg)
    key = load_key()
    api = API(key)
    resource = job_resource(cfg)
    for target in (resource, f"accounts/{cfg['account']}/models/{cfg['output_model_id']}"):
        code, _ = api.call("GET", target)
        if code != 404:
            raise ValueError("Job/model absence not established; no automatic resume or duplicate run")
    code, shape = api.call("GET", cfg["training_shape_version"])
    if code != 200:
        raise ValueError("Could not verify the pinned training shape")
    verify_shape(shape, cfg)
    run_dir = ROOT / cfg["run_dir"]
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = {"job_resource": resource, "status": "creating", "config": cfg}
    state_path = run_dir / "supervisor.json"
    state_path.write_text(json.dumps(state, indent=2))
    proc, attempted, confirmed_clean, response_received = None, False, False, False
    started = time.monotonic()
    try:
        target, payload = create_request(cfg)
        attempted = True
        code, response = api.call("POST", target, payload)  # Exactly one POST; no retry.
        response_received = True
        state["submission_http_status"] = code
        if code not in (200, 201):
            if code == 409:
                attempted = False  # Never delete a colliding pre-existing resource.
            raise RuntimeError(f"Trainer submission HTTP {code}: {response.get('message', '')}")
        if response.get("name") != resource:
            raise RuntimeError("Unexpected trainer ID; manual review required")
        state["status"] = "running"
        state_path.write_text(json.dumps(state, indent=2))
        env = os.environ.copy()
        env["FIREWORKS_API_KEY"] = key
        env["ROUTER_DEDICATED_WORKER"] = cfg["job_id"]
        with (run_dir / "worker.log").open("x") as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                    env=env, cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
            remaining = max(0, cfg["runtime_limit_seconds"] - (time.monotonic() - started))
            result = wait_worker(proc, remaining)
        if result != 0:
            raise RuntimeError(f"Training worker exited {result}; inspect the redacted worker log")
        worker = json.loads((run_dir / "worker-result.json").read_text())
        if worker.get("steps") != local_plan(cfg)["optimizer_steps"]:
            raise RuntimeError("Unexpected optimizer step count; review required")
        state["status"] = "training_complete"
    except BaseException as error:
        state["status"] = "stopped_before_completion"
        state["error"] = scrub(str(error), key)
        raise
    finally:
        try:
            stop_worker(proc)
        except Exception as error:
            state["worker_stop_error"] = scrub(str(error), key)
        # Remote cleanup must still run even if stopping the local process fails.
        if attempted:
            # A lost POST response can race a delayed create. Require several
            # consecutive absence checks; leave an explicit uncertainty record.
            confirmed_clean = cleanup(api, resource, absence_checks=1 if response_received else 3)
            if not response_received:
                state["submission_response_lost"] = True
                state["follow_up_status_check_required"] = True
        state["cleanup_confirmed"] = confirmed_clean
        state["elapsed_seconds"] = time.monotonic() - started
        state_path.write_text(json.dumps(state, indent=2))
        if not confirmed_clean:
            print("ACTION REQUIRED: trainer shutdown is unconfirmed. Check Fireworks now:", resource,
                  file=sys.stderr)
    if not confirmed_clean:
        raise RuntimeError("Training returned, but trainer shutdown is unconfirmed")
    print("Training finished and trainer is stopped/absent. Review results before inference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-runtime", action="store_true", help="Import/render with verified local tokenizer; no GPU")
    modes.add_argument("--launch", action="store_true", help="PAID: run only after human review and explicit approval")
    modes.add_argument("--cleanup", action="store_true", help="Stop ONLY the recorded trainer from this script")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = read_config()
    if args.worker:
        if os.environ.get("ROUTER_DEDICATED_WORKER") != cfg["job_id"]:
            raise ValueError("Worker must be started by the supervisor")
        key = load_key()
        sys.stdout, sys.stderr = SafeStream(sys.stdout, key), SafeStream(sys.stderr, key)
        recipe, options = runtime_recipe(cfg)
        result = recipe.main(options)
        (ROOT / cfg["run_dir"] / "worker-result.json").write_text(json.dumps(result, indent=2))
    elif args.cleanup:
        state = json.loads((ROOT / cfg["run_dir"] / "supervisor.json").read_text())
        if state.get("job_resource") != job_resource(cfg):
            raise ValueError("Cleanup ownership record does not match")
        if not cleanup(API(load_key()), job_resource(cfg)):
            raise RuntimeError("Shutdown unconfirmed; check Fireworks dashboard")
        print("Trainer stopped or absent.")
    elif args.launch:
        launch(cfg)
    else:
        print(json.dumps(local_plan(cfg), indent=2))
        if args.check_runtime:
            check_runtime(cfg)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        # Avoid unredacted SDK tracebacks, keys and signed URLs in console output.
        try:
            key = load_key()
        except Exception:
            key = ""
        print(scrub(f"{type(exc).__name__}: {exc}", key), file=sys.stderr)
        sys.exit(1)
