"""
Lium pod connection, initialization, and lifecycle management.
"""
import logging
from pathlib import Path

from eval.pod import PodManager

logger = logging.getLogger("distillation.remote_validator")


def init_pod(lium, pod_name: str, eval_script: str, eval_script_remote: str,
             teacher_model: str) -> PodManager:
    """Initialize and connect to the Lium GPU pod.

    Connects, uploads eval script, and ensures dependencies are installed.
    Returns the connected PodManager instance.
    """
    print(f"[validator] Initializing Lium client...", flush=True)
    print(f"[validator] Connecting to pod '{pod_name}'...", flush=True)
    pod = PodManager(lium, pod_name=pod_name)
    pod.connect()
    print(f"[validator] Connected to pod: {pod.pod.name if pod.pod else '?'}", flush=True)

    print("[validator] Uploading eval script...", flush=True)
    pod.upload(eval_script, eval_script_remote, max_attempts=5)
    print("[validator] Eval script uploaded", flush=True)

    print("[validator] Ensuring pod dependencies...", flush=True)
    pod.ensure_dependencies(teacher_model)
    print("[validator] Pod dependencies ready", flush=True)

    return pod
