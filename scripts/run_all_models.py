"""Run both nested-CV experiments concurrently, one process per GPU."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    ("VXM", "VoxelMorph", "voxelmorph", "cuda:0"),
    ("PROPOSED", "Proposed", "cnn_transformer_svf_2d", "cuda:1"),
)


def _stream_output(prefix: str, process: subprocess.Popen[str]) -> None:
    """Prefix a child's console stream; training remains in the child process."""
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def _terminate_all(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit(
            "Two CUDA GPUs are required: torch.cuda.is_available() must be true "
            "and torch.cuda.device_count() must be at least 2."
        )

    print("=" * 50)
    print("Starting parallel nested CV")
    print("=" * 50)
    print("VoxelMorph -> cuda:0")
    print("Proposed   -> cuda:1\n")
    print("Search:")
    print("  trials        = 15")
    print("  search epochs = 10")
    print("  parameters    = learning_rate, lambda_smooth, int_steps\n")
    print("Fixed:")
    print("  batch_size    = 8\n")
    print("Logs:")
    print("  outputs/logs/voxelmorph_cuda0.log")
    print("  outputs/logs/proposed_cuda1.log")
    print("=" * 50, flush=True)

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    processes: list[subprocess.Popen[str]] = []
    readers: list[threading.Thread] = []
    try:
        for tag, _display, model, device in EXPERIMENTS:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "nested_cv.py"),
                "--model", model,
                "--devices", device,
            ]
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=False,
                creationflags=creationflags,
            )
            processes.append(process)
            reader = threading.Thread(
                target=_stream_output, args=(tag, process), daemon=True
            )
            reader.start()
            readers.append(reader)

        exit_codes = [process.wait() for process in processes]
        for reader in readers:
            reader.join()
    except KeyboardInterrupt:
        print("\nInterrupt received; terminating both experiments...", flush=True)
        _terminate_all(processes)
        for reader in readers:
            reader.join(timeout=2)
        return 130

    print("=" * 50)
    print("Experiment summary")
    print("=" * 50)
    for (_tag, display, _model, device), code in zip(EXPERIMENTS, exit_codes):
        status = "SUCCESS" if code == 0 else f"FAILED (exit code {code})"
        print(f"{display:<11} [{device}] -> {status}")
    print("=" * 50)
    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
