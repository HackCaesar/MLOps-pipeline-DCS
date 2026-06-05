"""GPU sanity check via `docker run --gpus=all nvidia/cuda:... nvidia-smi`.

Run from host (NOT from inside a container). Returns 0 on success, non-zero
if Docker is missing, daemon not running, NVIDIA runtime not configured,
or GPU device isn't accessible.

This check feeds the project's GPU / CPU-fallback policy.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

DEFAULT_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"


def _print_remediation() -> None:
    print("", file=sys.stderr)
    print("To enable GPU in Docker on Windows:", file=sys.stderr)
    print("  1) Install NVIDIA driver on the Windows host.", file=sys.stderr)
    print("  2) Docker Desktop → Settings → Resources → WSL Integration → enable.", file=sys.stderr)
    print("  3) Install NVIDIA Container Toolkit (or rely on Docker Desktop 4.x+ which bundles support).", file=sys.stderr)
    print("  4) Verify: docker run --rm --gpus=all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi", file=sys.stderr)


def main(verbose: bool = False, image: str = DEFAULT_IMAGE) -> int:
    if shutil.which("docker") is None:
        print("FAIL: 'docker' CLI not found on PATH.", file=sys.stderr)
        _print_remediation()
        return 2

    cmd = ["docker", "run", "--rm", "--gpus=all", image, "nvidia-smi"]
    if verbose:
        print(f"$ {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        print("FAIL: docker executable disappeared between which() and run().", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print("FAIL: docker run timed out after 120s.", file=sys.stderr)
        return 3

    if proc.returncode != 0:
        print(f"FAIL: docker exited with code {proc.returncode}.", file=sys.stderr)
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        _print_remediation()
        return proc.returncode

    if verbose:
        print(proc.stdout)
    else:
        for line in proc.stdout.splitlines():
            if "NVIDIA-SMI" in line or "Driver Version" in line or "CUDA Version" in line:
                print(line)
    print("OK: GPU is accessible to Docker.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    args = p.parse_args()
    sys.exit(main(verbose=args.verbose, image=args.image))
