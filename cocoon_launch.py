#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from cocoon_ops import PACKAGE_DIR


AUTHORITY = PACKAGE_DIR / "mira_kite_authority.py"
DEFAULT_COCOON = PACKAGE_DIR / "cocoon_cognition_agency.py"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "mira_kite_authority_runtime"


def wait_for_url(url: str, timeout: float = 12.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= getattr(response, "status", 200) < 500:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def open_url(url: str) -> None:
    opener = os.environ.get("BROWSER") or "termux-open-url"
    candidates = [opener, "termux-open-url", "xdg-open"]
    for command in candidates:
        if not command:
            continue
        try:
            subprocess.run([command, url], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


def launch_background(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "authority_launch.log"
    pid_path = output_dir / "authority_launch.pid"
    cmd = [
        sys.executable,
        str(AUTHORITY),
        "--cocoon",
        str(Path(args.cocoon)),
        "--output-dir",
        str(output_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    log_file = log_path.open("ab")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PACKAGE_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    log_file.close()
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    url = f"http://{args.host}:{args.port}/"
    ready = wait_for_url(url, timeout=args.timeout)
    if not ready:
        print(f"[cocoon] launched background authority at {url} (pid {proc.pid}), but it is not responding yet.")
    else:
        print(f"[cocoon] launched authority at {url} (pid {proc.pid})")
    if not args.no_open:
        open_url(url)
    print(f"[cocoon] log: {log_path}")
    print(f"[cocoon] pid: {pid_path}")
    return 0


def launch_foreground(args: argparse.Namespace) -> int:
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(AUTHORITY),
            "--cocoon",
            str(Path(args.cocoon)),
            "--output-dir",
            str(Path(args.output_dir)),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the Cocoon Authority app from Termux.")
    parser.add_argument("--cocoon", default=str(DEFAULT_COCOON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args(argv)
    return launch_foreground(args) if args.foreground else launch_background(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
