#!/usr/bin/env python3
"""Serve the read-only ACE analysis WebUI over NEAT run directories.

Usage::

    uv run python scripts/run_webui.py [--runs-root .ace/neat] [--host 127.0.0.1]
        [--port 8766] [--no-browser]

Every directory under ``--runs-root`` containing ``run.json`` or ``generations.jsonl`` is a
run. Live runs are picked up while they are written (the UI polls).
"""
from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ACE analysis WebUI")
    parser.add_argument("--runs-root", type=Path, default=Path(".ace/neat"),
                        help="directory holding run directories (default .ace/neat)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    import uvicorn

    from ace.webui.app import create_app

    runs_root = args.runs_root.expanduser().resolve()
    app = create_app(runs_root)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"ACE WebUI: {url}  (runs root {runs_root}; Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
