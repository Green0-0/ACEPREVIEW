#!/usr/bin/env python3
"""Serve the ACE analysis WebUI over NEAT run directories.

Usage::

    uv run python scripts/run_webui.py [--runs-root .ace/neat] [--host 127.0.0.1]
        [--port 8766] [--no-browser] [--demo-project DIR] [--demo-model MODEL] [--no-demo]

Every directory under ``--runs-root`` containing ``run.json`` or ``generations.jsonl`` is a
run. Live runs are picked up while they are written (the UI polls).

A genome page can run its graph on a demo project with a prompt typed in the browser
(``--demo-project``, default ``agent2/smoke_repos/schedule``). The project is not modified;
only worker rollouts are written, into a temporary directory. The model (default
``claude-haiku-5-5`` with thinking off) reads its key from ``ANTHROPIC_API_KEY`` or from
``.env`` at the project root.
"""
from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    from ace.webui.demo import DEFAULT_MODEL, default_project_dir

    parser = argparse.ArgumentParser(description="Run the ACE analysis WebUI")
    parser.add_argument("--runs-root", type=Path, default=Path(".ace/neat"),
                        help="directory holding run directories (default .ace/neat)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--demo-project", type=Path, default=default_project_dir(),
                        help="project folder demo runs operate on (default agent2/smoke_repos/schedule)")
    parser.add_argument("--demo-model", default=DEFAULT_MODEL,
                        help=f"model for demo runs (default {DEFAULT_MODEL})")
    parser.add_argument("--no-demo", action="store_true", help="disable demo runs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    import uvicorn

    from ace.graph.llm import load_env_key
    from ace.webui.app import create_app
    from ace.webui.demo import DemoSettings

    runs_root = args.runs_root.expanduser().resolve()
    if args.no_demo or args.demo_project is None:
        demo = DemoSettings(project_dir=None)
        demo_note = "demo runs disabled"
    else:
        demo = DemoSettings(project_dir=args.demo_project.expanduser().resolve(), model=args.demo_model)
        key_set = load_env_key(demo.api_key_env_var or "", ROOT, ROOT / "agent2") if demo.api_key_env_var else True
        demo_note = (f"demo runs on {demo.project_dir} with {demo.model}; "
                     f"{demo.api_key_env_var} {'set' if key_set else 'NOT set (demo runs will refuse to start)'}")
    app = create_app(runs_root, demo)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"ACE WebUI: {url}  (runs root {runs_root}; {demo_note}; Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
