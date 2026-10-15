"""scripts/run_graph.py: rollouts go to --rollout-dir (default: <project>/.agent2/task)."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from fakes import FakeClient, text_message, tool_call_message

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_graph.py"

NEW_CLAMP = (
    "def clamp(value, low, high):\n"
    "    if low > high:\n"
    "        low, high = high, low\n"
    "    return max(low, min(high, value))\n"
)


def load_script():
    spec = importlib.util.spec_from_file_location("run_graph_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_factory(settings):
    def factory() -> FakeClient:
        return FakeClient([
            tool_call_message((
                "replace_element",
                json.dumps({"file_path": "pkg/utils.py", "element_path": "clamp",
                            "new_content": NEW_CLAMP}),
            )),
            text_message("Swapped reversed bounds."),
        ])

    return factory


def namespace(folder: Path, rollout_dir: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        folder=folder, prompt="fix clamp", graph=None, model="fake-model", api_base="http://x",
        api_key_env="FAKE_KEY", rollout_dir=rollout_dir, print_views=False,
    )


@pytest.mark.parametrize("explicit", [True, False])
def test_run_graph_rollout_dir(project_dir: Path, tmp_path: Path, monkeypatch, capsys, explicit):
    module = load_script()
    monkeypatch.setattr(module, "make_client_factory", fake_factory)
    rollouts = tmp_path / "rollouts" if explicit else None

    assert asyncio.run(module.run(namespace(project_dir, rollouts))) == 0

    expected = rollouts if explicit else project_dir / ".agent2" / "task"
    assert len(list(expected.glob("*.json"))) == 1
    assert (project_dir / ".agent2").exists() is (not explicit)
    assert "low, high = high, low" not in (project_dir / "pkg/utils.py").read_text()
    out = capsys.readouterr().out
    assert "diff --git a/pkg/utils.py b/pkg/utils.py" in out
    assert f"Rollouts saved to {expected.resolve()}" in out
