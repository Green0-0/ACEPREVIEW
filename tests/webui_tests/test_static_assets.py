"""Static-asset smoke tests for the ACE analysis WebUI frontend (recording spec §12).

No browser is needed: these check that every asset exists, that ``index.html`` references
all of them, that nothing is loaded from the network, that the frontend only calls API
paths from the §12.1 table, and that the backend serves the assets.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "ace" / "webui" / "static"

MODULES = ["app.js", "api.js", "ui.js", "charts.js", "graph.js", "diff.js", "chat.js"]
ASSETS = ["index.html", "style.css", "dev-mock.js", *MODULES]

# §12.1 endpoint table, path parameters written as {}.
SPEC_PATHS = {
    "/api/runs",
    "/api/runs/{}",
    "/api/runs/{}/progress",
    "/api/runs/{}/generations/{}",
    "/api/runs/{}/species",
    "/api/runs/{}/genomes",
    "/api/runs/{}/genomes/{}",
    "/api/runs/{}/genomes/{}/dot",
    "/api/runs/{}/evaluations",
    "/api/runs/{}/evaluations/{}/{}",
    "/api/runs/{}/evaluations/{}/{}/workers/{}/view",
    "/api/runs/{}/rollouts/{}/{}/{}",
    "/api/runs/{}/operators",
    "/api/runs/{}/compare",
    "/api/runs/{}/innovations",
    "/api/runs/{}/failures",
    "/api/health",
}


def read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def api_paths(source: str) -> set[str]:
    """API paths in string/template literals, with ${...} parameters normalised to {}."""
    found = set()
    literals = re.findall(r"`(/api/[^`]*)`", source) + re.findall(r"['\"](/api/[^'\"\s]*)['\"]", source)
    for literal in literals:
        literal = re.sub(r"\$\{qs\([^()]*\)\}", "", literal)  # query-string suffix
        literal = re.sub(r"\$\{[^}]*\}", "{}", literal)
        found.add(literal.split("?")[0])
    return found


@pytest.mark.parametrize("name", ASSETS)
def test_asset_exists_and_is_not_empty(name: str) -> None:
    path = STATIC / name
    assert path.is_file(), f"missing static asset {name}"
    assert path.stat().st_size > 200


def test_index_references_every_script_and_stylesheet() -> None:
    html = read("index.html")
    assert re.search(r'<link rel="stylesheet" href="/static/style\.css">', html)
    assert re.search(r'<script type="module" src="/static/app\.js"></script>', html)
    for module in MODULES:
        assert f'/static/{module}"' in html, f"index.html does not reference {module}"
    # the dev mock is loaded on demand by app.js only with ?mock=1
    assert "import('./dev-mock.js')" in read("app.js")
    for element in ('id="app"', 'id="toasts"', 'id="modal-root"', 'id="theme-toggle"', 'id="health-chip"'):
        assert element in html


def test_every_module_import_resolves() -> None:
    for name in [*MODULES, "dev-mock.js"]:
        for target in re.findall(r"""(?:from|import\()\s*['"](\./[^'"]+)['"]""", read(name)):
            assert (STATIC / target).is_file(), f"{name} imports missing {target}"


@pytest.mark.parametrize("name", ASSETS)
def test_no_external_asset_urls(name: str) -> None:
    source = read(name)
    # SVG/XML namespace identifiers are not fetched; everything else must be local.
    urls = [u for u in re.findall(r"https?://[^\s'\"`)]+", source) if not u.startswith("http://www.w3.org/")]
    if name == "dev-mock.js":
        urls = [u for u in urls if "api.anthropic.com" in u]  # fixture data, never fetched
        assert all("api.anthropic.com" in u for u in urls)
        return
    assert urls == [], f"{name} references external URLs: {urls}"
    assert not re.search(r"""(src|href)=["']//""", source)


def test_api_paths_match_spec_table() -> None:
    used = api_paths(read("api.js")) | api_paths(read("app.js"))
    assert used, "no API paths found"
    unknown = used - SPEC_PATHS
    assert not unknown, f"paths not in spec 12.1: {sorted(unknown)}"
    assert used == SPEC_PATHS, f"spec endpoints never used: {sorted(SPEC_PATHS - used)}"
    # views go through api.js only
    assert api_paths(read("app.js")) == set()


def test_routes_cover_spec_views() -> None:
    app = read("app.js")
    for fragment in ("generations", "species", "genomes", "genome", "eval", "operators", "failures", "compare", "innovations"):
        assert f"\\/{fragment}" in app, f"no hash route for {fragment}"
    assert "POLL_MS = 3000" in app and "live-badge" in app


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("name", [*MODULES, "dev-mock.js"])
def test_javascript_parses(name: str) -> None:
    result = subprocess.run(["node", "--check", str(STATIC / name)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_backend_serves_frontend(client) -> None:  # noqa: ANN001 - fixture from conftest
    index = client.get("/")
    assert index.status_code == 200
    assert "/static/app.js" in index.text
    for name in ASSETS:
        res = client.get(f"/static/{name}")
        assert res.status_code == 200, name
        if name.endswith(".js"):
            assert "javascript" in res.headers["content-type"], res.headers["content-type"]
        if name.endswith(".css"):
            assert "text/css" in res.headers["content-type"]
