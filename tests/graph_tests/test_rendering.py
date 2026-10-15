"""Tests for ace.graph.rendering."""
from __future__ import annotations

from ace.graph.rendering import measure, render_file_view, render_view
from ace.graph.views import FileView, Level, View

MODELS = "pkg/models.py"


def _lines(text: str) -> list[str]:
    return text.split("\n")


def test_full_file_renders_every_line_with_prefixes(session):
    view = View()
    view.add_file("pkg/utils.py", Level.FULL)
    text = render_view(session, view)
    source = session.files["pkg/utils.py"].current_state.bytes.decode().rstrip("\n").split("\n")
    expected = ["--- pkg/utils.py ---"] + [f"{i}| {line}" for i, line in enumerate(source, 1)]
    assert _lines(text) == expected


def test_prefix_lines_false(session):
    view = View()
    view.add_file("notes.txt", Level.FULL)
    text = render_view(session, view, prefix_lines=False)
    assert text == "--- notes.txt ---\nline one\nline two\nline three\nline four"


def test_collapsed_file_is_a_skeleton(session):
    view = View()
    view.add_file(MODELS, Level.COLLAPSED)
    text = render_view(session, view)
    lines = _lines(text)
    assert lines[0] == f"--- {MODELS} ---"
    # top-level code outside elements in full
    assert "1| \"\"\"Data models.\"\"\"" in lines
    assert "2| import json" in lines
    assert "3| from pkg.utils import slugify" in lines
    # headers + docstrings
    assert "6| class Item:" in lines
    assert '7|     """An inventory item."""' in lines
    assert any(line.startswith("21| class Cart:") for line in lines)
    # bodies hidden
    assert "def slug" not in text
    assert "def __init__" not in text
    assert "    ... [BODY HIDDEN: 11 LINES] ..." in lines  # Item lines 8-18
    assert "    ... [BODY HIDDEN: 9 LINES] ..." in lines  # Cart lines 23-31


def test_nested_full_element_expands_in_place_with_ancestor_header(session):
    view = View()
    view.add_file(MODELS, Level.COLLAPSED)
    view.add_element(MODELS, "Item.slug", Level.FULL)
    lines = _lines(render_view(session, view))
    item = lines.index("6| class Item:")
    assert lines[item + 1] == '7|     """An inventory item."""'
    assert lines[item + 2] == "    ... [BODY HIDDEN: 5 LINES] ..."  # lines 8-12 (__init__)
    assert lines[item + 3 : item + 6] == [
        "13|     def slug(self):",
        '14|         """Return a URL slug."""',
        "15|         return slugify(self.name)",
    ]
    assert lines[item + 6] == "    ... [BODY HIDDEN: 3 LINES] ..."  # lines 16-18 (to_json)
    # Cart stays collapsed
    assert "    ... [BODY HIDDEN: 9 LINES] ..." in lines


def test_composition_rule_union_of_collapsed_file_and_full_element(session):
    collapsed_parent = View()
    collapsed_parent.add_file(MODELS, Level.COLLAPSED)
    element_parent = View()
    element_parent.add_element(MODELS, "Item.slug", Level.FULL)

    merged = View.union([collapsed_parent, element_parent])
    assert merged.files[MODELS].levels == {"": Level.COLLAPSED, "Item.slug": Level.FULL}

    expected = View()
    expected.add_file(MODELS, Level.COLLAPSED)
    expected.add_element(MODELS, "Item.slug", Level.FULL)
    text = render_view(session, merged)
    assert text == render_view(session, expected)
    # one skeleton section, not a separate element section
    assert text.count("--- ") == 1
    assert "15|         return slugify(self.name)" in text
    assert "3| from pkg.utils import slugify" in text
    assert "def to_json" not in text


def test_collapsed_element_inside_skeleton_shows_child_signatures(session):
    view = View()
    view.add_file(MODELS, Level.COLLAPSED)
    view.add_element(MODELS, "Cart", Level.COLLAPSED)
    lines = _lines(render_view(session, view))
    assert "24|     def __init__(self):" in lines
    assert "27|     def add(self, item):" in lines
    assert "30|     def total(self):" in lines
    assert not any("self.items.append" in line for line in lines)
    # Item untouched: no child signatures
    assert not any("def slug" in line for line in lines)


def test_element_only_view_renders_separate_sections(session):
    view = View()
    view.add_element(MODELS, "Item.slug", Level.FULL)
    view.add_element("pkg/utils.py", "clamp", Level.FULL)
    view.add_element("pkg/service.py", "checkout", Level.COLLAPSED)
    text = render_view(session, view)
    sections = text.split("\n\n")
    assert [s.split("\n")[0] for s in sections] == [
        f"--- {MODELS}::Item.slug ---",
        "--- pkg/service.py::checkout ---",
        "--- pkg/utils.py::clamp ---",
    ]
    assert sections[0].split("\n")[1:] == [
        "13|     def slug(self):",
        '14|         """Return a URL slug."""',
        "15|         return slugify(self.name)",
    ]
    # no top-level code when "" is absent
    assert "import" not in text
    assert "class Item" not in text
    checkout = sections[1].split("\n")
    assert checkout[1] == "6| def checkout(cart: Cart, discount: float = 0.0) -> float:"
    assert checkout[2] == '7|     """Apply a discount to the cart total."""'
    assert checkout[3] == "    ... [BODY HIDDEN: 2 LINES] ..."
    assert sections[2].split("\n")[1:] == [
        "12| def clamp(value, low, high):",
        "13|     return max(low, min(high, value))",
    ]


def test_element_only_nested_under_listed_ancestor_is_one_section(session):
    view = View()
    view.add_element(MODELS, "Item", Level.COLLAPSED)
    view.add_element(MODELS, "Item.slug", Level.FULL)
    text = render_view(session, view)
    assert text.count("--- ") == 1
    lines = _lines(text)
    assert lines[0] == f"--- {MODELS}::Item ---"
    assert "9|     def __init__(self, name, price):" in lines  # child signature
    assert "15|         return slugify(self.name)" in lines  # FULL child expanded
    assert "17|     def to_json(self):" in lines
    assert not any("json.dumps" in line for line in lines)


def test_line_ranges_on_unparsed_file(session):
    view = View()
    view.add_lines("notes.txt", 2, 3)
    view.add_lines("notes.txt", 10, 20)  # past EOF -> ignored when rendering
    assert render_view(session, view) == "--- notes.txt (lines 2-3) ---\n2| line two\n3| line three"


def test_unparsed_file_with_file_key_renders_fully(session):
    view = View()
    view.add_file("README.md", Level.COLLAPSED)
    view.add_lines("README.md", 1, 1)  # already shown -> no extra section
    text = render_view(session, view)
    assert text == (
        "--- README.md ---\n1| # Sample project\n2| \n3| A tiny project used in tests."
    )


def test_line_ranges_exclude_lines_already_shown(session):
    view = View()
    view.add_element(MODELS, "Item.slug", Level.FULL)  # lines 13-15
    view.add_lines(MODELS, 12, 17)
    text = render_view(session, view)
    sections = text.split("\n\n")
    headers = [s.split("\n")[0] for s in sections]
    assert headers == [
        f"--- {MODELS}::Item.slug ---",
        f"--- {MODELS} (lines 12-12) ---",
        f"--- {MODELS} (lines 16-17) ---",
    ]
    assert sections[2].split("\n")[1:] == [
        "16| ",
        "17|     def to_json(self):",
    ]


def test_line_ranges_inside_hidden_skeleton_body_are_rendered(session):
    view = View()
    view.add_file("pkg/utils.py", Level.COLLAPSED)
    view.add_lines("pkg/utils.py", 7, 9)
    text = render_view(session, view)
    assert text.endswith(
        '--- pkg/utils.py (lines 9-9) ---\n9|     return PATTERN.sub("-", text.lower()).strip("-")'
    )


def test_render_view_sorted_and_skips_missing(session):
    view = View()
    view.add_file("notes.txt", Level.FULL)
    view.add_file("README.md", Level.FULL)
    view.add_file("pkg/ghost.py", Level.FULL)
    view.add_element(MODELS, "Nope", Level.FULL)  # unknown element -> nothing rendered
    text = render_view(session, view)
    assert text.index("--- README.md ---") < text.index("--- notes.txt ---")
    assert "ghost" not in text
    assert MODELS not in text
    assert render_view(session, View()) == ""


def test_llm_path_keys_render_like_path_keys(session):
    slug = session.files[MODELS].code_nodes["Item.slug"]
    raw = FileView(MODELS, {slug.llm_path: Level.FULL})
    canonical = FileView(MODELS, {"Item.slug": Level.FULL})
    file = session.files[MODELS]
    assert render_file_view(file, raw) == render_file_view(file, canonical)


def test_render_is_deterministic(session):
    view = View()
    view.add_file(MODELS, Level.COLLAPSED)
    view.add_element(MODELS, "Cart.total", Level.FULL)
    view.add_lines("notes.txt", 1, 1)
    other = View.from_json(view.to_json())
    assert render_view(session, view) == render_view(session, other)


def test_measure_equals_len(session):
    view = View()
    view.add_file(MODELS, Level.COLLAPSED)
    view.add_element(MODELS, "Item.slug", Level.FULL)
    text = render_view(session, view)
    assert measure(text) == len(text)
    assert measure("") == 0
    assert measure("äb\n") == 3
