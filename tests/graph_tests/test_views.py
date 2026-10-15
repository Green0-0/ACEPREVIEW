"""Tests for ace.graph.views."""
from __future__ import annotations

import json

from ace.graph.views import FileView, Level, View, merge_ranges, normalize_element_path


def test_level_ordering():
    assert Level.FULL > Level.COLLAPSED
    assert int(Level.FULL) == 2 and int(Level.COLLAPSED) == 1


def test_merge_ranges_sorts_and_coalesces_adjacent_and_overlapping():
    assert merge_ranges([(10, 12), (4, 6), (1, 3), (5, 8), (14, 14)]) == [(1, 8), (10, 12), (14, 14)]
    assert merge_ranges([]) == []


def test_add_file_and_element_keep_max_level():
    view = View()
    view.add_file("a.py", Level.FULL)
    view.add_file("a.py", Level.COLLAPSED)
    view.add_element("a.py", "A.f", Level.COLLAPSED)
    view.add_element("a.py", "A.f", Level.FULL)
    view.add_element("a.py", "A.f", Level.COLLAPSED)
    assert view.files["a.py"].levels == {"": Level.FULL, "A.f": Level.FULL}


def test_add_element_strips_llm_path_line_suffix():
    view = View()
    view.add_element("a.py", "MyClass.my_function.42", Level.FULL)
    view.add_element("a.py", "", Level.COLLAPSED)
    assert view.files["a.py"].levels == {"MyClass.my_function": Level.FULL, "": Level.COLLAPSED}


def test_add_lines_merges_and_is_lenient():
    view = View()
    view.add_lines("n.txt", 5, 7)
    view.add_lines("n.txt", 1, 2)
    view.add_lines("n.txt", 3, 4)  # adjacent on both sides -> one range
    view.add_lines("n.txt", 12, 10)  # reversed -> swapped
    view.add_lines("other.txt", -3, 0)  # entirely before line 1 -> ignored
    assert view.files["n.txt"].line_ranges == [(1, 7), (10, 12)]
    assert "other.txt" not in view.files


def test_merge_takes_max_level_and_unions_ranges():
    a = View()
    a.add_file("pkg/models.py", Level.COLLAPSED)
    a.add_element("pkg/models.py", "Item", Level.FULL)
    a.add_lines("notes.txt", 1, 2)
    b = View()
    b.add_file("pkg/models.py", Level.FULL)
    b.add_element("pkg/models.py", "Item", Level.COLLAPSED)
    b.add_element("pkg/models.py", "Cart", Level.COLLAPSED)
    b.add_lines("notes.txt", 2, 3)
    b.add_lines("notes.txt", 6, 6)

    merged = a.merge(b)
    models = merged.files["pkg/models.py"]
    assert models.levels == {"": Level.FULL, "Item": Level.FULL, "Cart": Level.COLLAPSED}
    assert merged.files["notes.txt"].line_ranges == [(1, 3), (6, 6)]
    # inputs untouched
    assert a.files["pkg/models.py"].levels[""] == Level.COLLAPSED
    assert a.files["notes.txt"].line_ranges == [(1, 2)]
    assert "Cart" not in a.files["pkg/models.py"].levels
    # merged is independent of its inputs
    merged.add_lines("notes.txt", 10, 11)
    assert b.files["notes.txt"].line_ranges == [(2, 3), (6, 6)]


def test_union_of_many_views():
    views = []
    for index in range(3):
        view = View()
        view.add_lines("notes.txt", index + 1, index + 1)
        views.append(view)
    collapsed = View()
    collapsed.add_file("pkg/utils.py", Level.COLLAPSED)
    views.append(collapsed)
    result = View.union(views)
    assert result.files["notes.txt"].line_ranges == [(1, 3)]
    assert result.files["pkg/utils.py"].levels == {"": Level.COLLAPSED}
    assert View.union([]).is_empty()


def test_counts_weighting():
    view = View()
    view.add_file("pkg/models.py", Level.COLLAPSED)
    view.add_element("pkg/models.py", "Item.slug", Level.FULL)
    view.add_element("pkg/models.py", "Cart", Level.COLLAPSED)
    view.add_lines("pkg/models.py", 1, 3)
    view.add_lines("pkg/models.py", 30, 31)
    view.add_lines("notes.txt", 1, 2)
    view.add_file("pkg/utils.py", Level.FULL)
    assert view.counts() == {
        ("pkg/models.py", ""): 1 + 2,  # COLLAPSED file + two line ranges
        ("pkg/models.py", "Item.slug"): 2,
        ("pkg/models.py", "Cart"): 1,
        ("notes.txt", ""): 1,
        ("pkg/utils.py", ""): 2,
    }


def test_counts_sum_across_parents_by_consumer():
    parents = []
    for level in (Level.FULL, Level.COLLAPSED, Level.FULL):
        view = View()
        view.add_element("pkg/models.py", "Item", level)
        parents.append(view)
    seeds: dict[tuple[str, str], int] = {}
    for view in parents:
        for key, value in view.counts().items():
            seeds[key] = seeds.get(key, 0) + value
    assert seeds == {("pkg/models.py", "Item"): 5}


def test_is_empty():
    assert View().is_empty()
    assert View({"a.py": FileView("a.py")}).is_empty()
    view = View()
    view.add_lines("a.py", 1, 1)
    assert not view.is_empty()


def test_json_round_trip():
    view = View()
    view.add_file("pkg/models.py", Level.COLLAPSED)
    view.add_element("pkg/models.py", "Item.slug", Level.FULL)
    view.add_lines("pkg/models.py", 3, 4)
    view.add_lines("notes.txt", 1, 2)
    view.add_lines("notes.txt", 4, 4)
    data = view.to_json()
    text = json.dumps(data)  # JSON-serialisable
    restored = View.from_json(json.loads(text))
    assert restored == view
    assert restored.files["pkg/models.py"].levels["Item.slug"] is Level.FULL
    assert restored.files["notes.txt"].line_ranges == [(1, 2), (4, 4)]
    assert all(isinstance(r, tuple) for r in restored.files["notes.txt"].line_ranges)
    assert View.from_json({"files": {"a.py": {"levels": {"": "FULL"}}}}).files["a.py"].levels == {
        "": Level.FULL
    }


def test_normalize_element_path(session):
    models = session.files["pkg/models.py"]
    slug = models.code_nodes["Item.slug"]
    assert normalize_element_path(models, "Item.slug") == "Item.slug"
    assert normalize_element_path(models, slug.llm_path) == "Item.slug"
    assert normalize_element_path(models, "Item.missing") is None
    assert normalize_element_path(models, "") == ""
    assert normalize_element_path(session.files["notes.txt"], "anything") is None


def test_restrict_to_drops_unknown_files_and_elements(session):
    slug_llm = session.files["pkg/models.py"].code_nodes["Item.slug"].llm_path
    view = View()
    view.add_file("pkg/models.py", Level.COLLAPSED)
    view.files["pkg/models.py"].levels[slug_llm] = Level.FULL  # raw llm_path key
    view.add_element("pkg/models.py", "Item.nope", Level.FULL)
    view.add_element("pkg/models.py", "Ghost", Level.COLLAPSED)
    view.add_file("pkg/missing.py", Level.FULL)
    view.add_element("notes.txt", "fake", Level.FULL)  # elements on unparsed file
    view.add_lines("notes.txt", 3, 99)  # clamped to 4 lines
    view.add_lines("README.md", 50, 60)  # entirely past EOF -> dropped
    view.add_element("README.md", "fake", Level.FULL)

    restricted = view.restrict_to(session)
    assert set(restricted.files) == {"pkg/models.py", "notes.txt"}
    assert restricted.files["pkg/models.py"].levels == {"": Level.COLLAPSED, "Item.slug": Level.FULL}
    assert restricted.files["notes.txt"].levels == {}
    assert restricted.files["notes.txt"].line_ranges == [(3, 4)]
    # original untouched
    assert "pkg/missing.py" in view.files
