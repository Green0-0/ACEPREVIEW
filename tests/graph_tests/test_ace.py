"""Tests for ace.graph.ace (GRAPH_NEAT_DESIGN.md section 4.1)."""
from __future__ import annotations

import ast

import pytest

from ace.graph.ace import (
    AceBlock,
    AceError,
    AceFormat,
    AceTarget,
    ace_instructions,
    apply_ace,
    parse_ace,
)
from ace.graph.types import GraphFailure

ALL = AceFormat.ELEMENT_FILE_LINES


def block(header: str, body: str = "", lang: str = "") -> str:
    if body and not body.endswith("\n"):
        body += "\n"
    return f"```{lang}\n@@ {header}\n{body}```\n"


def run(session, text: str, fmt: AceFormat = ALL) -> dict[str, bytes | None]:
    return apply_ace(parse_ace(text, fmt), session, fmt).files


def models(files) -> str:
    return files["pkg/models.py"].decode()


# --------------------------------------------------------------------------- parsing

@pytest.mark.parametrize(
    ("header", "target", "element", "lines"),
    [
        ("pkg/models.py::Item.slug", AceTarget.ELEMENT, "Item.slug", None),
        ("pkg/models.py::Item.slug.14", AceTarget.ELEMENT, "Item.slug.14", None),
        ("pkg/models.py", AceTarget.FILE, None, None),
        ("pkg/extra.py::NEW", AceTarget.NEW, None, None),
        ("pkg/utils.py::DELETE", AceTarget.DELETE, None, None),
        ("notes.txt::2-3", AceTarget.LINES, None, (2, 3)),
        ("notes.txt::4", AceTarget.LINES, None, (4, 4)),
        ("pkg/models.py::Cart.total::AFTER", AceTarget.AFTER, "Cart.total", None),
        ("pkg/models.py::Cart::BEFORE", AceTarget.BEFORE, "Cart", None),
    ],
)
def test_parse_header_kinds(header, target, element, lines):
    (parsed,) = parse_ace(block(header, "x = 1\n"), ALL)
    assert parsed.target is target
    assert parsed.element == element
    assert parsed.lines == lines
    assert parsed.path == header.split("::")[0]
    assert parsed.body == "x = 1\n"


def test_parse_ignores_language_tag_and_non_ace_blocks():
    text = (
        "Some prose.\n```python\nprint('example')\n```\n"
        + block("pkg/utils.py::clamp", "def clamp(v, lo, hi):\n    return v\n", lang="python")
        + "More prose.\n```\nnot a header\n```\n"
    )
    (parsed,) = parse_ace(text, AceFormat.ELEMENT)
    assert parsed.target is AceTarget.ELEMENT
    assert parsed.body == "def clamp(v, lo, hi):\n    return v\n"


def test_parse_zero_blocks_raises():
    with pytest.raises(AceError):
        parse_ace("No code here.\n```\nprint(1)\n```\n", ALL)
    with pytest.raises(AceError):
        parse_ace("", ALL)
    assert issubclass(AceError, GraphFailure)


def test_parse_empty_body_and_multiple_blocks():
    text = block("pkg/utils.py::DELETE") + block("notes.txt::1-1", "first\n")
    first, second = parse_ace(text, ALL)
    assert first.body == "" and first.target is AceTarget.DELETE
    assert second.lines == (1, 1)


@pytest.mark.parametrize(
    ("fmt", "header"),
    [
        (AceFormat.ELEMENT, "pkg/models.py"),
        (AceFormat.ELEMENT, "notes.txt::1-2"),
        (AceFormat.ELEMENT_FILE, "notes.txt::1-2"),
        (AceFormat.FILE_LINES, "pkg/models.py::Item.slug"),
        (AceFormat.FILE_LINES, "pkg/models.py::Cart::AFTER"),
        (AceFormat.FILE_LINES, "pkg/models.py::Cart::BEFORE"),
    ],
)
def test_format_gating_rejects(fmt, header):
    with pytest.raises(AceError, match=header.replace(".", r"\.")):
        parse_ace(block(header, "x\n"), fmt)


@pytest.mark.parametrize(
    ("fmt", "header"),
    [
        (AceFormat.ELEMENT, "pkg/models.py::Item.slug"),
        (AceFormat.ELEMENT, "pkg/new.py::NEW"),
        (AceFormat.ELEMENT, "pkg/utils.py::DELETE"),
        (AceFormat.ELEMENT_FILE, "pkg/models.py"),
        (AceFormat.ELEMENT_FILE_LINES, "notes.txt::1-2"),
        (AceFormat.FILE_LINES, "pkg/models.py"),
        (AceFormat.FILE_LINES, "notes.txt::1-2"),
        (AceFormat.FILE_LINES, "pkg/new.py::NEW"),
    ],
)
def test_format_gating_accepts(fmt, header):
    assert len(parse_ace(block(header, "x\n"), fmt)) == 1


def test_apply_rechecks_format(session):
    blocks = [AceBlock("pkg/models.py", AceTarget.FILE, None, None, "x = 1\n")]
    with pytest.raises(AceError):
        apply_ace(blocks, session, AceFormat.ELEMENT)


@pytest.mark.parametrize("header", ["pkg/models.py::Item::SIDEWAYS", "pkg/models.py::3-1", "::Item"])
def test_malformed_headers(header):
    with pytest.raises(AceError):
        parse_ace(block(header, "x\n"), ALL)


def test_unclosed_header_block_raises():
    with pytest.raises(AceError):
        parse_ace("```\n@@ pkg/models.py::Item.slug\ndef slug(self):\n    pass\n", ALL)


# --------------------------------------------------------------------------- application

def test_replace_nested_method_reindents(session):
    body = 'def slug(self):\n    """Upper slug."""\n    return slugify(self.name).upper()\n'
    files = run(session, block("pkg/models.py::Item.slug", body, lang="python"))
    text = models(files)
    assert '    def slug(self):\n        """Upper slug."""\n        return slugify(self.name).upper()\n\n    def to_json' in text
    assert "Return a URL slug" not in text
    ast.parse(text)
    assert set(files) == {"pkg/models.py"}


def test_replace_accepts_llm_path_and_already_indented_body(session):
    body = '    def slug(self):\n        return "x"\n'
    llm_path = session.files["pkg/models.py"].code_nodes["Item.slug"].llm_path
    text = models(run(session, block(f"pkg/models.py::{llm_path}", body)))
    assert '    def slug(self):\n        return "x"\n' in text
    ast.parse(text)


def test_after_last_method_stays_inside_class(session):
    body = "def count(self):\n    return len(self.items)\n"
    text = models(run(session, block("pkg/models.py::Cart.total::AFTER", body)))
    assert text.endswith(
        "        return sum(item.price for item in self.items)\n\n"
        "    def count(self):\n        return len(self.items)\n"
    )
    tree = ast.parse(text)
    cart = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Cart")
    assert [f.name for f in cart.body if isinstance(f, ast.FunctionDef)] == ["__init__", "add", "total", "count"]


def test_before_inserts_sibling(session):
    body = "def reset(self):\n    self.items = []\n"
    text = models(run(session, block("pkg/models.py::Cart.add::BEFORE", body)))
    assert "    def reset(self):\n        self.items = []\n\n    def add(self, item):" in text
    cart = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "Cart")
    assert [f.name for f in cart.body if isinstance(f, ast.FunctionDef)] == ["__init__", "reset", "add", "total"]


def test_top_level_after_uses_two_blank_lines(session):
    files = run(session, block("pkg/utils.py::slugify::AFTER", "def noop():\n    pass\n"))
    text = files["pkg/utils.py"].decode()
    assert 'strip("-")\n\n\ndef noop():\n    pass\n\n\ndef clamp' in text
    ast.parse(text)


def test_new_file(session):
    files = run(session, block("pkg/extra.py::NEW", "VALUE = 2\n"), AceFormat.ELEMENT)
    assert files == {"pkg/extra.py": b"VALUE = 2\n"}


def test_new_on_existing_path_raises(session):
    with pytest.raises(AceError, match="already exists"):
        run(session, block("pkg/utils.py::NEW", "x = 1\n"))


def test_delete_file(session):
    assert run(session, block("pkg/utils.py::DELETE"), AceFormat.ELEMENT) == {"pkg/utils.py": None}


@pytest.mark.parametrize(
    "header",
    ["pkg/missing.py::DELETE", "pkg/missing.py::foo", "pkg/missing.py", "pkg/missing.py::1-2",
     "pkg/models.py::Item.nope", "notes.txt::line", "notes.txt::9-10"],
)
def test_unknown_targets_raise(session, header):
    with pytest.raises(AceError):
        run(session, block(header, "x\n"))


def test_whole_file_replacement(session):
    files = run(session, block("README.md", "# New\n"), AceFormat.ELEMENT_FILE)
    assert files == {"README.md": b"# New\n"}


def test_line_span_on_unparsed_file(session):
    files = run(session, block("notes.txt::2-3", "LINE TWO\nLINE THREE"), AceFormat.FILE_LINES)
    assert files["notes.txt"] == b"line one\nLINE TWO\nLINE THREE\nline four\n"


def test_line_span_appends_missing_newline(session):
    blocks = [AceBlock("notes.txt", AceTarget.LINES, None, (2, 2), "TWO")]
    files = apply_ace(blocks, session, AceFormat.FILE_LINES).files
    assert files["notes.txt"] == b"line one\nTWO\nline three\nline four\n"


def test_line_span_last_line_and_deletion(session):
    files = run(session, block("notes.txt::4", "end\n") + block("notes.txt::1-1"), AceFormat.FILE_LINES)
    assert files["notes.txt"] == b"line two\nline three\nend\n"


def test_overlapping_blocks_raise(session):
    text = block("pkg/models.py::Cart", "class Cart:\n    pass\n") + block(
        "pkg/models.py::Cart.total", "def total(self):\n    return 0\n"
    )
    with pytest.raises(AceError, match="Overlapping"):
        run(session, text)


def test_overlapping_lines_and_element_raise(session):
    # lines 13-14 are inside Item.slug
    text = block("pkg/models.py::13-14", "    pass\n") + block(
        "pkg/models.py::Item.slug", "def slug(self):\n    return 1\n"
    )
    with pytest.raises(AceError, match="Overlapping"):
        run(session, text)


def test_insert_into_replaced_ancestor_raises(session):
    text = block("pkg/models.py::Cart", "class Cart:\n    pass\n") + block(
        "pkg/models.py::Cart.total::AFTER", "def count(self):\n    return 0\n"
    )
    with pytest.raises(AceError, match="Overlapping"):
        run(session, text)


def test_file_and_element_conflict_raises(session):
    text = block("pkg/models.py", "x = 1\n") + block(
        "pkg/models.py::Item.slug", "def slug(self):\n    return 1\n"
    )
    with pytest.raises(AceError, match="Conflicting"):
        run(session, text)


def test_two_whole_file_blocks_conflict(session):
    text = block("pkg/utils.py::DELETE") + block("pkg/utils.py", "x = 1\n")
    with pytest.raises(AceError, match="Conflicting"):
        run(session, text)


def test_two_elements_same_file_applied_bottom_up(session):
    text = (
        block("pkg/models.py::Item.slug", "def slug(self):\n    return 'item'\n")
        + block("pkg/models.py::Cart.total", "def total(self):\n    return 42\n")
        + block("pkg/models.py::Cart.total::AFTER", "def empty(self):\n    return not self.items\n")
        + block("pkg/models.py::Item::BEFORE", "BASE = 1\n")
    )
    text_out = models(run(session, text, AceFormat.ELEMENT))
    assert "    def slug(self):\n        return 'item'\n" in text_out
    assert "    def total(self):\n        return 42\n\n    def empty(self):\n        return not self.items\n" in text_out
    assert "BASE = 1\n\n\nclass Item:" in text_out
    tree = ast.parse(text_out)
    cart = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Cart")
    assert [f.name for f in cart.body if isinstance(f, ast.FunctionDef)] == ["__init__", "add", "total", "empty"]


def test_replace_and_before_same_element(session):
    text = block("pkg/utils.py::clamp", "def clamp(v, lo, hi):\n    return v\n") + block(
        "pkg/utils.py::clamp::BEFORE", "def first():\n    pass\n"
    )
    out = run(session, text)["pkg/utils.py"].decode()
    assert out.endswith("def first():\n    pass\n\n\ndef clamp(v, lo, hi):\n    return v\n")


def test_multiple_files_and_diff(session):
    text = block("pkg/utils.py::clamp", "def clamp(v, lo, hi):\n    return v\n") + block(
        "notes.txt::1-1", "LINE ONE\n"
    ) + block("pkg/service.py::DELETE")
    edit = apply_ace(parse_ace(text, ALL), session, ALL)
    assert edit.touched_paths() == {"pkg/utils.py", "notes.txt", "pkg/service.py"}
    # Same files as the materialised workspace's diff; the content is the faithful git
    # form (Workspace.diff drops terminators, so the texts themselves differ).
    diffs = dict(edit.per_file_diff(session))
    assert sorted(diffs) == [path for path, _ in edit.to_workspace(session).diff()]
    assert "-line one\n+LINE ONE\n" in diffs["notes.txt"]
    assert session.files["notes.txt"].current_state.bytes.startswith(b"line one")  # pristine kept


# --------------------------------------------------------------------------- instructions

def test_instructions_list_only_permitted_headers():
    element = ace_instructions(AceFormat.ELEMENT)
    assert "::Class.method::AFTER" in element and "::NEW" in element and "::DELETE" in element
    assert "::12-40" not in element and "`@@ path/to/file.py`" not in element
    assert "signature" in element and "disjoint" in element
    assert "discouraged" not in element

    full = ace_instructions(AceFormat.ELEMENT_FILE_LINES)
    assert "::12-40" in full and "`@@ path/to/file.py`" in full and "discouraged" in full

    file_lines = ace_instructions(AceFormat.FILE_LINES)
    assert "::12-40" in file_lines and "Class.method" not in file_lines.split("Example:")[0]
    for fmt in AceFormat:
        assert "```" in ace_instructions(fmt)


def test_instruction_examples_parse():
    for fmt in AceFormat:
        example = ace_instructions(fmt).split("Example:\n", 1)[1]
        assert len(parse_ace(example, fmt)) == 1
