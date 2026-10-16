from ace.swebench.proxy import localization_scores, parse_patch, patch_similarity

GOLD = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -10,7 +10,8 @@ def f(x):
     a = 1
     b = 2
-    return x + a
+    if x is None:
+        return a
+    return x + a + b
     c = 3
     d = 4
"""

PARTIAL = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -10,7 +10,7 @@ def f(x):
     a = 1
     b = 2
-    return x + a
+    return x + a + b
     c = 3
     d = 4
"""

WRONG_FILE = """diff --git a/pkg/other.py b/pkg/other.py
--- a/pkg/other.py
+++ b/pkg/other.py
@@ -1,3 +1,3 @@
 import os
-print("hello")
+print("bye")
 x = 1
"""

TEST_ONLY = """diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,3 @@
 def test_f():
-    assert True
+    assert f(None) == 1
+    assert f(1) == 4
"""

TWO_FILES = GOLD + WRONG_FILE


def test_parse():
    p = parse_patch(TWO_FILES)
    assert set(p) == {"pkg/core.py", "pkg/other.py"}
    assert p["pkg/core.py"].n_changed == 4
    assert p["pkg/core.py"].old_changed[0] == 12


def test_identical_is_one():
    assert patch_similarity(GOLD, GOLD) == 1.0


def test_whitespace_insensitive():
    assert patch_similarity(GOLD.replace("x + a + b", "x  +  a + b"), GOLD) == 1.0


def test_empty_is_zero():
    assert patch_similarity("", GOLD) == 0.0
    assert patch_similarity("not a diff at all", GOLD) == 0.0


def test_wrong_file_low():
    assert patch_similarity(WRONG_FILE, GOLD) == 0.0
    assert patch_similarity(TEST_ONLY, GOLD) == 0.0


def test_partial_between():
    s = patch_similarity(PARTIAL, GOLD)
    assert 0.5 < s < 1.0
    s2 = patch_similarity(TWO_FILES, GOLD)  # right fix plus an extra file
    assert 0.3 < s2 < 1.0
    assert patch_similarity(PARTIAL, GOLD) > patch_similarity(WRONG_FILE, GOLD)


def test_localization():
    d = localization_scores(PARTIAL, GOLD)
    assert d["file_f1"] == 1.0
    assert d["hunk_overlap"] == 1.0
    d = localization_scores(WRONG_FILE, GOLD)
    assert d["file_f1"] == 0.0 and d["hunk_overlap"] == 0.0
