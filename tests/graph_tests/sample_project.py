"""Shared sample project + fixtures used by graph and neat tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SAMPLE_FILES = {
    "pkg/__init__.py": "",
    "pkg/models.py": '''"""Data models."""
import json
from pkg.utils import slugify


class Item:
    """An inventory item."""

    def __init__(self, name, price):
        self.name = name
        self.price = price

    def slug(self):
        """Return a URL slug."""
        return slugify(self.name)

    def to_json(self):
        return json.dumps({"name": self.name, "price": self.price})


class Cart:
    """A shopping cart."""

    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def total(self):
        return sum(item.price for item in self.items)
''',
    "pkg/utils.py": '''"""Utilities."""
import re

PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(text):
    """Lowercase and dash-join the text."""
    return PATTERN.sub("-", text.lower()).strip("-")


def clamp(value, low, high):
    return max(low, min(high, value))
''',
    "pkg/service.py": '''"""Service layer."""
from pkg.models import Cart, Item
from pkg.utils import clamp


def checkout(cart: Cart, discount: float = 0.0) -> float:
    """Apply a discount to the cart total."""
    discount = clamp(discount, 0.0, 1.0)
    return cart.total() * (1 - discount)


def build_cart(names):
    cart = Cart()
    for name in names:
        cart.add(Item(name, 1.0))
    return cart
''',
    "README.md": "# Sample project\n\nA tiny project used in tests.\n",
    "notes.txt": "line one\nline two\nline three\nline four\n",
}


def write_sample_project(root: Path) -> Path:
    for relative, content in SAMPLE_FILES.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    return write_sample_project(tmp_path / "project")


@pytest.fixture
def session(project_dir: Path):
    from agent2.core.session import Session

    return Session(str(project_dir))
