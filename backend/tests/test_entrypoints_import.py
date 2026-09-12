"""Every entrypoint parses and exposes `main`.

This exists because a syntax error shipped in `plan_carry_xs.py` and the whole
suite stayed green: the tests import the strategy packages, and nothing
imported the command-line files that wrap them. A broken entrypoint is
invisible to a test suite that never loads it, and it is the only part of this
repo a person actually types.

Parsing rather than importing: an import would need the vendored SDK, the
network and credentials, and `conftest.py` deliberately provides none of those.
`ast.parse` catches the class of failure that actually happened - a heredoc
that turned `\\n` into a real newline inside a string literal - without needing
any of it.
"""

import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
ENTRYPOINTS = sorted(
    path for path in BACKEND.glob("*.py")
    if path.name not in {"config.py", "state.py", "server.py", "market_data.py",
                         "support_resistance.py"}
)


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: p.name)
def test_entrypoint_parses(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: p.name)
def test_entrypoint_defines_main(path):
    """Every one of these is run as `python backend\\<name>.py`, and the
    convention here is `raise SystemExit(main())` so an exit code means
    something to a scheduler."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "main" in names, path.name


def test_the_strategy_packages_parse_too():
    for path in (BACKEND / "trading" / "strategies").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


MANGLED = ((0x07, "bell", "a"), (0x08, "backspace", "b"),
           (0x0b, "vertical tab", "v"), (0x0c, "form feed", "f"))


def test_no_source_file_carries_a_stray_control_character():
    """A bell or a form feed in a source file is a botched `\\a` or `\\f`.

    Both have appeared here from shell heredocs mangling Windows paths, and
    both are invisible in a diff.
    """
    for path in list(BACKEND.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        raw = path.read_bytes()
        for code, name, letter in MANGLED:
            assert bytes([code]) not in raw, "{} in {} (a mangled backslash-{})".format(
                name, path, letter)


def test_the_prose_files_are_not_mangled_either():
    """The same corruption, in the files that are actually read.

    The check above only ever scanned `backend/**/*.py`, and the damage had
    been accumulating for months in README.md - which is this project's lab
    notebook and the most-read file in it. Found 2026-09-12: 7 bells, 4 tabs,
    2 vertical tabs, 1 form feed and 1 backspace, every one of them a Windows
    path a shell heredoc had eaten. `backend\\analysis\\touch_calibration.py`
    had been rendered as `backend<BEL>nalysis<TAB>ouch_calibration.py`, which
    is not a path anyone can type.

    A tab is legal in most prose, so it is only a finding HERE: nothing in
    these files is meant to be tab-indented, and every tab yet seen in them was
    a `\\t` that used to be part of `backend\\trading`.
    """
    root = BACKEND.parent
    for name in ("README.md", "CLAUDE.md"):
        path = root / name
        if not path.exists():
            continue
        raw = path.read_bytes()
        for code, label, letter in MANGLED + ((0x09, "tab", "t"),):
            assert bytes([code]) not in raw, \
                "{} in {} (a mangled backslash-{})".format(label, name, letter)
