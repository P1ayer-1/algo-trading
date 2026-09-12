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


def test_no_source_file_carries_a_stray_control_character():
    """A bell or a form feed in a source file is a botched `\\a` or `\\f`.

    Both have appeared here from shell heredocs mangling Windows paths, and
    both are invisible in a diff.
    """
    for path in list(BACKEND.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        raw = path.read_bytes()
        for code, name in ((0x07, "bell"), (0x0c, "form feed"), (0x08, "backspace")):
            assert bytes([code]) not in raw, "{} in {}".format(name, path)
