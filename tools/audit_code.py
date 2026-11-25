"""Statically audit maintained Python documentation without importing any module."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def audit(root: Path) -> dict:
    """Count every named function, including private helpers and nested functions.

    The result measures documentation presence, not correctness or test coverage.
    Frozen experiment code and external dependency directories are outside this
    explicitly maintained source/test/tool tree and are never imported here.
    """
    files = sorted(
        path for folder in ("src", "tests", "tools") for path in (root / folder).rglob("*.py")
    )
    missing = []
    syntax_errors = []
    function_count = 0
    documented = 0
    module_missing = []
    class_count = 0
    class_missing = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append({"file": relative, "line": exc.lineno, "message": exc.msg})
            continue
        if not ast.get_docstring(tree):
            module_missing.append(relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_count += 1
                if not ast.get_docstring(node):
                    class_missing.append(
                        {"file": relative, "class": node.name, "line": node.lineno}
                    )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
                doc = ast.get_docstring(node)
                if doc and len(doc.strip()) >= 12:
                    documented += 1
                else:
                    missing.append({"file": relative, "function": node.name, "line": node.lineno})
    return {
        "python_files": len(files),
        "named_functions": function_count,
        "documented_functions": documented,
        "classes": class_count,
        "missing_class_docstrings": class_missing,
        "missing_function_docstrings": missing,
        "missing_module_docstrings": module_missing,
        "syntax_errors": syntax_errors,
        "passed": not (missing or class_missing or module_missing or syntax_errors),
    }


def main(argv: list[str] | None = None) -> int:
    """Print the audit and optionally save its JSON result for reproducible checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(Path(__file__).resolve().parents[1])
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
