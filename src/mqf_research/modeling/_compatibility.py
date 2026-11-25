"""Explicit boundary for immutable, provenance-bound historical research helpers.

The maintained engines are self-contained. Input reconstruction and independent
statistics still reuse historical helpers so their scientific definitions do
not change merely as a consequence of the maintenance work.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def activate_legacy_root(root: str | Path) -> None:
    """Resolve legacy imports from the configured workspace and retain their guards.

    Frozen helper modules contain assertions. Refuse optimized Python for any
    workflow importing them, rather than silently disable those source/protocol
    checks. Standalone maintained engines have persistent checks and can run
    under ``-O``. No file is modified or imported by this boundary itself.
    """
    if not __debug__:
        raise RuntimeError(
            "Archived compatibility helpers require Python without -O/-OO; "
            "their provenance and chronology assertions must remain active."
        )
    root = Path(root).resolve()
    expected = root / "work"
    if not expected.is_dir():
        raise FileNotFoundError(f"Configured research root lacks frozen helpers: {expected}")
    existing = sys.modules.get("work")
    if existing is not None:
        locations = [Path(value).resolve() for value in getattr(existing, "__path__", [])]
        if expected not in locations:
            raise ImportError("Another 'work' package is loaded from a different research root")
    if str(root) not in sys.path:
        # Only the explicitly configured, preserved workspace is made importable.
        sys.path.insert(0, str(root))


def load_legacy_module(name: str, root: str | Path):
    """Import a frozen helper only after checking its configured workspace boundary.

    The module's actual source path must reside in the configured ``work`` tree;
    an already-loaded similarly named module from elsewhere is rejected.
    """
    if not name.startswith("work."):
        raise ValueError(
            "Only explicit frozen work.* helpers belong to this compatibility boundary"
        )
    activate_legacy_root(root)
    module = importlib.import_module(name)
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(Path(root).resolve() / "work"):
        raise ImportError(f"Frozen helper resolved outside configured workspace: {name}")
    return module


def maintained_source_paths() -> list[Path]:
    """List all maintained model/config sources whose behavior must be frozen per run.

    Freezing only engine.py would miss chronology guards, output protection, and
    path resolution. Include shared helpers and provenance alongside every model
    workflow module; no data or evaluation outcome is opened here.
    """
    modeling = Path(__file__).resolve().parent
    shared = [modeling.parent / name for name in ("common", "workflows", "steps")]
    sources = sorted(modeling.rglob("*.py"))
    sources.extend(path for directory in shared for path in sorted(directory.rglob("*.py")))
    return sources + [
        modeling / "PROVENANCE.json",
        modeling.parent / "config.py",
        modeling.parent / "provenance.py",
    ]


def require_run_destination(path: str | Path) -> Path:
    """Reject report destinations outside the configured maintained output tree.

    Reports read predictions and also write tables, HTML, and completion receipts.
    An ``--out`` override must never turn a frozen historical run into a writable
    destination. Read-only archival verification uses the original audit scripts.
    """
    from mqf_research.config import get_settings

    path = Path(path).resolve()
    if not path.is_relative_to(get_settings().output_root.resolve()):
        raise ValueError("Report destination must be inside the configured maintained output root")
    return path


def bind_run_configuration(destination: str | Path, *, create: bool = False) -> Path:
    """Freeze resolved non-secret paths and reject changed settings when resuming.

    Source-code hashes alone cannot detect environment-variable changes. This
    receipt binds all resolved Settings paths and the experiment directory, so a
    later process cannot silently combine old receipts with a different root.
    Only protocol initialization may create the receipt; later stages read it.
    """
    from mqf_research.config import get_settings

    destination = Path(destination).resolve()
    settings = get_settings()
    expected = {
        name: str(getattr(settings, name).resolve())
        for name in ("research_root", "source_project", "output_root", "maintained_root")
    }
    expected["run_directory"] = str(destination)
    target = destination / "RUN_CONFIG.json"
    if target.exists():
        actual = json.loads(target.read_text(encoding="utf-8-sig"))
        if actual != expected:
            raise ValueError("Resolved run configuration changed; use a fresh output root")
    elif create:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
    else:
        raise FileNotFoundError("Missing RUN_CONFIG.json; initialize the maintained protocol first")
    return target
