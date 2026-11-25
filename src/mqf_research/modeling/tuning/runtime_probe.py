"""Import-only diagnostics; reads no research data and fits no model."""

import faulthandler
import importlib
import sys
import time


def main():
    """Print installed scientific-library versions without reading data or fitting.

    Diagnostic imports run only through this explicit entrypoint. Always cancel
    the traceback timer, including when an import fails.
    """
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    print("Python " + sys.version, flush=True)
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        for name in [
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "torch",
            "joblib",
            "pyarrow",
            "filelock",
        ]:
            start = time.perf_counter()
            print("Importing " + name, flush=True)
            module = importlib.import_module(name)
            print(f"{name} {module.__version__}: {time.perf_counter() - start:.1f}s", flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
    print("Runtime imports complete.", flush=True)


if __name__ == "__main__":
    main()
