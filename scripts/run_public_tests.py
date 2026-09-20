"""Run the repository's lightweight tests without external test dependencies."""

import importlib.util
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DIR = ROOT / "tests"


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        path.stem,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def main():

    test_files = sorted(
        TEST_DIR.glob("test_*.py")
    )

    n_passed = 0

    for path in test_files:

        module = load_module(
            path
        )

        functions = [
            function
            for name, function
            in inspect.getmembers(
                module,
                inspect.isfunction,
            )
            if name.startswith(
                "test_"
            )
        ]

        for function in functions:
            print(
                f"{path.name}::{function.__name__}",
                end=" ... ",
            )

            function()

            print("PASS")

            n_passed += 1

    print()
    print(
        f"OK - {n_passed} public tests passed."
    )


if __name__ == "__main__":
    main()
