"""Install the built wheel in an isolated environment and smoke-test its CLI."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import venv
from pathlib import Path


def find_wheel(directory: Path) -> Path:
    """Require exactly one wheel so the tested artifact is unambiguous."""
    wheels = tuple(sorted(directory.glob("choravox-*.whl")))
    if len(wheels) != 1:
        raise ValueError(f"expected exactly one choravox wheel in {directory}, found {len(wheels)}")
    return wheels[0]


def smoke_wheel(wheel: Path) -> None:
    """Install without dependency resolution and exercise import plus entry point."""
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("wheel path must name an existing .whl file")
    with tempfile.TemporaryDirectory(prefix="choravox-wheel-") as temporary:
        environment = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        python = environment / "bin" / "python"
        subprocess.run(
            [python, "-m", "pip", "install", "--no-deps", str(wheel)],
            check=True,
        )
        subprocess.run(
            [
                python,
                "-c",
                "import choravox; from choravox.cli import main; "
                "raise SystemExit(main(['--version']))",
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", type=Path)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    smoke_wheel(args.wheel if args.wheel is not None else find_wheel(args.dist))


if __name__ == "__main__":
    main()
