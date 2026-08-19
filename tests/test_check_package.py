from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_package import find_wheel, smoke_wheel


def test_find_wheel_requires_one_unambiguous_artifact(tmp_path) -> None:
    with pytest.raises(ValueError, match="found 0"):
        find_wheel(tmp_path)
    wheel = tmp_path / "choravox-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"fixture")
    assert find_wheel(tmp_path) == wheel
    (tmp_path / "choravox-0.1.1-py3-none-any.whl").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="found 2"):
        find_wheel(tmp_path)


def test_smoke_wheel_rejects_non_wheel_path(tmp_path) -> None:
    path = tmp_path / "package.zip"
    path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match=r"existing \.whl"):
        smoke_wheel(path)


@pytest.mark.slow
def test_built_wheel_installs_and_runs_entry_point() -> None:
    smoke_wheel(find_wheel(Path("dist")))
