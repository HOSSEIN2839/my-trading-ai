from pathlib import Path

from src.paths import PROJECT_ROOT, project_path


def test_relative_paths_resolve_from_repository_root():
    assert project_path("data/model.joblib") == PROJECT_ROOT / "data" / "model.joblib"


def test_absolute_paths_are_preserved(tmp_path):
    path = tmp_path / "custom.sqlite3"
    assert project_path(path) == path


def test_home_expansion_is_supported():
    result = project_path("~/custom-path")
    assert result.is_absolute()
    assert str(result).startswith(str(Path.home()))
