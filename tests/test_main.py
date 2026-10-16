from pathlib import Path

import main


def test_initialize_lastlog_uses_project_root(tmp_path, monkeypatch):
    project_root = Path(main.__file__).resolve().parent
    log_path = project_root / "lastlog_test"
    if log_path.exists():
        log_path.unlink()

    monkeypatch.chdir(tmp_path)
    created = main.initialize_lastlog(Path("lastlog_test"))

    assert created == log_path
    assert log_path.exists()
    assert not (tmp_path / "lastlog_test").exists()
    log_path.unlink()
