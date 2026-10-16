from pathlib import Path

from lidar.platform import ConfigManager


def test_config_manager_initializes_profile_paths(tmp_path: Path) -> None:
    config = ConfigManager(config_dir=tmp_path / "configs", active_profile="default")

    assert config.config_dir == tmp_path / "configs"
    assert config.profile_path == tmp_path / "configs" / "default"
    assert config.profile_file == tmp_path / "configs" / "default" / "default_profile.json"
    assert config.values["theme"] == "dark"
