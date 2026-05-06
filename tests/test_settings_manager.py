from __future__ import annotations

from voice_lan_stt.settings_manager import GuiSettings, SettingsManager


def test_settings_manager_round_trip(tmp_path) -> None:
    path = tmp_path / "settings.json"
    manager = SettingsManager(path)

    manager.save(
        GuiSettings(
            base_url="http://192.168.1.50:1234/v1",
            model="whisper-local",
            selected_microphone="USB Mic",
            window_width=1000,
            window_height=700,
            window_x=20,
            window_y=30,
        )
    )

    loaded = manager.load()

    assert loaded.base_url == "http://192.168.1.50:1234/v1"
    assert loaded.model == "whisper-local"
    assert loaded.selected_microphone == "USB Mic"
    assert loaded.window_width == 1000
    assert loaded.window_height == 700
    assert loaded.window_x == 20
    assert loaded.window_y == 30


def test_settings_manager_returns_defaults_for_bad_json(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{bad json", encoding="utf-8")

    loaded = SettingsManager(path).load()

    assert loaded == GuiSettings()
