from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SETTINGS_FILENAME = "settings.json"


@dataclass
class GuiSettings:
    base_url: str = "http://192.168.1.50:1234/v1"
    model: str = "whisper-1"
    selected_microphone: str | None = None
    window_width: int = 920
    window_height: int = 680
    window_x: int | None = None
    window_y: int | None = None


def settings_path() -> Path:
    return Path(__file__).resolve().parents[2] / SETTINGS_FILENAME


class SettingsManager:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings_path()

    def load(self) -> GuiSettings:
        if not self.path.exists():
            return GuiSettings()

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return GuiSettings()

        if not isinstance(raw, dict):
            return GuiSettings()

        defaults = GuiSettings()
        data: dict[str, Any] = asdict(defaults)
        data.update({key: raw.get(key, value) for key, value in data.items()})
        return GuiSettings(**data)

    def save(self, settings: GuiSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
