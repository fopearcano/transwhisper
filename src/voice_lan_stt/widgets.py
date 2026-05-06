from __future__ import annotations

try:
    from PySide6.QtWidgets import QProgressBar
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PySide6 is required for the desktop GUI. Install with: pip install PySide6"
    ) from exc


class LevelMeter(QProgressBar):
    def __init__(self) -> None:
        super().__init__()
        self.setRange(0, 100)
        self.setTextVisible(False)
        self.setFixedHeight(18)
        self.setValue(0)

    def set_level(self, level: float) -> None:
        # RMS values are usually small; this scaling gives a useful visible meter
        # without doing expensive waveform rendering.
        self.setValue(max(0, min(100, int(level * 500))))
