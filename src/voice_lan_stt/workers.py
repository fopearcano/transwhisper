from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import QObject, Signal, Slot
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PySide6 is required for the desktop GUI. Install with: pip install PySide6"
    ) from exc

from .config import Settings
from .lmstudio_client import WhisperCppClient
from .recorder import ManualRecordingSession


class DeviceListWorker(QObject):
    success = Signal(list)
    error = Signal(str)

    @Slot()
    def run(self) -> None:
        try:
            import sounddevice as sd

            devices = sd.query_devices()
        except Exception as exc:
            self.error.emit(f"No microphone devices could be queried: {exc}")
            return

        input_devices: list[dict[str, object]] = []
        for index, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) <= 0:
                continue
            name = str(device.get("name", f"Input {index}"))
            input_devices.append({"index": index, "name": name})

        if not input_devices:
            self.error.emit("No microphone input devices were found.")
            return

        self.success.emit(input_devices)


class ServerTestWorker(QObject):
    success = Signal(str)
    error = Signal(str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            message = WhisperCppClient(self.settings, timeout=10).test_server()
        except Exception as exc:
            self.error.emit(readable_error(exc))
            return

        self.success.emit(message)


class RecordingWorker(QObject):
    started = Signal()
    level = Signal(float)
    stopped = Signal(object)
    error = Signal(str)

    def __init__(self, sample_rate: int, device_index: int | None) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.session: ManualRecordingSession | None = None

    @Slot()
    def start(self) -> None:
        try:
            self.session = ManualRecordingSession(
                self.sample_rate,
                device=self.device_index,
                level_callback=self.level.emit,
            )
            self.session.start()
        except Exception as exc:
            self.error.emit(readable_error(exc))
            return

        self.started.emit()

    @Slot()
    def stop(self) -> None:
        if self.session is None:
            self.error.emit("Recording is not running.")
            return

        try:
            wav_path = self.session.stop_to_temp_wav()
        except Exception as exc:
            self.error.emit(readable_error(exc))
            return
        finally:
            self.session.close()
            self.session = None

        self.stopped.emit(wav_path)

    @Slot()
    def cancel(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None


class TranscribeWorker(QObject):
    success = Signal(str)
    error = Signal(str)

    def __init__(self, settings: Settings, wav_path: Path) -> None:
        super().__init__()
        self.settings = settings
        self.wav_path = wav_path

    @Slot()
    def run(self) -> None:
        try:
            transcript = WhisperCppClient(self.settings).transcribe(self.wav_path)
            if transcript.strip() == "":
                raise ValueError("Whisper.cpp returned an empty transcript.")
        except Exception as exc:
            self.error.emit(readable_error(exc))
            return
        finally:
            self.wav_path.unlink(missing_ok=True)

        self.success.emit(transcript)


def readable_error(exc: Exception) -> str:
    message = str(exc).strip()
    if message == "":
        message = exc.__class__.__name__

    lowered = message.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return f"Timeout while contacting Whisper.cpp: {message}"
    if "could not reach" in lowered or "connection" in lowered:
        return f"Whisper.cpp unreachable: {message}"
    if "inference" in lowered or "unsupported" in lowered:
        return f"Unsupported audio endpoint: {message}"
    if "text field" in lowered or "json" in lowered:
        return f"Invalid response from Whisper.cpp: {message}"
    if "microphone" in lowered or "input device" in lowered:
        return f"Microphone error: {message}"
    return message
