from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

try:
    from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFormLayout,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PySide6 is required for the desktop GUI. Install with: pip install PySide6"
    ) from exc

from .config import DEFAULT_API_KEY, DEFAULT_SAMPLE_RATE, DEFAULT_STT_MODEL, Settings
from .settings_manager import GuiSettings, SettingsManager
from .widgets import LevelMeter
from .workers import DeviceListWorker, RecordingWorker, ServerTestWorker, TranscribeWorker

GUI_DEFAULT_BASE_URL = "http://192.168.1.141:8080"


class MainWindow(QMainWindow):
    stop_recording_requested = Signal()
    cancel_recording_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TransWhisper - Voive LAN STT")

        self.settings_manager = SettingsManager()
        self.saved_settings = self.settings_manager.load()

        self.worker_threads: list[QThread] = []
        self.worker_refs: dict[QThread, QObject] = {}
        self.recording_thread: QThread | None = None
        self.recording_worker: RecordingWorker | None = None
        self.latest_transcript = ""
        self.elapsed_seconds = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_recording_timer)

        self.base_url_input = QLineEdit(self.saved_settings.base_url or GUI_DEFAULT_BASE_URL)
        self.model_input = QLineEdit(self.saved_settings.model or DEFAULT_STT_MODEL)
        self.device_combo = QComboBox()
        self.refresh_devices_button = QPushButton("Refresh Devices")
        self.test_server_button = QPushButton("Test Server")
        self.start_button = QPushButton("Start Recording")
        self.stop_button = QPushButton("Stop Recording")
        self.copy_latest_button = QPushButton("Copy Latest Transcript")
        self.clear_button = QPushButton("Clear Transcript")

        self.timer_label = QLabel("00:00")
        self.timer_label.setStyleSheet("font-size: 28px; font-weight: 700;")
        self.level_meter = LevelMeter()
        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.status_label = QLabel("Ready")

        self._build_layout()
        self._connect_signals()
        self._apply_theme()
        self._restore_window_geometry()
        self.set_recording_controls(recording=False)
        self.refresh_devices()

    def _build_layout(self) -> None:
        form = QFormLayout()
        form.addRow("Whisper.cpp Base URL", self.base_url_input)
        form.addRow("STT Model", self.model_input)

        device_row = QHBoxLayout()
        device_row.addWidget(self.device_combo, stretch=1)
        device_row.addWidget(self.refresh_devices_button)
        form.addRow("Microphone", device_row)

        controls = QGridLayout()
        controls.addWidget(self.test_server_button, 0, 0)
        controls.addWidget(self.start_button, 0, 1)
        controls.addWidget(self.stop_button, 0, 2)
        controls.addWidget(self.copy_latest_button, 1, 0)
        controls.addWidget(self.clear_button, 1, 1)

        live_row = QHBoxLayout()
        live_row.addWidget(QLabel("Timer"))
        live_row.addWidget(self.timer_label)
        live_row.addWidget(QLabel("Mic level"))
        live_row.addWidget(self.level_meter, stretch=1)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(controls)
        layout.addLayout(live_row)
        layout.addWidget(self.transcript_text, stretch=1)
        layout.addWidget(self.status_label)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

    def _connect_signals(self) -> None:
        self.refresh_devices_button.clicked.connect(self.refresh_devices)
        self.test_server_button.clicked.connect(self.test_server)
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.copy_latest_button.clicked.connect(self.copy_latest_transcript)
        self.clear_button.clicked.connect(self.clear_transcripts)
        self.base_url_input.textChanged.connect(self.save_settings)
        self.model_input.textChanged.connect(self.save_settings)
        self.device_combo.currentIndexChanged.connect(self.save_settings)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #1f2329; color: #f1f3f5; }
            QLineEdit, QTextEdit, QComboBox {
                background: #111418; color: #f1f3f5; border: 1px solid #3a414a;
                padding: 6px;
            }
            QPushButton {
                background: #2f6fed; color: #ffffff; border: 0; padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:disabled { background: #424852; color: #9aa0a6; }
            QProgressBar {
                background: #111418; border: 1px solid #3a414a;
            }
            QProgressBar::chunk { background: #2fcd73; }
            """
        )

    def _restore_window_geometry(self) -> None:
        self.resize(self.saved_settings.window_width, self.saved_settings.window_height)
        if self.saved_settings.window_x is not None and self.saved_settings.window_y is not None:
            self.move(self.saved_settings.window_x, self.saved_settings.window_y)

    def current_settings(self) -> Settings:
        return Settings(
            base_url=(self.base_url_input.text().strip() or GUI_DEFAULT_BASE_URL).rstrip("/"),
            api_key=DEFAULT_API_KEY,
            stt_model=self.model_input.text().strip() or DEFAULT_STT_MODEL,
            sample_rate=DEFAULT_SAMPLE_RATE,
        )

    def selected_device_index(self) -> int | None:
        data = self.device_combo.currentData()
        return int(data) if data is not None else None

    @Slot()
    def refresh_devices(self) -> None:
        self.set_status("Refreshing microphones")
        worker = DeviceListWorker()
        thread = self._start_worker(worker, worker.run)
        worker.success.connect(self.on_devices_loaded)
        worker.error.connect(self.on_devices_error)
        thread.start()

    @Slot(list)
    def on_devices_loaded(self, devices: list[dict[str, object]]) -> None:
        selected_name = self.saved_settings.selected_microphone
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        selected_index = 0
        for index, device in enumerate(devices):
            name = str(device["name"])
            self.device_combo.addItem(name, int(device["index"]))
            if selected_name == name:
                selected_index = index
        self.device_combo.setCurrentIndex(selected_index)
        self.device_combo.blockSignals(False)
        self.set_status("Ready")
        self.set_recording_controls(recording=False)
        self.save_settings()

    @Slot(str)
    def on_devices_error(self, message: str) -> None:
        self.device_combo.clear()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.set_status("Error")
        self.append_error(message)

    @Slot()
    def test_server(self) -> None:
        self.set_busy(True)
        self.set_status("Sending to Whisper.cpp")
        worker = ServerTestWorker(self.current_settings())
        thread = self._start_worker(worker, worker.run)
        worker.success.connect(self.on_server_success)
        worker.error.connect(self.on_worker_error)
        thread.start()

    @Slot(str)
    def on_server_success(self, message: str) -> None:
        self.set_busy(False)
        self.set_status("Done")
        self.append_system_message(message)

    @Slot()
    def start_recording(self) -> None:
        if self.device_combo.count() == 0:
            self.set_status("Error")
            self.append_error("No microphone is selected. Refresh devices and choose an input.")
            return

        self.set_busy(True)
        self.set_recording_controls(recording=True)
        self.elapsed_seconds = 0
        self.timer_label.setText("00:00")
        self.level_meter.set_level(0.0)
        self.set_status("RECORDING")

        worker = RecordingWorker(DEFAULT_SAMPLE_RATE, self.selected_device_index())
        thread = QThread(self)
        self.recording_worker = worker
        self.recording_thread = thread
        self.worker_refs[thread] = worker
        self.worker_threads.append(thread)

        worker.moveToThread(thread)
        thread.started.connect(worker.start)
        self.stop_recording_requested.connect(worker.stop)
        self.cancel_recording_requested.connect(worker.cancel)
        worker.started.connect(self.on_recording_started)
        worker.level.connect(self.level_meter.set_level)
        worker.stopped.connect(self.on_recording_stopped)
        worker.error.connect(self.on_recording_error)
        worker.stopped.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda: self._forget_thread(thread))
        thread.start()

    @Slot()
    def on_recording_started(self) -> None:
        self.timer.start(1000)

    @Slot()
    def stop_recording(self) -> None:
        if self.recording_worker is None:
            return
        self.set_status("Saving audio")
        self.stop_button.setEnabled(False)
        self.timer.stop()
        self.stop_recording_requested.emit()

    @Slot(object)
    def on_recording_stopped(self, wav_path: object) -> None:
        self.recording_worker = None
        self.recording_thread = None
        self.level_meter.set_level(0.0)
        self.set_status("Sending to Whisper.cpp")
        self.start_transcription(wav_path)

    @Slot(str)
    def on_recording_error(self, message: str) -> None:
        self.timer.stop()
        self.recording_worker = None
        self.recording_thread = None
        self.level_meter.set_level(0.0)
        self.set_busy(False)
        self.set_recording_controls(recording=False)
        self.set_status("Error")
        self.append_error(message)

    def start_transcription(self, wav_path: object) -> None:
        worker = TranscribeWorker(self.current_settings(), wav_path)
        thread = self._start_worker(worker, worker.run)
        worker.success.connect(self.on_transcription_success)
        worker.error.connect(self.on_worker_error)
        thread.start()

    @Slot(str)
    def on_transcription_success(self, transcript: str) -> None:
        self.latest_transcript = transcript
        self.append_transcript(transcript)
        self.set_busy(False)
        self.set_recording_controls(recording=False)
        self.set_status("Done")

    @Slot(str)
    def on_worker_error(self, message: str) -> None:
        self.set_busy(False)
        self.set_recording_controls(recording=False)
        self.set_status("Error")
        self.append_error(message)

    @Slot()
    def update_recording_timer(self) -> None:
        self.elapsed_seconds += 1
        minutes, seconds = divmod(self.elapsed_seconds, 60)
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

    @Slot()
    def copy_latest_transcript(self) -> None:
        QApplication.clipboard().setText(self.latest_transcript)
        self.set_status("Done")

    @Slot()
    def clear_transcripts(self) -> None:
        self.transcript_text.clear()
        self.latest_transcript = ""
        self.set_status("Ready")

    def append_transcript(self, transcript: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        block = f"[{timestamp}]\n{transcript.strip()}\n"
        self.transcript_text.append(block)
        self.transcript_text.verticalScrollBar().setValue(
            self.transcript_text.verticalScrollBar().maximum()
        )

    def append_system_message(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.transcript_text.append(f"[{timestamp}]\n{message}\n")

    def append_error(self, message: str) -> None:
        self.transcript_text.append(f"Error: {message}\n")
        self.transcript_text.verticalScrollBar().setValue(
            self.transcript_text.verticalScrollBar().maximum()
        )

    @Slot(str)
    def set_status(self, status: str) -> None:
        self.status_label.setText(status)

    def set_busy(self, busy: bool) -> None:
        self.test_server_button.setEnabled(not busy)
        self.refresh_devices_button.setEnabled(not busy)
        if not self.stop_button.isEnabled():
            self.start_button.setEnabled(not busy and self.device_combo.count() > 0)

    def set_recording_controls(self, recording: bool) -> None:
        self.start_button.setEnabled(not recording and self.device_combo.count() > 0)
        self.stop_button.setEnabled(recording)
        self.base_url_input.setEnabled(not recording)
        self.model_input.setEnabled(not recording)
        self.device_combo.setEnabled(not recording)

    def _start_worker(self, worker: QObject, run_slot: Any) -> QThread:
        thread = QThread(self)
        self.worker_refs[thread] = worker
        self.worker_threads.append(thread)
        worker.moveToThread(thread)
        thread.started.connect(run_slot)
        if hasattr(worker, "success"):
            worker.success.connect(thread.quit)
        if hasattr(worker, "error"):
            worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda: self._forget_thread(thread))
        return thread

    def _forget_thread(self, thread: QThread) -> None:
        self.worker_refs.pop(thread, None)
        if thread in self.worker_threads:
            self.worker_threads.remove(thread)
        if thread is self.recording_thread:
            self.recording_thread = None
        thread.deleteLater()

    @Slot()
    def save_settings(self) -> None:
        selected_name = self.device_combo.currentText() or None
        geometry = self.geometry()
        self.settings_manager.save(
            GuiSettings(
                base_url=(self.base_url_input.text().strip() or GUI_DEFAULT_BASE_URL).rstrip("/"),
                model=self.model_input.text().strip() or DEFAULT_STT_MODEL,
                selected_microphone=selected_name,
                window_width=self.width(),
                window_height=self.height(),
                window_x=geometry.x(),
                window_y=geometry.y(),
            )
        )

    def closeEvent(self, event: Any) -> None:
        self.save_settings()
        if self.recording_worker is not None:
            self.cancel_recording_requested.emit()
        for thread in list(self.worker_threads):
            thread.quit()
            thread.wait(1500)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
