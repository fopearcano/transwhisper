from __future__ import annotations

import queue
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.io import wavfile


class MicrophoneError(RuntimeError):
    """Raised when microphone capture fails."""


@dataclass(frozen=True)
class VadOptions:
    sample_rate: int
    threshold: float = 0.01
    silence_ms: int = 800
    min_speech_ms: int = 250
    max_segment_seconds: float = 15.0
    block_ms: int = 30

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if self.threshold <= 0:
            raise ValueError("threshold must be greater than zero")
        if self.silence_ms <= 0:
            raise ValueError("silence_ms must be greater than zero")
        if self.min_speech_ms <= 0:
            raise ValueError("min_speech_ms must be greater than zero")
        if self.max_segment_seconds <= 0:
            raise ValueError("max_segment_seconds must be greater than zero")
        if self.block_ms <= 0:
            raise ValueError("block_ms must be greater than zero")


def _load_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise MicrophoneError(
            "Microphone capture is unavailable. Install sounddevice and ensure PortAudio "
            "can access an input device."
        ) from exc
    return sd


def _write_temp_wav(sample_rate: int, recording: Any) -> Path:
    temp_file = tempfile.NamedTemporaryFile(
        prefix="voice_lan_stt_",
        suffix=".wav",
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    wavfile.write(temp_path, sample_rate, recording)
    return temp_path


class ManualRecordingSession:
    def __init__(
        self,
        sample_rate: int,
        device: int | None = None,
        level_callback: Callable[[float], None] | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        self.sample_rate = sample_rate
        self.device = device
        self.level_callback = level_callback
        self._stream = None
        self._chunks = []
        self._started = False

    def start(self) -> None:
        if self._started:
            raise MicrophoneError("Recording is already running.")

        sd = _load_sounddevice()

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            if status:
                # Temporary stream warnings can happen; keep any usable audio.
                pass
            self._chunks.append(indata.copy())
            if self.level_callback is not None:
                self.level_callback(rms_for_audio(indata))

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                device=self.device,
                channels=1,
                dtype="int16",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            self.close()
            raise MicrophoneError(
                "Could not record from the microphone. Check that an input device is "
                "connected and that this terminal has microphone permission."
            ) from exc

        self._started = True

    def stop_to_temp_wav(self) -> Path:
        if not self._started:
            raise MicrophoneError("Recording is not running.")

        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception as exc:
            raise MicrophoneError("Could not stop microphone recording cleanly.") from exc
        finally:
            self._stream = None
            self._started = False

        if not self._chunks:
            raise MicrophoneError("No microphone audio was captured.")

        try:
            import numpy as np
        except ImportError as exc:
            raise MicrophoneError("Microphone capture requires numpy.") from exc

        recording = np.concatenate(self._chunks, axis=0)
        return _write_temp_wav(self.sample_rate, recording)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._started = False


def record_to_temp_wav(seconds: float, sample_rate: int) -> Path:
    if seconds <= 0:
        raise ValueError("seconds must be greater than zero")

    sd = _load_sounddevice()
    frame_count = int(seconds * sample_rate)
    try:
        print(f"Recording {seconds:g}s at {sample_rate} Hz...")
        recording = sd.rec(
            frame_count,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
    except Exception as exc:
        raise MicrophoneError(
            "Could not record from the microphone. Check that an input device is "
            "connected and that this terminal has microphone permission."
        ) from exc

    return _write_temp_wav(sample_rate, recording)


def record_until_enter_to_temp_wav(
    sample_rate: int,
    wait_for_stop: Callable[[], str] = input,
) -> Path:
    session = ManualRecordingSession(sample_rate)
    try:
        session.start()
        wait_for_stop()
        return session.stop_to_temp_wav()
    finally:
        session.close()


def rms_for_audio(audio: Any) -> float:
    try:
        import numpy as np
    except ImportError as exc:
        raise MicrophoneError("Microphone capture requires numpy.") from exc

    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return 0.0
    samples = samples / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))


def listen_for_vad_segments(options: VadOptions) -> Iterator[Path]:
    options.validate()
    sd = _load_sounddevice()

    try:
        import numpy as np
    except ImportError as exc:
        raise MicrophoneError("Microphone capture requires numpy.") from exc

    audio_queue: queue.Queue = queue.Queue()
    block_size = max(1, int(options.sample_rate * options.block_ms / 1000))
    silence_frames_required = int(options.sample_rate * options.silence_ms / 1000)
    min_speech_frames = int(options.sample_rate * options.min_speech_ms / 1000)
    max_segment_frames = int(options.sample_rate * options.max_segment_seconds)

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            # Stream warnings are transient on some devices; the main loop can
            # still use any clean audio blocks that arrive.
            pass
        audio_queue.put(indata.copy())

    def reset_segment() -> tuple[list, int, int, int]:
        return [], 0, 0, 0

    chunks, total_frames, speech_frames, silence_frames = reset_segment()

    try:
        with sd.InputStream(
            samplerate=options.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=block_size,
            callback=callback,
        ):
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.25)
                except queue.Empty:
                    continue

                chunk_frames = len(chunk)
                is_speech = rms_for_audio(chunk) >= options.threshold

                if not chunks and not is_speech:
                    continue

                chunks.append(chunk)
                total_frames += chunk_frames

                if is_speech:
                    speech_frames += chunk_frames
                    silence_frames = 0
                else:
                    silence_frames += chunk_frames

                speech_ended = silence_frames >= silence_frames_required
                segment_too_long = total_frames >= max_segment_frames
                if not speech_ended and not segment_too_long:
                    continue

                if speech_frames >= min_speech_frames:
                    recording = np.concatenate(chunks, axis=0)
                    yield _write_temp_wav(options.sample_rate, recording)

                chunks, total_frames, speech_frames, silence_frames = reset_segment()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise MicrophoneError(
            "Could not listen to the microphone. Check that an input device is "
            "connected and that this terminal has microphone permission."
        ) from exc
