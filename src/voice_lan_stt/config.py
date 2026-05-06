from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_STT_MODEL = "whisper-1"
DEFAULT_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Settings:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    stt_model: str = DEFAULT_STT_MODEL
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def transcription_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/audio/transcriptions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def load_settings() -> Settings:
    return Settings(
        base_url=os.getenv("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/"),
        api_key=os.getenv("LMSTUDIO_API_KEY", DEFAULT_API_KEY).strip() or DEFAULT_API_KEY,
        stt_model=os.getenv("LMSTUDIO_STT_MODEL", DEFAULT_STT_MODEL).strip() or DEFAULT_STT_MODEL,
        sample_rate=_read_int_env("SAMPLE_RATE", DEFAULT_SAMPLE_RATE),
    )
