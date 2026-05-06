from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://192.168.1.141:8080"
DEFAULT_API_KEY = ""
DEFAULT_STT_MODEL = "whisper.cpp"
DEFAULT_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Settings:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    stt_model: str = DEFAULT_STT_MODEL
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def transcription_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/inference"

    @property
    def server_url(self) -> str:
        return self.base_url.rstrip("/")


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
        base_url=_read_first_env(
            "WHISPERCPP_BASE_URL",
            "LMSTUDIO_BASE_URL",
            default=DEFAULT_BASE_URL,
        ).rstrip("/"),
        api_key=_read_first_env(
            "WHISPERCPP_API_KEY",
            "LMSTUDIO_API_KEY",
            default=DEFAULT_API_KEY,
        ),
        stt_model=_read_first_env(
            "WHISPERCPP_STT_MODEL",
            "LMSTUDIO_STT_MODEL",
            default=DEFAULT_STT_MODEL,
        ),
        sample_rate=_read_int_env("SAMPLE_RATE", DEFAULT_SAMPLE_RATE),
    )


def _read_first_env(*names: str, default: str) -> str:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is not None and raw_value.strip() != "":
            return raw_value.strip()
    return default
