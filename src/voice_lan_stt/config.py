from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://192.168.1.141:8080"
DEFAULT_INFERENCE_PATH = "/inference"
DEFAULT_MODEL_PATH = "models/ggml-base.en.bin"
DEFAULT_LANGUAGE = "en"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TEMPERATURE_INC = 0.2
DEFAULT_RESPONSE_FORMAT = "json"


@dataclass(frozen=True)
class Settings:
    base_url: str = DEFAULT_BASE_URL
    inference_path: str = DEFAULT_INFERENCE_PATH
    model_path: str = DEFAULT_MODEL_PATH
    language: str = DEFAULT_LANGUAGE
    sample_rate: int = DEFAULT_SAMPLE_RATE
    temperature: float = DEFAULT_TEMPERATURE
    temperature_inc: float = DEFAULT_TEMPERATURE_INC
    response_format: str = DEFAULT_RESPONSE_FORMAT

    @property
    def transcription_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.inference_path.strip('/')}"

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


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc


def load_settings() -> Settings:
    return Settings(
        base_url=_read_env("WHISPERCPP_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        inference_path=normalize_inference_path(
            _read_env("WHISPERCPP_INFERENCE_PATH", DEFAULT_INFERENCE_PATH)
        ),
        model_path=_read_env("WHISPERCPP_MODEL_PATH", DEFAULT_MODEL_PATH),
        language=_read_env("WHISPERCPP_LANGUAGE", DEFAULT_LANGUAGE),
        sample_rate=_read_int_env("SAMPLE_RATE", DEFAULT_SAMPLE_RATE),
        temperature=_read_float_env("WHISPERCPP_TEMPERATURE", DEFAULT_TEMPERATURE),
        temperature_inc=_read_float_env("WHISPERCPP_TEMPERATURE_INC", DEFAULT_TEMPERATURE_INC),
        response_format=_read_env("WHISPERCPP_RESPONSE_FORMAT", DEFAULT_RESPONSE_FORMAT),
    )


def normalize_inference_path(value: str) -> str:
    stripped = value.strip()
    if stripped == "":
        return DEFAULT_INFERENCE_PATH
    return "/" + stripped.strip("/")


def _read_env(name: str, default: str) -> str:
    raw_value = os.getenv(name)
    if raw_value is not None and raw_value.strip() != "":
        return raw_value.strip()
    return default
