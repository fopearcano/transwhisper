from __future__ import annotations

import pytest

from voice_lan_stt.config import (
    DEFAULT_BASE_URL,
    DEFAULT_INFERENCE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SAMPLE_RATE,
    load_settings,
)


def test_load_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "WHISPERCPP_BASE_URL",
        "WHISPERCPP_INFERENCE_PATH",
        "WHISPERCPP_MODEL_PATH",
        "WHISPERCPP_LANGUAGE",
        "WHISPERCPP_TEMPERATURE",
        "WHISPERCPP_TEMPERATURE_INC",
        "WHISPERCPP_RESPONSE_FORMAT",
        "SAMPLE_RATE",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.inference_path == DEFAULT_INFERENCE_PATH
    assert settings.model_path == DEFAULT_MODEL_PATH
    assert settings.sample_rate == DEFAULT_SAMPLE_RATE
    assert settings.transcription_url == "http://192.168.1.141:8080/inference"
    assert settings.server_url == "http://192.168.1.141:8080"


def test_load_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPERCPP_BASE_URL", "http://192.168.1.141:8080/")
    monkeypatch.setenv("WHISPERCPP_INFERENCE_PATH", "transcribe")
    monkeypatch.setenv("WHISPERCPP_MODEL_PATH", "models/ggml-small.en.bin")
    monkeypatch.setenv("WHISPERCPP_LANGUAGE", "auto")
    monkeypatch.setenv("WHISPERCPP_TEMPERATURE", "0.1")
    monkeypatch.setenv("WHISPERCPP_TEMPERATURE_INC", "0.3")
    monkeypatch.setenv("WHISPERCPP_RESPONSE_FORMAT", "text")
    monkeypatch.setenv("SAMPLE_RATE", "22050")

    settings = load_settings()

    assert settings.base_url == "http://192.168.1.141:8080"
    assert settings.inference_path == "/transcribe"
    assert settings.model_path == "models/ggml-small.en.bin"
    assert settings.language == "auto"
    assert settings.temperature == 0.1
    assert settings.temperature_inc == 0.3
    assert settings.response_format == "text"
    assert settings.sample_rate == 22050


def test_invalid_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAMPLE_RATE", "not-an-int")

    with pytest.raises(ValueError, match="SAMPLE_RATE must be an integer"):
        load_settings()
