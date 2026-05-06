from __future__ import annotations

import pytest

from voice_lan_stt.config import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_STT_MODEL,
    load_settings,
)


def test_load_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "WHISPERCPP_BASE_URL",
        "WHISPERCPP_API_KEY",
        "WHISPERCPP_STT_MODEL",
        "LMSTUDIO_BASE_URL",
        "LMSTUDIO_API_KEY",
        "LMSTUDIO_STT_MODEL",
        "SAMPLE_RATE",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.api_key == DEFAULT_API_KEY
    assert settings.stt_model == DEFAULT_STT_MODEL
    assert settings.sample_rate == DEFAULT_SAMPLE_RATE
    assert settings.transcription_url == "http://192.168.1.141:8080/inference"
    assert settings.server_url == "http://192.168.1.141:8080"


def test_load_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPERCPP_BASE_URL", "http://192.168.1.141:8080/")
    monkeypatch.setenv("WHISPERCPP_API_KEY", "test-key")
    monkeypatch.setenv("WHISPERCPP_STT_MODEL", "local-whisper")
    monkeypatch.setenv("SAMPLE_RATE", "22050")

    settings = load_settings()

    assert settings.base_url == "http://192.168.1.141:8080"
    assert settings.api_key == "test-key"
    assert settings.stt_model == "local-whisper"
    assert settings.sample_rate == 22050


def test_invalid_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAMPLE_RATE", "not-an-int")

    with pytest.raises(ValueError, match="SAMPLE_RATE must be an integer"):
        load_settings()
