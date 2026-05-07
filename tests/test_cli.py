from __future__ import annotations

from argparse import Namespace

import pytest

from voice_lan_stt.cli import build_parser, settings_from_args


def test_parse_record_defaults() -> None:
    args = build_parser().parse_args(["record"])

    assert args.command == "record"
    assert args.seconds == 5.0


def test_parse_record_seconds() -> None:
    args = build_parser().parse_args(["record", "--seconds", "2.5"])

    assert args.command == "record"
    assert args.seconds == 2.5


def test_parse_record_keep_audio() -> None:
    args = build_parser().parse_args(["record", "--keep-audio"])

    assert args.command == "record"
    assert args.keep_audio is True


def test_parse_ptt_copy() -> None:
    args = build_parser().parse_args(["ptt", "--copy"])

    assert args.command == "ptt"
    assert args.copy is True


def test_parse_ptt_keep_audio() -> None:
    args = build_parser().parse_args(["ptt", "--keep-audio"])

    assert args.command == "ptt"
    assert args.keep_audio is True


def test_parse_listen_defaults() -> None:
    args = build_parser().parse_args(["listen"])

    assert args.command == "listen"
    assert args.threshold == 0.01
    assert args.silence_ms == 800
    assert args.min_speech_ms == 250
    assert args.max_segment_seconds == 15.0
    assert args.keep_audio is False


def test_parse_listen_options() -> None:
    args = build_parser().parse_args(
        [
            "listen",
            "--threshold",
            "0.02",
            "--silence-ms",
            "600",
            "--min-speech-ms",
            "300",
            "--max-segment-seconds",
            "8",
            "--sample-rate",
            "22050",
        ]
    )

    assert args.command == "listen"
    assert args.threshold == 0.02
    assert args.silence_ms == 600
    assert args.min_speech_ms == 300
    assert args.max_segment_seconds == 8.0
    assert args.sample_rate == 22050


def test_parse_test_server() -> None:
    args = build_parser().parse_args(["test-server"])

    assert args.command == "test-server"


def test_parse_global_language_option() -> None:
    args = build_parser().parse_args(["--language", "es", "record"])

    assert args.command == "record"
    assert args.language == "es"


def test_parse_diagnose() -> None:
    args = build_parser().parse_args(["diagnose"])

    assert args.command == "diagnose"


def test_parse_server_command() -> None:
    args = build_parser().parse_args(["server-command"])

    assert args.command == "server-command"


def test_parse_history_options() -> None:
    args = build_parser().parse_args(["history", "--limit", "20", "--search", "keyword"])

    assert args.command == "history"
    assert args.limit == 20
    assert args.search == "keyword"


def test_parse_export_options() -> None:
    args = build_parser().parse_args(["export", "--format", "json", "--search", "keyword"])

    assert args.command == "export"
    assert args.format == "json"
    assert args.search == "keyword"


def test_settings_from_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "WHISPERCPP_BASE_URL",
        "WHISPERCPP_INFERENCE_PATH",
        "WHISPERCPP_MODEL_PATH",
        "WHISPERCPP_LANGUAGE",
        "SAMPLE_RATE",
    ):
        monkeypatch.delenv(name, raising=False)

    args = Namespace(
        base_url="http://192.168.1.141:8080/",
        inference_path="/transcribe",
        model_path="models/ggml-small.en.bin",
        language="auto",
        sample_rate=22050,
        temperature=0.1,
        temperature_inc=0.3,
        response_format="text",
    )

    settings = settings_from_args(args)

    assert settings.base_url == "http://192.168.1.141:8080"
    assert settings.inference_path == "/transcribe"
    assert settings.model_path == "models/ggml-small.en.bin"
    assert settings.language == "auto"
    assert settings.sample_rate == 22050
    assert settings.temperature == 0.1
    assert settings.temperature_inc == 0.3
    assert settings.response_format == "text"
