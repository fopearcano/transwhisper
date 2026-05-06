from __future__ import annotations

import pytest

from voice_lan_stt.diagnostics import (
    CheckResult,
    DiagnosticReport,
    ParsedBaseUrl,
    format_diagnostic_report,
    likely_fixes,
    parse_base_url,
)


def test_parse_base_url_with_explicit_lan_port() -> None:
    parsed = parse_base_url("http://192.168.1.50:1234/v1")

    assert parsed.scheme == "http"
    assert parsed.host == "192.168.1.50"
    assert parsed.port == 1234
    assert parsed.path == "/v1"


def test_parse_base_url_uses_default_https_port() -> None:
    parsed = parse_base_url("https://lmstudio.local/v1")

    assert parsed.scheme == "https"
    assert parsed.host == "lmstudio.local"
    assert parsed.port == 443
    assert parsed.path == "/v1"


def test_parse_base_url_rejects_missing_scheme() -> None:
    with pytest.raises(ValueError, match="http:// or https://"):
        parse_base_url("192.168.1.50:1234/v1")


def test_format_diagnostic_report_success() -> None:
    report = DiagnosticReport(
        hostname="client-host",
        base_url="http://192.168.1.50:1234/v1",
        model="whisper-1",
        parsed_url=ParsedBaseUrl("http", "192.168.1.50", 1234, "/v1"),
        tcp=CheckResult(True, "connected"),
        models=CheckResult(True, "GET /models returned 1 model(s)"),
        audio_endpoint=CheckResult(True, "accepted silent WAV"),
        available_models=["whisper-1"],
    )
    report.likely_fixes = likely_fixes(report)

    formatted = format_diagnostic_report(report)

    assert "Local hostname: client-host" in formatted
    assert "TCP connection: OK - connected" in formatted
    assert "- whisper-1 (configured)" in formatted
    assert "No obvious LAN issue" in formatted


def test_format_diagnostic_report_tcp_failure_includes_actionable_fixes() -> None:
    report = DiagnosticReport(
        hostname="client-host",
        base_url="http://192.168.1.50:1234/v1",
        model="whisper-1",
        parsed_url=ParsedBaseUrl("http", "192.168.1.50", 1234, "/v1"),
        tcp=CheckResult(False, "connection refused"),
        models=CheckResult(False, "skipped because TCP connection failed"),
        audio_endpoint=CheckResult(False, "skipped because TCP connection failed"),
    )
    report.likely_fixes = likely_fixes(report)

    formatted = format_diagnostic_report(report)

    assert "TCP connection: FAIL - connection refused" in formatted
    assert "LM Studio server may not be running" in formatted
    assert "Firewall may be blocking" in formatted
    assert "bound only to localhost" in formatted


def test_format_diagnostic_report_wrong_model_fix() -> None:
    report = DiagnosticReport(
        hostname="client-host",
        base_url="http://localhost:1234/v1",
        model="missing-model",
        parsed_url=ParsedBaseUrl("http", "localhost", 1234, "/v1"),
        tcp=CheckResult(True, "connected"),
        models=CheckResult(True, "GET /models returned 1 model(s)"),
        audio_endpoint=CheckResult(False, "model unavailable"),
        available_models=["whisper-1"],
    )
    report.likely_fixes = likely_fixes(report)

    formatted = format_diagnostic_report(report)

    assert "Wrong model name" in formatted
    assert "missing-model" in formatted
    assert "use the LM Studio host IP instead of localhost" in formatted
