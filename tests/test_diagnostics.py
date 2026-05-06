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
    parsed = parse_base_url("http://192.168.1.141:8080")

    assert parsed.scheme == "http"
    assert parsed.host == "192.168.1.141"
    assert parsed.port == 8080
    assert parsed.path == ""


def test_parse_base_url_uses_default_https_port() -> None:
    parsed = parse_base_url("https://whisper.local")

    assert parsed.scheme == "https"
    assert parsed.host == "whisper.local"
    assert parsed.port == 443
    assert parsed.path == ""


def test_parse_base_url_rejects_missing_scheme() -> None:
    with pytest.raises(ValueError, match="http:// or https://"):
        parse_base_url("192.168.1.141:8080")


def test_format_diagnostic_report_success() -> None:
    report = DiagnosticReport(
        hostname="client-host",
        base_url="http://192.168.1.141:8080",
        model="whisper.cpp",
        parsed_url=ParsedBaseUrl("http", "192.168.1.141", 8080, ""),
        tcp=CheckResult(True, "connected"),
        server=CheckResult(True, "server reachable"),
        audio_endpoint=CheckResult(True, "POST /inference accepted a short silent WAV"),
    )
    report.likely_fixes = likely_fixes(report)

    formatted = format_diagnostic_report(report)

    assert "Local hostname: client-host" in formatted
    assert "TCP connection: OK - connected" in formatted
    assert "POST /inference: OK" in formatted
    assert "No obvious LAN issue" in formatted


def test_format_diagnostic_report_tcp_failure_includes_actionable_fixes() -> None:
    report = DiagnosticReport(
        hostname="client-host",
        base_url="http://192.168.1.141:8080",
        model="whisper.cpp",
        parsed_url=ParsedBaseUrl("http", "192.168.1.141", 8080, ""),
        tcp=CheckResult(False, "connection refused"),
        server=CheckResult(False, "skipped because TCP connection failed"),
        audio_endpoint=CheckResult(False, "skipped because TCP connection failed"),
    )
    report.likely_fixes = likely_fixes(report)

    formatted = format_diagnostic_report(report)

    assert "TCP connection: FAIL - connection refused" in formatted
    assert "Whisper.cpp server may not be running" in formatted
    assert "Firewall may be blocking" in formatted
    assert "bound only to localhost" in formatted


def test_format_diagnostic_report_localhost_fix() -> None:
    report = DiagnosticReport(
        hostname="client-host",
        base_url="http://localhost:8080",
        model="whisper.cpp",
        parsed_url=ParsedBaseUrl("http", "localhost", 8080, ""),
        tcp=CheckResult(True, "connected"),
        server=CheckResult(True, "server reachable"),
        audio_endpoint=CheckResult(False, "unsupported"),
    )
    report.likely_fixes = likely_fixes(report)

    formatted = format_diagnostic_report(report)

    assert "use the Whisper.cpp host IP instead of localhost" in formatted
    assert "POST /inference" in formatted
