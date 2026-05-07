from __future__ import annotations

import socket
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .config import Settings
from .whispercpp_client import (
    EndpointUnsupportedError,
    ModelUnavailableError,
    ServerUnreachableError,
    WhisperCppClient,
)


@dataclass(frozen=True)
class ParsedBaseUrl:
    scheme: str
    host: str
    port: int
    path: str


@dataclass
class CheckResult:
    ok: bool
    message: str


@dataclass
class DiagnosticReport:
    hostname: str
    base_url: str
    inference_path: str
    model_path: str
    parsed_url: ParsedBaseUrl | None
    url_error: str | None = None
    tcp: CheckResult = field(default_factory=lambda: CheckResult(False, "not run"))
    server: CheckResult = field(default_factory=lambda: CheckResult(False, "not run"))
    audio_endpoint: CheckResult = field(default_factory=lambda: CheckResult(False, "not run"))
    likely_fixes: list[str] = field(default_factory=list)


def parse_base_url(base_url: str) -> ParsedBaseUrl:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Base URL must start with http:// or https://")
    if not parsed.hostname:
        raise ValueError("Base URL must include a hostname or IP address")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Base URL includes an invalid port") from exc

    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    return ParsedBaseUrl(
        scheme=parsed.scheme,
        host=parsed.hostname,
        port=port,
        path=parsed.path or "",
    )


def test_tcp_connection(parsed_url: ParsedBaseUrl, timeout: float = 3.0) -> CheckResult:
    try:
        with socket.create_connection((parsed_url.host, parsed_url.port), timeout=timeout):
            return CheckResult(True, f"connected to {parsed_url.host}:{parsed_url.port}")
    except OSError as exc:
        return CheckResult(
            False, f"could not connect to {parsed_url.host}:{parsed_url.port}: {exc}"
        )


def create_silent_wav(duration_seconds: float = 0.1, sample_rate: int = 16000) -> Path:
    frame_count = max(1, int(duration_seconds * sample_rate))
    temp_file = tempfile.NamedTemporaryFile(
        prefix="voice_lan_stt_diagnose_",
        suffix=".wav",
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    with wave.open(str(temp_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)

    return temp_path


def check_server(settings: Settings) -> CheckResult:
    try:
        message = WhisperCppClient(settings, timeout=10).test_server()
    except Exception as exc:
        return CheckResult(False, str(exc))

    return CheckResult(True, message)


def check_audio_endpoint(settings: Settings) -> CheckResult:
    wav_path: Path | None = None
    try:
        wav_path = create_silent_wav(sample_rate=settings.sample_rate)
        WhisperCppClient(settings, timeout=20).transcribe(wav_path)
    except EndpointUnsupportedError as exc:
        return CheckResult(False, f"audio endpoint unsupported: {exc}")
    except ModelUnavailableError as exc:
        return CheckResult(False, f"model unavailable: {exc}")
    except ServerUnreachableError as exc:
        return CheckResult(False, f"server unreachable during audio probe: {exc}")
    except Exception as exc:
        return CheckResult(False, f"audio probe failed: {exc}")
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)

    return CheckResult(True, f"POST {settings.inference_path} accepted a short silent WAV")


def run_diagnostics(settings: Settings) -> DiagnosticReport:
    hostname = _safe_hostname()
    report = DiagnosticReport(
        hostname=hostname,
        base_url=settings.base_url,
        inference_path=settings.inference_path,
        model_path=settings.model_path,
        parsed_url=None,
    )

    try:
        report.parsed_url = parse_base_url(settings.base_url)
    except Exception as exc:
        report.url_error = str(exc)
        report.likely_fixes = likely_fixes(report)
        return report

    report.tcp = test_tcp_connection(report.parsed_url)
    if not report.tcp.ok:
        report.server = CheckResult(False, "skipped because TCP connection failed")
        report.audio_endpoint = CheckResult(False, "skipped because TCP connection failed")
        report.likely_fixes = likely_fixes(report)
        return report

    report.server = check_server(settings)
    report.audio_endpoint = check_audio_endpoint(settings)
    report.likely_fixes = likely_fixes(report)
    return report


def format_diagnostic_report(report: DiagnosticReport) -> str:
    lines = [
        "Voice LAN STT diagnostics",
        f"Local hostname: {report.hostname}",
        f"Configured Whisper.cpp base URL: {report.base_url}",
        f"Configured inference path: {report.inference_path}",
        f"Configured server model path: {report.model_path}",
    ]

    if report.parsed_url is None:
        lines.append(f"Parsed target: ERROR - {report.url_error}")
    else:
        lines.append(
            "Parsed target: "
            f"{report.parsed_url.scheme}://{report.parsed_url.host}:{report.parsed_url.port}"
            f"{report.parsed_url.path}"
        )

    lines.extend(
        [
            _format_check("TCP connection", report.tcp),
            _format_check("GET server root", report.server),
        ]
    )

    lines.append(_format_check(f"POST {report.inference_path}", report.audio_endpoint))
    lines.append("Likely fixes:")
    if report.likely_fixes:
        lines.extend(f"- {fix}" for fix in report.likely_fixes)
    else:
        lines.append("- No obvious LAN issue detected by these checks.")

    return "\n".join(lines)


def likely_fixes(report: DiagnosticReport) -> list[str]:
    fixes: list[str] = []

    if report.parsed_url is None:
        return [
            "Set WHISPERCPP_BASE_URL to a full URL such as http://192.168.1.141:8080.",
        ]

    host = report.parsed_url.host
    if host in {"localhost", "127.0.0.1", "::1"}:
        fixes.append(
            "If this client is not the Whisper.cpp machine, use the Whisper.cpp "
            "host IP instead of localhost."
        )

    if not report.tcp.ok:
        fixes.extend(
            [
                "Whisper.cpp server may not be running; start whisper-server.exe.",
                "Wrong IP or port: confirm the server is reachable at "
                f"{host}:{report.parsed_url.port}.",
                "Firewall may be blocking inbound connections; allow the Whisper.cpp "
                "port, usually 8080.",
                "Whisper.cpp may be bound only to localhost; start it with a host that listens "
                "on the LAN interface.",
            ]
        )
        return _dedupe(fixes)

    if not report.server.ok:
        fixes.extend(
            [
                "The TCP port is reachable, but the HTTP server root failed; confirm "
                "whisper-server.exe is the process listening on this port.",
            ]
        )

    if not report.audio_endpoint.ok:
        message = report.audio_endpoint.message.lower()
        if "unsupported" in message or "404" in message:
            fixes.append(
                "Audio endpoint unsupported: confirm whisper-server was started with "
                f"--inference-path {report.inference_path}."
            )
        if "model unavailable" in message:
            fixes.append(
                "Wrong model or no model loaded; restart whisper-server.exe with "
                "the intended ggml model file."
            )
        if "unreachable" in message:
            fixes.append(
                "Connection dropped during the audio probe; recheck firewall, "
                "IP address, and Whisper.cpp server status."
            )

    return _dedupe(fixes)


def _format_check(label: str, result: CheckResult) -> str:
    status = "OK" if result.ok else "FAIL"
    return f"{label}: {status} - {result.message}"


def _safe_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped
