from __future__ import annotations

import socket
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .config import Settings
from .lmstudio_client import (
    EndpointUnsupportedError,
    LMStudioClient,
    ModelUnavailableError,
    ServerUnreachableError,
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
    model: str
    parsed_url: ParsedBaseUrl | None
    url_error: str | None = None
    tcp: CheckResult = field(default_factory=lambda: CheckResult(False, "not run"))
    models: CheckResult = field(default_factory=lambda: CheckResult(False, "not run"))
    audio_endpoint: CheckResult = field(default_factory=lambda: CheckResult(False, "not run"))
    available_models: list[str] = field(default_factory=list)
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


def check_models(settings: Settings) -> tuple[CheckResult, list[str]]:
    try:
        models = LMStudioClient(settings, timeout=10).list_models()
    except Exception as exc:
        return CheckResult(False, str(exc)), []

    if models:
        return CheckResult(True, f"GET /models returned {len(models)} model(s)"), models
    return CheckResult(True, "GET /models succeeded but returned no models"), []


def check_audio_endpoint(settings: Settings) -> CheckResult:
    wav_path: Path | None = None
    try:
        wav_path = create_silent_wav(sample_rate=settings.sample_rate)
        LMStudioClient(settings, timeout=20).transcribe(wav_path)
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

    return CheckResult(True, "/audio/transcriptions accepted a short silent WAV")


def run_diagnostics(settings: Settings) -> DiagnosticReport:
    hostname = _safe_hostname()
    report = DiagnosticReport(
        hostname=hostname,
        base_url=settings.base_url,
        model=settings.stt_model,
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
        report.models = CheckResult(False, "skipped because TCP connection failed")
        report.audio_endpoint = CheckResult(False, "skipped because TCP connection failed")
        report.likely_fixes = likely_fixes(report)
        return report

    report.models, report.available_models = check_models(settings)
    report.audio_endpoint = check_audio_endpoint(settings)
    report.likely_fixes = likely_fixes(report)
    return report


def format_diagnostic_report(report: DiagnosticReport) -> str:
    lines = [
        "Voice LAN STT diagnostics",
        f"Local hostname: {report.hostname}",
        f"Configured LM Studio base URL: {report.base_url}",
        f"Configured model: {report.model}",
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
            _format_check("GET /models", report.models),
        ]
    )

    if report.available_models:
        lines.append("Available models:")
        for model in report.available_models:
            marker = " (configured)" if model == report.model else ""
            lines.append(f"- {model}{marker}")

    lines.append(_format_check("POST /audio/transcriptions", report.audio_endpoint))
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
            "Set LMSTUDIO_BASE_URL to a full URL such as http://192.168.1.50:1234/v1.",
        ]

    host = report.parsed_url.host
    if host in {"localhost", "127.0.0.1", "::1"}:
        fixes.append(
            "If this client is not the LM Studio machine, use the LM Studio "
            "host IP instead of localhost."
        )

    if not report.tcp.ok:
        fixes.extend(
            [
                "LM Studio server may not be running; enable the local server in LM Studio.",
                "Wrong IP or port: confirm the server is reachable at "
                f"{host}:{report.parsed_url.port}.",
                "Firewall may be blocking inbound connections; allow the LM Studio "
                "port, usually 1234.",
                "LM Studio may be bound only to localhost; configure it to listen "
                "on the LAN interface.",
            ]
        )
        return _dedupe(fixes)

    if not report.models.ok:
        fixes.extend(
            [
                "The server is reachable, but the OpenAI-compatible /models route "
                "failed; confirm the base URL ends with /v1.",
                "Confirm LM Studio's local server is enabled and serving "
                "OpenAI-compatible endpoints.",
            ]
        )

    if report.available_models and report.model not in report.available_models:
        fixes.append(
            f"Wrong model name: configured model {report.model!r} was not listed by GET /models."
        )

    if not report.audio_endpoint.ok:
        message = report.audio_endpoint.message.lower()
        if "unsupported" in message or "404" in message:
            fixes.append(
                "Audio endpoint unsupported: update LM Studio or load/use an "
                "STT-capable Whisper model."
            )
        if "model unavailable" in message:
            fixes.append(
                "Wrong model name or no STT model loaded; set LMSTUDIO_STT_MODEL "
                "to a listed Whisper/STT model."
            )
        if "unreachable" in message:
            fixes.append(
                "Connection dropped during the audio probe; recheck firewall, "
                "IP address, and LM Studio server status."
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
