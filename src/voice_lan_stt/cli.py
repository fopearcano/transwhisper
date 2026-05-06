from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings, load_settings
from .diagnostics import format_diagnostic_report, run_diagnostics
from .history import (
    HistoryStore,
    export_records,
    format_history,
    save_transcript_record,
)
from .lmstudio_client import (
    EndpointUnsupportedError,
    LMStudioClient,
    LMStudioError,
    ModelUnavailableError,
    ServerUnreachableError,
)
from .recorder import (
    MicrophoneError,
    VadOptions,
    listen_for_vad_segments,
    record_to_temp_wav,
    record_until_enter_to_temp_wav,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-lan-stt",
        description="Record microphone audio and transcribe it through LM Studio over LAN.",
    )
    parser.add_argument("--base-url", help="LM Studio base URL, e.g. http://192.168.1.50:1234/v1")
    parser.add_argument(
        "--api-key", help="LM Studio API key. Defaults to LMSTUDIO_API_KEY or lm-studio."
    )
    parser.add_argument(
        "--model", help="STT model name. Defaults to LMSTUDIO_STT_MODEL or whisper-1."
    )
    parser.add_argument(
        "--sample-rate", type=int, help="Microphone sample rate. Defaults to SAMPLE_RATE or 16000."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Record audio and print transcription.")
    record_parser.add_argument(
        "--seconds", type=float, default=5.0, help="Recording duration in seconds."
    )
    record_parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the recorded WAV file in the app data directory.",
    )

    ptt_parser = subparsers.add_parser("ptt", help="Push-to-talk recording with Enter start/stop.")
    ptt_parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the transcript to the clipboard using pyperclip.",
    )
    ptt_parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the recorded WAV file in the app data directory.",
    )

    listen_parser = subparsers.add_parser(
        "listen",
        help="Continuously listen and transcribe speech segments with basic RMS VAD.",
    )
    listen_parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep each recorded WAV segment in the app data directory.",
    )
    listen_parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="RMS speech threshold on normalized int16 audio. Defaults to 0.01.",
    )
    listen_parser.add_argument(
        "--silence-ms",
        type=int,
        default=800,
        help="Milliseconds of below-threshold audio before a segment ends.",
    )
    listen_parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=250,
        help="Minimum above-threshold speech duration required before transcription.",
    )
    listen_parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=15.0,
        help="Maximum segment duration before forcing transcription.",
    )
    listen_parser.add_argument(
        "--sample-rate",
        type=int,
        default=argparse.SUPPRESS,
        help="Microphone sample rate. Defaults to SAMPLE_RATE or 16000.",
    )

    history_parser = subparsers.add_parser("history", help="Show local transcript history.")
    history_parser.add_argument("--limit", type=int, default=10, help="Maximum rows to show.")
    history_parser.add_argument("--search", help="Search transcript text.")

    export_parser = subparsers.add_parser("export", help="Export local transcript history.")
    export_parser.add_argument("--format", choices=("txt", "json"), required=True)
    export_parser.add_argument("--search", help="Search transcript text before exporting.")

    subparsers.add_parser("diagnose", help="Run LAN connectivity diagnostics.")
    subparsers.add_parser("test-server", help="List models from GET /models.")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    return Settings(
        base_url=(args.base_url or settings.base_url).rstrip("/"),
        api_key=args.api_key or settings.api_key,
        stt_model=args.model or settings.stt_model,
        sample_rate=args.sample_rate or settings.sample_rate,
    )


def save_history(
    *,
    settings: Settings,
    mode: str,
    wav_path: Path,
    transcript: str,
    keep_audio: bool,
) -> None:
    save_transcript_record(
        mode=mode,
        lmstudio_base_url=settings.base_url,
        model=settings.stt_model,
        wav_path=wav_path,
        transcript_text=transcript,
        keep_audio=keep_audio,
    )


def run_record(settings: Settings, seconds: float, keep_audio: bool = False) -> int:
    client = LMStudioClient(settings)
    wav_path: Path | None = None

    try:
        wav_path = record_to_temp_wav(seconds=seconds, sample_rate=settings.sample_rate)
        transcript = client.transcribe(wav_path)
        save_history(
            settings=settings,
            mode="fixed",
            wav_path=wav_path,
            transcript=transcript,
            keep_audio=keep_audio,
        )
    except (
        MicrophoneError,
        ServerUnreachableError,
        ModelUnavailableError,
        EndpointUnsupportedError,
        LMStudioError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)

    print(transcript)
    return 0


def copy_to_clipboard(text: str) -> None:
    try:
        import pyperclip
    except ImportError as exc:
        raise RuntimeError(
            "Clipboard copy requires the optional dependency pyperclip. "
            "Install with: pip install 'voice-lan-stt[clipboard]'"
        ) from exc

    try:
        pyperclip.copy(text)
    except pyperclip.PyperclipException as exc:
        raise RuntimeError(f"Could not copy transcript to clipboard: {exc}") from exc


def run_ptt(settings: Settings, copy: bool = False, keep_audio: bool = False) -> int:
    client = LMStudioClient(settings)
    wav_path: Path | None = None

    try:
        print("READY")
        input("Press Enter to start recording.")
        print("RECORDING")
        wav_path = record_until_enter_to_temp_wav(
            sample_rate=settings.sample_rate,
            wait_for_stop=lambda: input("Press Enter to stop recording."),
        )
        print("TRANSCRIBING")
        transcript = client.transcribe(wav_path)
        save_history(
            settings=settings,
            mode="PTT",
            wav_path=wav_path,
            transcript=transcript,
            keep_audio=keep_audio,
        )
    except (
        MicrophoneError,
        ServerUnreachableError,
        ModelUnavailableError,
        EndpointUnsupportedError,
        LMStudioError,
        RuntimeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)

    print("TRANSCRIPT")
    print(transcript)
    if copy:
        try:
            copy_to_clipboard(transcript)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    return 0


def run_listen(
    settings: Settings,
    threshold: float,
    silence_ms: int,
    min_speech_ms: int,
    max_segment_seconds: float,
    keep_audio: bool = False,
) -> int:
    client = LMStudioClient(settings)
    options = VadOptions(
        sample_rate=settings.sample_rate,
        threshold=threshold,
        silence_ms=silence_ms,
        min_speech_ms=min_speech_ms,
        max_segment_seconds=max_segment_seconds,
    )

    try:
        print("LISTENING")
        print("Press Ctrl+C to stop.")
        for wav_path in listen_for_vad_segments(options):
            try:
                print("TRANSCRIBING")
                transcript = client.transcribe(wav_path)
                save_history(
                    settings=settings,
                    mode="listen",
                    wav_path=wav_path,
                    transcript=transcript,
                    keep_audio=keep_audio,
                )
            finally:
                wav_path.unlink(missing_ok=True)

            print("TRANSCRIPT")
            print(transcript)
    except KeyboardInterrupt:
        print("Stopped.")
        return 0
    except (
        MicrophoneError,
        ServerUnreachableError,
        ModelUnavailableError,
        EndpointUnsupportedError,
        LMStudioError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


def run_history(limit: int, search: str | None = None) -> int:
    try:
        records = HistoryStore().search(limit=limit, keyword=search)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_history(records))
    return 0


def run_export(export_format: str, search: str | None = None) -> int:
    try:
        records = HistoryStore().all(keyword=search)
        print(export_records(records, export_format))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def run_diagnose(settings: Settings) -> int:
    try:
        report = run_diagnostics(settings)
        print(format_diagnostic_report(report))
    except Exception as exc:
        print("Voice LAN STT diagnostics")
        print(f"Diagnostic failed unexpectedly: {exc}")
        print("Likely fixes:")
        print("- Check LMSTUDIO_BASE_URL, LMSTUDIO_STT_MODEL, and SAMPLE_RATE.")
        print("- Confirm LM Studio local server is running and reachable.")
    return 0


def run_test_server(settings: Settings) -> int:
    client = LMStudioClient(settings)
    try:
        models = client.list_models()
    except (ServerUnreachableError, EndpointUnsupportedError, LMStudioError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if models:
        print("Available models:")
        for model in models:
            print(f"- {model}")
    else:
        print("Server reachable, but no models were returned.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "history":
        return run_history(limit=args.limit, search=args.search)
    if args.command == "export":
        return run_export(export_format=args.format, search=args.search)

    try:
        settings = settings_from_args(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.command == "record":
        return run_record(
            settings=settings,
            seconds=args.seconds,
            keep_audio=args.keep_audio,
        )
    if args.command == "ptt":
        return run_ptt(settings=settings, copy=args.copy, keep_audio=args.keep_audio)
    if args.command == "listen":
        return run_listen(
            settings=settings,
            threshold=args.threshold,
            silence_ms=args.silence_ms,
            min_speech_ms=args.min_speech_ms,
            max_segment_seconds=args.max_segment_seconds,
            keep_audio=args.keep_audio,
        )
    if args.command == "diagnose":
        return run_diagnose(settings=settings)
    if args.command == "test-server":
        return run_test_server(settings=settings)

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
