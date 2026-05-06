from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from .config import Settings


class WhisperCppError(RuntimeError):
    """Base class for Whisper.cpp client errors."""


class ServerUnreachableError(WhisperCppError):
    """Raised when the Whisper.cpp server cannot be reached."""


class ModelUnavailableError(WhisperCppError):
    """Raised when the requested STT model is unavailable."""


class EndpointUnsupportedError(WhisperCppError):
    """Raised when the target Whisper.cpp endpoint is unsupported."""


class WhisperCppClient:
    def __init__(self, settings: Settings, timeout: float = 60.0) -> None:
        self.settings = settings
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        if not self.settings.api_key:
            return {}
        return {"Authorization": f"Bearer {self.settings.api_key}"}

    def test_server(self) -> str:
        try:
            response = requests.get(
                self.settings.server_url,
                headers=self.headers,
                timeout=10,
            )
        except requests.RequestException as exc:
            raise ServerUnreachableError(
                f"Could not reach Whisper.cpp at {self.settings.base_url}. "
                "Check that the local server is running and reachable over LAN."
            ) from exc

        if response.status_code >= 500:
            self._raise_for_error_status(response)
        return (
            f"Whisper.cpp server reachable at {self.settings.base_url} "
            f"(HTTP {response.status_code})."
        )

    def list_models(self) -> list[str]:
        self.test_server()
        return [self.settings.stt_model]

    def transcribe(self, wav_path: Path) -> str:
        try:
            with wav_path.open("rb") as audio_file:
                response = requests.post(
                    self.settings.transcription_url,
                    headers=self.headers,
                    data={
                        "temperature": "0.0",
                        "temperature_inc": "0.2",
                        "response_format": "json",
                    },
                    files={"file": (wav_path.name, audio_file, "audio/wav")},
                    timeout=self.timeout,
                )
        except requests.RequestException as exc:
            raise ServerUnreachableError(
                f"Could not reach Whisper.cpp at {self.settings.base_url}. "
                "Check the server URL, host IP, firewall, and port 8080."
            ) from exc

        if response.status_code == 404:
            raise EndpointUnsupportedError(
                "The server does not support /inference. "
                "Confirm whisper-server.exe is running and reachable on the configured port."
            )
        if response.status_code in {400, 404, 422} and self._mentions_model(response):
            raise ModelUnavailableError(
                f"Model {self.settings.stt_model!r} is unavailable. Start whisper-server.exe "
                "with the intended model file."
            )
        self._raise_for_error_status(response)

        transcript = _transcript_from_response(response)
        if not isinstance(transcript, str):
            raise WhisperCppError("The transcription response did not include a text field.")
        return transcript

    @staticmethod
    def _mentions_model(response: requests.Response) -> bool:
        body = response.text.lower()
        return "model" in body and (
            "not found" in body or "unknown" in body or "unavailable" in body
        )

    @staticmethod
    def _raise_for_error_status(response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = _response_error_message(response)
            raise WhisperCppError(message) from exc


def _response_error_message(response: requests.Response) -> str:
    details: Any
    try:
        details = response.json()
    except ValueError:
        details = response.text

    return f"Whisper.cpp returned HTTP {response.status_code}. Response: {details!r}"


def _transcript_from_response(response: requests.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()

    if isinstance(payload, dict):
        text = payload.get("text")
        return text if isinstance(text, str) else None
    return None


LMStudioError = WhisperCppError
LMStudioClient = WhisperCppClient
