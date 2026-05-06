from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from .config import Settings


class LMStudioError(RuntimeError):
    """Base class for LM Studio client errors."""


class ServerUnreachableError(LMStudioError):
    """Raised when the LM Studio server cannot be reached."""


class ModelUnavailableError(LMStudioError):
    """Raised when the requested STT model is unavailable."""


class EndpointUnsupportedError(LMStudioError):
    """Raised when the target LM Studio endpoint is unsupported."""


class LMStudioClient:
    def __init__(self, settings: Settings, timeout: float = 60.0) -> None:
        self.settings = settings
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.api_key}"}

    def list_models(self) -> list[str]:
        try:
            response = requests.get(
                self.settings.models_url,
                headers=self.headers,
                timeout=10,
            )
        except requests.RequestException as exc:
            raise ServerUnreachableError(
                f"Could not reach LM Studio at {self.settings.base_url}. "
                "Check that the local server is running and reachable over LAN."
            ) from exc

        if response.status_code == 404:
            raise EndpointUnsupportedError(
                f"LM Studio did not expose GET {self.settings.models_url}."
            )
        self._raise_for_error_status(response)

        payload = response.json()
        models = payload.get("data", [])
        return [str(item.get("id", item)) for item in models]

    def transcribe(self, wav_path: Path) -> str:
        try:
            with wav_path.open("rb") as audio_file:
                response = requests.post(
                    self.settings.transcription_url,
                    headers=self.headers,
                    data={"model": self.settings.stt_model},
                    files={"file": (wav_path.name, audio_file, "audio/wav")},
                    timeout=self.timeout,
                )
        except requests.RequestException as exc:
            raise ServerUnreachableError(
                f"Could not reach LM Studio at {self.settings.base_url}. "
                "Check the server URL, host IP, firewall, and port 1234."
            ) from exc

        if response.status_code == 404:
            raise EndpointUnsupportedError(
                "The server does not support /audio/transcriptions. "
                "Confirm your LM Studio version and that an STT/Whisper model is loaded."
            )
        if response.status_code in {400, 404, 422} and self._mentions_model(response):
            raise ModelUnavailableError(
                f"Model {self.settings.stt_model!r} is unavailable. Load it in LM Studio "
                "or set LMSTUDIO_STT_MODEL to an available model."
            )
        self._raise_for_error_status(response)

        payload = response.json()
        transcript = payload.get("text")
        if not isinstance(transcript, str):
            raise LMStudioError("The transcription response did not include a text field.")
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
            raise LMStudioError(message) from exc


def _response_error_message(response: requests.Response) -> str:
    details: Any
    try:
        details = response.json()
    except ValueError:
        details = response.text

    return f"LM Studio returned HTTP {response.status_code}. Response: {details!r}"
