from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from voice_lan_stt.config import Settings
from voice_lan_stt.lmstudio_client import (
    EndpointUnsupportedError,
    LMStudioClient,
    ModelUnavailableError,
    ServerUnreachableError,
)


def make_response(status_code: int, payload: dict | None = None, text: str = "") -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.text = text
    response.json.return_value = payload if payload is not None else {}
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError("boom", response=response)
    else:
        response.raise_for_status.return_value = None
    return response


def test_list_models(monkeypatch: pytest.MonkeyPatch) -> None:
    response = make_response(200, {"data": [{"id": "whisper-1"}, {"id": "other"}]})
    get = Mock(return_value=response)
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.get", get)

    client = LMStudioClient(Settings())

    assert client.list_models() == ["whisper-1", "other"]
    get.assert_called_once()
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer lm-studio"}


def test_list_models_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    get = Mock(side_effect=requests.ConnectionError("no route"))
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.get", get)

    client = LMStudioClient(Settings(base_url="http://192.168.1.50:1234/v1"))

    with pytest.raises(ServerUnreachableError, match="Could not reach LM Studio"):
        client.list_models()


def test_transcribe_returns_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFFfake")
    response = make_response(200, {"text": "hello lan"})
    post = Mock(return_value=response)
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.post", post)

    client = LMStudioClient(Settings(stt_model="whisper-local"))

    assert client.transcribe(wav_path) == "hello lan"
    post.assert_called_once()
    assert post.call_args.kwargs["data"] == {"model": "whisper-local"}
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer lm-studio"}


def test_transcribe_unsupported_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFFfake")
    response = make_response(404, {"error": "not found"}, text="not found")
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.post", Mock(return_value=response))

    client = LMStudioClient(Settings())

    with pytest.raises(EndpointUnsupportedError, match="/audio/transcriptions"):
        client.transcribe(wav_path)


def test_transcribe_model_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFFfake")
    response = make_response(422, {"error": "model unavailable"}, text="model unavailable")
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.post", Mock(return_value=response))

    client = LMStudioClient(Settings(stt_model="missing-model"))

    with pytest.raises(ModelUnavailableError, match="missing-model"):
        client.transcribe(wav_path)
