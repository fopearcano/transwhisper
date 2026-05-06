from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from voice_lan_stt.config import Settings
from voice_lan_stt.lmstudio_client import (
    EndpointUnsupportedError,
    ModelUnavailableError,
    ServerUnreachableError,
    WhisperCppClient,
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


def test_server_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    response = make_response(200, text="<html>server</html>")
    get = Mock(return_value=response)
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.get", get)

    client = WhisperCppClient(Settings())

    assert "Whisper.cpp server reachable" in client.test_server()
    get.assert_called_once()
    assert get.call_args.kwargs["headers"] == {}


def test_server_reachable_with_missing_root(monkeypatch: pytest.MonkeyPatch) -> None:
    response = make_response(404, text="not found")
    get = Mock(return_value=response)
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.get", get)

    client = WhisperCppClient(Settings())

    assert "HTTP 404" in client.test_server()


def test_list_models_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    get = Mock(side_effect=requests.ConnectionError("no route"))
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.get", get)

    client = WhisperCppClient(Settings(base_url="http://192.168.1.141:8080"))

    with pytest.raises(ServerUnreachableError, match="Could not reach Whisper.cpp"):
        client.test_server()


def test_transcribe_returns_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFFfake")
    response = make_response(200, {"text": "hello lan"})
    post = Mock(return_value=response)
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.post", post)

    client = WhisperCppClient(Settings(stt_model="whisper-local"))

    assert client.transcribe(wav_path) == "hello lan"
    post.assert_called_once()
    assert post.call_args.kwargs["data"] == {
        "temperature": "0.0",
        "temperature_inc": "0.2",
        "response_format": "json",
    }
    assert post.call_args.kwargs["headers"] == {}


def test_transcribe_returns_plain_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFFfake")
    response = make_response(200, None, text="plain transcript")
    response.json.side_effect = ValueError("not json")
    post = Mock(return_value=response)
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.post", post)

    client = WhisperCppClient(Settings())

    assert client.transcribe(wav_path) == "plain transcript"


def test_transcribe_unsupported_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFFfake")
    response = make_response(404, {"error": "not found"}, text="not found")
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.post", Mock(return_value=response))

    client = WhisperCppClient(Settings())

    with pytest.raises(EndpointUnsupportedError, match="/inference"):
        client.transcribe(wav_path)


def test_transcribe_model_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFFfake")
    response = make_response(422, {"error": "model unavailable"}, text="model unavailable")
    monkeypatch.setattr("voice_lan_stt.lmstudio_client.requests.post", Mock(return_value=response))

    client = WhisperCppClient(Settings(stt_model="missing-model"))

    with pytest.raises(ModelUnavailableError, match="missing-model"):
        client.transcribe(wav_path)
