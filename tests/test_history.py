from __future__ import annotations

import json

from voice_lan_stt.history import HistoryStore, export_records, format_history


def test_history_insert_and_search(tmp_path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite3")

    first = store.insert(
        mode="fixed",
        duration_seconds=5.0,
        transcript_text="hello from the microphone",
        audio_path=None,
        lmstudio_base_url="http://192.168.1.141:8080",
        model="whisper.cpp",
        created_at="2026-05-06T10:00:00+00:00",
    )
    store.insert(
        mode="listen",
        duration_seconds=1.25,
        transcript_text="different phrase",
        audio_path=tmp_path / "clip.wav",
        lmstudio_base_url="http://192.168.1.141:8080",
        model="whisper-lan",
        created_at="2026-05-06T10:01:00+00:00",
    )

    results = store.search(limit=10, keyword="microphone")

    assert len(results) == 1
    assert results[0].id == first.id
    assert results[0].mode == "fixed"
    assert results[0].transcript_text == "hello from the microphone"
    assert results[0].audio_path is None


def test_history_limit_orders_newest_first(tmp_path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite3")
    for index in range(3):
        store.insert(
            mode="PTT",
            duration_seconds=float(index),
            transcript_text=f"entry {index}",
            audio_path=None,
            lmstudio_base_url="http://192.168.1.141:8080",
            model="whisper.cpp",
            created_at=f"2026-05-06T10:0{index}:00+00:00",
        )

    results = store.search(limit=2)

    assert [record.transcript_text for record in results] == ["entry 2", "entry 1"]


def test_export_records_as_json(tmp_path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.insert(
        mode="listen",
        duration_seconds=2.5,
        transcript_text="json export text",
        audio_path=None,
        lmstudio_base_url="http://192.168.1.141:8080",
        model="whisper.cpp",
        created_at="2026-05-06T10:00:00+00:00",
    )

    payload = json.loads(export_records(store.all(), "json"))

    assert payload[0]["mode"] == "listen"
    assert payload[0]["transcript_text"] == "json export text"
    assert payload[0]["duration_seconds"] == 2.5


def test_export_records_as_txt(tmp_path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.insert(
        mode="fixed",
        duration_seconds=3.0,
        transcript_text="plain text export",
        audio_path=None,
        lmstudio_base_url="http://192.168.1.141:8080",
        model="whisper.cpp",
        created_at="2026-05-06T10:00:00+00:00",
    )

    exported = export_records(store.all(), "txt")

    assert "plain text export" in exported
    assert "fixed" in exported
    assert "3.00s" in exported


def test_format_history_empty() -> None:
    assert format_history([]) == "No transcript history found."
