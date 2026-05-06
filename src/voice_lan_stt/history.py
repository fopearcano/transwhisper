from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import uuid
import wave
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

APP_DIR_NAME = "voice_lan_stt"
DB_FILENAME = "history.sqlite3"


@dataclass(frozen=True)
class TranscriptRecord:
    id: int
    created_at: str
    mode: str
    duration_seconds: float | None
    transcript_text: str
    audio_path: str | None
    lmstudio_base_url: str
    model: str


def user_data_dir() -> Path:
    if sys.platform == "win32":
        root = os.getenv("APPDATA")
        if root:
            return Path(root) / APP_DIR_NAME
        return Path.home() / "AppData" / "Roaming" / APP_DIR_NAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME

    root = os.getenv("XDG_DATA_HOME")
    if root:
        return Path(root) / APP_DIR_NAME
    return Path.home() / ".local" / "share" / APP_DIR_NAME


def default_db_path() -> Path:
    return user_data_dir() / DB_FILENAME


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
    if frame_rate <= 0:
        return 0.0
    return frame_count / frame_rate


def preserve_audio_file(wav_path: Path, data_dir: Path | None = None) -> Path:
    target_dir = (data_dir or user_data_dir()) / "audio"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{utc_now_iso().replace(':', '-')}_{uuid.uuid4().hex}.wav"
    shutil.move(str(wav_path), target_path)
    return target_path


def save_transcript_record(
    *,
    mode: str,
    wav_path: Path,
    transcript_text: str,
    lmstudio_base_url: str,
    model: str,
    keep_audio: bool = False,
    store: HistoryStore | None = None,
) -> TranscriptRecord:
    audio_path = preserve_audio_file(wav_path) if keep_audio else None
    duration_seconds = wav_duration_seconds(audio_path or wav_path)
    return (store or HistoryStore()).insert(
        mode=mode,
        duration_seconds=duration_seconds,
        transcript_text=transcript_text,
        audio_path=audio_path,
        lmstudio_base_url=lmstudio_base_url,
        model=model,
    )


class HistoryStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_db_path()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    duration_seconds REAL,
                    transcript_text TEXT NOT NULL,
                    audio_path TEXT,
                    lmstudio_base_url TEXT NOT NULL,
                    model TEXT NOT NULL
                )
                """
            )

    def insert(
        self,
        *,
        mode: str,
        duration_seconds: float | None,
        transcript_text: str,
        audio_path: Path | None,
        lmstudio_base_url: str,
        model: str,
        created_at: str | None = None,
    ) -> TranscriptRecord:
        self.initialize()
        timestamp = created_at or utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO transcripts (
                    created_at,
                    mode,
                    duration_seconds,
                    transcript_text,
                    audio_path,
                    lmstudio_base_url,
                    model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    mode,
                    duration_seconds,
                    transcript_text,
                    str(audio_path) if audio_path is not None else None,
                    lmstudio_base_url,
                    model,
                ),
            )
            record_id = int(cursor.lastrowid)

        return TranscriptRecord(
            id=record_id,
            created_at=timestamp,
            mode=mode,
            duration_seconds=duration_seconds,
            transcript_text=transcript_text,
            audio_path=str(audio_path) if audio_path is not None else None,
            lmstudio_base_url=lmstudio_base_url,
            model=model,
        )

    def search(self, *, limit: int = 10, keyword: str | None = None) -> list[TranscriptRecord]:
        self.initialize()
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        sql = "SELECT * FROM transcripts"
        params: list[object] = []
        if keyword:
            sql += " WHERE transcript_text LIKE ?"
            params.append(f"%{keyword}%")
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._record_from_row(row) for row in rows]

    def all(self, *, keyword: str | None = None) -> list[TranscriptRecord]:
        self.initialize()
        sql = "SELECT * FROM transcripts"
        params: list[object] = []
        if keyword:
            sql += " WHERE transcript_text LIKE ?"
            params.append(f"%{keyword}%")
        sql += " ORDER BY id ASC"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._record_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> TranscriptRecord:
        return TranscriptRecord(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            mode=str(row["mode"]),
            duration_seconds=(
                float(row["duration_seconds"]) if row["duration_seconds"] is not None else None
            ),
            transcript_text=str(row["transcript_text"]),
            audio_path=str(row["audio_path"]) if row["audio_path"] is not None else None,
            lmstudio_base_url=str(row["lmstudio_base_url"]),
            model=str(row["model"]),
        )


def format_history(records: Sequence[TranscriptRecord]) -> str:
    if not records:
        return "No transcript history found."

    lines: list[str] = []
    for record in records:
        duration = (
            f"{record.duration_seconds:.2f}s" if record.duration_seconds is not None else "unknown"
        )
        lines.append(
            f"#{record.id} {record.created_at} {record.mode} {duration} model={record.model}"
        )
        lines.append(record.transcript_text)
        if record.audio_path:
            lines.append(f"audio={record.audio_path}")
        lines.append("")
    return "\n".join(lines).rstrip()


def export_records(records: Sequence[TranscriptRecord], export_format: str) -> str:
    if export_format == "json":
        return json.dumps([asdict(record) for record in records], indent=2)
    if export_format == "txt":
        return format_history(records)
    raise ValueError(f"unsupported export format {export_format!r}")
