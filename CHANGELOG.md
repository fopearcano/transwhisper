# Changelog

## Unreleased

- Switched the default server target from LM Studio to Whisper.cpp `whisper-server.exe`.
- Changed the default server URL to `http://192.168.1.141:8080`.
- Updated the desktop title to `TransWhisper - Voive LAN STT`.

## 0.1.0

- Added CLI recording modes: fixed duration, push-to-talk, and continuous listen with basic RMS VAD.
- Added Whisper.cpp LAN transcription client and server diagnostics.
- Added local SQLite transcript history and export commands.
- Added optional PySide6 desktop GUI.
- Upgraded the GUI with Start/Stop recording, live timer, microphone level meter, microphone device selection, local `settings.json`, timestamped transcript appends, and worker-thread recording/transcription.
- Added tests, Ruff configuration, and GitHub Actions workflow.
