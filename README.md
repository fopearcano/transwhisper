# voice-lan-stt

Minimal Python CLI that records microphone audio, saves it as a temporary WAV file, sends it to an LM Studio server on your local network, and prints the returned Whisper/STT transcription.

No cloud APIs are used. The CLI talks only to the LM Studio-compatible server URL you configure.

## Quickstart

```bash
cd voice_lan_stt
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python -m voice_lan_stt.cli diagnose
python -m voice_lan_stt.cli record --seconds 5
```

On Windows PowerShell:

```powershell
cd voice_lan_stt
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
python -m voice_lan_stt.cli diagnose
python -m voice_lan_stt.cli record --seconds 5
```

## Requirements

- Python 3.10+
- Windows or Linux
- A microphone available to PortAudio/sounddevice
- LM Studio running a Whisper/STT model with the local server enabled

## Install

```bash
cd voice_lan_stt
python -m venv .venv
source .venv/bin/activate  # Linux
pip install -e ".[dev]"
```

Clipboard copy for push-to-talk is optional:

```bash
pip install -e ".[clipboard]"
```

The desktop GUI is optional:

```bash
pip install -e ".[gui]"
```

On Windows PowerShell:

```powershell
cd voice_lan_stt
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

For clipboard copy on Windows:

```powershell
pip install -e ".[clipboard]"
```

For the desktop GUI on Windows:

```powershell
pip install -e ".[gui]"
```

## Configure

Defaults:

- `LMSTUDIO_BASE_URL=http://localhost:1234/v1`
- `LMSTUDIO_API_KEY=lm-studio`
- `LMSTUDIO_STT_MODEL=whisper-1`
- `SAMPLE_RATE=16000`

For a LAN server, point the base URL at the host machine running LM Studio:

```bash
export LMSTUDIO_BASE_URL=http://192.168.1.50:1234/v1
export LMSTUDIO_STT_MODEL=whisper-1
```

CLI flags can override environment variables:

```bash
python -m voice_lan_stt.cli --base-url http://192.168.1.50:1234/v1 --model whisper-1 record --seconds 5
```

## LM Studio LAN Setup Notes

1. In LM Studio, load a Whisper/STT-capable model.
2. Enable the local server.
3. If another computer will call it over LAN, configure LM Studio to bind/listen on the LAN interface if needed instead of only `localhost`.
4. Use the host machine IP address, for example `http://192.168.1.50:1234/v1`.
5. Allow inbound traffic through the host firewall on port `1234`.
6. From the client machine, run `test-server` first to confirm the server is reachable.

## Usage

Check that LM Studio is reachable and list available models:

```bash
python -m voice_lan_stt.cli test-server
```

Run LAN diagnostics:

```bash
python -m voice_lan_stt.cli diagnose
```

The diagnostic command prints the local hostname, configured LM Studio base URL, parsed host and port, TCP connection result, `GET /models` result, and a short `/audio/transcriptions` probe using a generated silent WAV. It never needs microphone access. Use it when a LAN client cannot reach LM Studio; the likely-fixes section calls out common issues like firewall rules, wrong IP address, LM Studio not running, LM Studio bound only to `localhost`, unsupported audio endpoints, and wrong model names.

Record five seconds, send the WAV to `/audio/transcriptions`, and print the transcript:

```bash
python -m voice_lan_stt.cli record --seconds 5
```

Keep the WAV file for a recording:

```bash
python -m voice_lan_stt.cli record --seconds 5 --keep-audio
```

Use push-to-talk mode. Press Enter once to start recording, then press Enter again to stop and transcribe:

```bash
python -m voice_lan_stt.cli ptt
```

Copy the transcript to the clipboard after transcription:

```bash
python -m voice_lan_stt.cli ptt --copy
```

Continuously listen and transcribe detected speech segments:

```bash
python -m voice_lan_stt.cli listen
```

Keep WAV files for detected speech segments:

```bash
python -m voice_lan_stt.cli listen --keep-audio
```

Tune the simple VAD:

```bash
python -m voice_lan_stt.cli listen --threshold 0.015 --silence-ms 700 --min-speech-ms 300 --max-segment-seconds 12
```

The `listen` command uses basic RMS-based voice activity detection. It is not neural VAD, so it will not understand speech semantically and can be fooled by background noise, fans, keyboard sounds, or quiet speakers. Raise `--threshold` if it triggers on noise; lower it if it misses speech. Increase `--silence-ms` if it cuts phrases too aggressively.

Show local transcript history:

```bash
python -m voice_lan_stt.cli history
python -m voice_lan_stt.cli history --limit 20
python -m voice_lan_stt.cli history --search "keyword"
```

Export transcript history:

```bash
python -m voice_lan_stt.cli export --format txt
python -m voice_lan_stt.cli export --format json
```

Transcripts are stored in a local SQLite database. By default, temporary WAV files are deleted after transcription and only transcript metadata is kept. Pass `--keep-audio` to retain audio files in the app data directory.

Default app data locations:

- Windows: `%APPDATA%\voice_lan_stt`
- Linux: `~/.local/share/voice_lan_stt`, or `$XDG_DATA_HOME/voice_lan_stt`
- macOS: `~/Library/Application Support/voice_lan_stt`

## Desktop GUI

Launch the minimal PySide6 desktop UI:

```bash
python gui.py
```

You can also run it as a module after installing the package:

```bash
python -m voice_lan_stt.gui
```

The GUI provides LM Studio Base URL and model fields, a microphone dropdown with refresh, Start Recording and Stop Recording buttons, a live `MM:SS` timer, a lightweight microphone level meter, Test Server, Copy Latest Transcript, Clear Transcript, a transcript area, and a status label. Recording and transcription run in worker threads, so the window remains responsive. Temporary WAV files are deleted after transcription.

Transcripts are appended chronologically with timestamps, for example:

```text
[14:22:31]
Hello this is a test.
```

GUI settings are saved in `settings.json`, including the last LM Studio URL, selected microphone, selected model, window size, and window position. The GUI does not use a database.

Example LAN call:

```bash
LMSTUDIO_BASE_URL=http://192.168.1.50:1234/v1 python -m voice_lan_stt.cli record --seconds 5
```

## Troubleshooting

- `Could not reach LM Studio`: confirm LM Studio server is running, the base URL includes `/v1`, the host IP is correct, and the firewall allows port `1234`.
- `The server does not support /audio/transcriptions`: confirm your LM Studio version exposes the OpenAI-compatible audio transcription endpoint and that an STT/Whisper model is loaded.
- `Model '...' is unavailable`: run `test-server`, then set `LMSTUDIO_STT_MODEL` or `--model` to an available STT model.
- `Could not record from the microphone`: confirm a mic is connected and the terminal has microphone permission.
- `Clipboard copy requires the optional dependency pyperclip`: install the clipboard extra with `pip install -e ".[clipboard]"`.

## Tests

```bash
pytest
ruff check .
ruff format --check .
```
