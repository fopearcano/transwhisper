# voice-lan-stt

Minimal Python CLI and desktop GUI that records microphone audio, saves it as a temporary WAV file, sends it to a Whisper.cpp `whisper-server.exe` server on your local network, and prints or displays the returned transcription.

No cloud APIs are used. The app talks only to the Whisper.cpp server URL you configure.

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
- Whisper.cpp `whisper-server.exe` running with a local Whisper model

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

- `WHISPERCPP_BASE_URL=http://192.168.1.141:8080`
- `WHISPERCPP_INFERENCE_PATH=/inference`
- `WHISPERCPP_MODEL_PATH=models/ggml-base.en.bin`
- `WHISPERCPP_LANGUAGE=en`
- `WHISPERCPP_TEMPERATURE=0.0`
- `WHISPERCPP_TEMPERATURE_INC=0.2`
- `WHISPERCPP_RESPONSE_FORMAT=json`
- `SAMPLE_RATE=16000`

For a LAN server, point the base URL at the host machine running `whisper-server.exe`:

```bash
export WHISPERCPP_BASE_URL=http://192.168.1.141:8080
export WHISPERCPP_INFERENCE_PATH=/inference
export WHISPERCPP_MODEL_PATH=models/ggml-base.en.bin
```

CLI flags can override environment variables:

```bash
python -m voice_lan_stt.cli --base-url http://192.168.1.141:8080 record --seconds 5
```

Set the transcription language with `WHISPERCPP_LANGUAGE` or `--language`. Use a whisper.cpp language code such as `en`, `es`, `fr`, `de`, or `auto`:

```bash
python -m voice_lan_stt.cli --language es record --seconds 5
```

`WHISPERCPP_MODEL_PATH` is the model path used when printing server command hints and storing local metadata. The client does not send an OpenAI-style `model` field; the model is loaded by `whisper-server` itself.

## Whisper.cpp LAN Setup Notes

1. Build or download Whisper.cpp with `whisper-server.exe`.
2. Start the server with a model file, host binding, and port `8080`.
3. If another computer will call it over LAN, bind/listen on the LAN interface instead of only `localhost`.
4. Use the host machine IP address, for example `http://192.168.1.141:8080`.
5. Allow inbound traffic through the host firewall on port `8080`.
6. From the client machine, run `test-server` first to confirm the server is reachable.

Example Windows server command:

```powershell
.\whisper-server.exe --host 0.0.0.0 --port 8080 --model .\models\ggml-base.en.bin --inference-path /inference --language en
```

The app can print a matching command from your configured URL, port, model path, inference path, and language:

```bash
python -m voice_lan_stt.cli server-command
```

## Usage

Check that Whisper.cpp is reachable:

```bash
python -m voice_lan_stt.cli test-server
```

Run LAN diagnostics:

```bash
python -m voice_lan_stt.cli diagnose
```

The diagnostic command prints the local hostname, configured Whisper.cpp base URL, parsed host and port, TCP connection result, server root result, and a short `/inference` probe using a generated silent WAV. It never needs microphone access. Use it when a LAN client cannot reach Whisper.cpp; the likely-fixes section calls out common issues like firewall rules, wrong IP address, `whisper-server.exe` not running, server bound only to `localhost`, unsupported `/inference` endpoints, and missing model files on the server.

Record five seconds, send the WAV to `/inference`, and print the transcript:

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

The GUI window title is `TransWhisper - Voive LAN STT`. It provides Whisper.cpp Base URL, inference path, server model path, and language fields, a microphone dropdown with refresh, Start Recording and Stop Recording buttons, a live `MM:SS` timer, a lightweight microphone level meter, Test Server, Copy Latest Transcript, Clear Transcript, a transcript area, and a status label. Recording and transcription run in worker threads, so the window remains responsive. Temporary WAV files are deleted after transcription.

Transcripts are appended chronologically with timestamps, for example:

```text
[14:22:31]
Hello this is a test.
```

GUI settings are saved in `settings.json`, including the last Whisper.cpp URL, inference path, server model path, language, selected microphone, window size, and window position. The GUI does not use a database.

Example LAN call:

```bash
WHISPERCPP_BASE_URL=http://192.168.1.141:8080 python -m voice_lan_stt.cli record --seconds 5
```

## Troubleshooting

- `Could not reach Whisper.cpp`: confirm `whisper-server.exe` is running, the host IP is correct, and the firewall allows port `8080`.
- `The server does not support /inference`: confirm you are pointing at Whisper.cpp `whisper-server.exe`.
- Empty transcript: confirm the server was started with a valid `--model` file and the microphone captured speech.
- `Could not record from the microphone`: confirm a mic is connected and the terminal has microphone permission.
- `Clipboard copy requires the optional dependency pyperclip`: install the clipboard extra with `pip install -e ".[clipboard]"`.

## Tests

```bash
pytest
ruff check .
ruff format --check .
```
