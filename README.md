# kokoro-tts-service

A self-hosted text-to-speech HTTP service built on
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). It exposes an
OpenAI-compatible `POST /v1/audio/speech` endpoint that returns WAV audio, so
local voice clients can use a familiar API without a hosted TTS provider. Named
voice profiles, OpenAI voice aliases and weighted voice blends are managed by
the service.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Network access on first run to download the Kokoro weights (about 330 MB)
  from Hugging Face. `GET /health` and `GET /v1/audio/voices` also call
  `list_repo_files()`, so those endpoints need network access even when the
  model is already cached.
- CUDA is optional; the default device is `cpu`.

## Install

```bash
uv sync
```

## Run

```bash
uv run uvicorn service:app --host 127.0.0.1 --port 8880
```

This is a plain ASGI application, so any ASGI server can host it. With the
default `KOKORO_WARMUP=1` the service synthesizes a short phrase at startup;
set `KOKORO_WARMUP=0` to skip that.

## Example

```bash
curl -sS -X POST http://127.0.0.1:8880/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input": "Hello from Kokoro.", "response_format": "wav"}' \
  --output hello.wav

file hello.wav
# hello.wav: RIFF (little-endian) data, WAVE audio, 16 bit, mono 24000 Hz
```

A request with only `input` uses the active profile for voice and speed.
Provider-native fields may be sent explicitly:

```bash
curl -sS -X POST http://127.0.0.1:8880/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input": "A faster voice.", "voice": "af_bella", "speed": 1.2}' \
  --output faster.wav
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Status, model, device and voice count |
| `GET` | `/v1/models` | OpenAI-style model list |
| `GET` | `/v1/audio/models` | Model ids served |
| `GET` | `/v1/audio/voices` | Available Kokoro voices |
| `GET` | `/v1/audio/voice-map` | OpenAI alias to Kokoro voice mapping |
| `GET` | `/v1/audio/defaults` | Active profile's voice and speed |
| `GET` | `/v1/audio/profiles` | All profiles and the active name |
| `PUT` | `/v1/audio/profiles/{name}` | Create or replace a profile |
| `POST` | `/v1/audio/profiles/{name}/activate` | Select the active profile |
| `POST` | `/v1/audio/speech` | Synthesize WAV audio |
| `GET` | `/v1/audio/capabilities` | Advertised transport capabilities |
| `POST` | `/cognitiveservices/v1` | Azure-compatible synthesis route |

## Profiles and voices

Profiles live in a JSON store at `KOKORO_PROFILE_STORE`, by default
`~/runs/kokoro-tts/profiles.json`. The service creates the file and its parent
directory on first use. It is provider-owned state: back it up if you care about
the profiles.

```bash
curl -sS http://127.0.0.1:8880/v1/audio/profiles
curl -sS -X PUT http://127.0.0.1:8880/v1/audio/profiles/narrator \
  -H 'content-type: application/json' \
  -d '{"voice": "af_heart", "speed": 1.0}'
curl -sS -X POST http://127.0.0.1:8880/v1/audio/profiles/narrator/activate
```

Profile and request voice values accept:

- a Kokoro voice id, such as `af_heart`;
- an OpenAI alias, such as `alloy` (see `/v1/audio/voice-map`);
- an equal blend, such as `af_heart,af_bella`;
- a weighted blend, such as `af_heart:70,af_bella:30`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `KOKORO_MODEL` | `hexgrad/Kokoro-82M` | Hugging Face model id or local path |
| `KOKORO_LANG` | `a` | Kokoro language code |
| `KOKORO_VOICE` | `af_heart` | Fallback voice |
| `KOKORO_SPEED` | `1.0` | Fallback speed |
| `KOKORO_DEVICE` | `cpu` | `cpu` or `cuda` |
| `KOKORO_WARMUP` | `1` | Synthesize a warmup phrase at startup |
| `KOKORO_PROFILE_STORE` | `~/runs/kokoro-tts/profiles.json` | Profile store path |

## Voice Lab

The service serves a static browser playground at `/lab/`, for example
`http://127.0.0.1:8880/lab/`. It defaults to a service at
`http://127.0.0.1:8880` and includes optional buttons that point a separate
`voice-agentd` instance, expected at `http://127.0.0.1:8893`, at the selected
profile. `voice-agentd` is not part of this repository; the synthesis and
profile controls work without it.

## Migration check

`scripts/check-profile-migration.py` exercises profile discovery, active-profile
resolution and profile activation against a temporary copy of the store. It
never writes the real store and needs no model download.

```bash
uv run python scripts/check-profile-migration.py
```

If `KOKORO_PROFILE_STORE` / `--store` points at an existing store, that store is
copied and verified byte-for-byte unchanged at the end. If it does not exist,
the check uses a temporary default store and prints a note. Expected output ends
with a line beginning `profile migration ok:`.

## Verify

On a fresh clone, after `uv sync`:

```bash
uv run python scripts/check-profile-migration.py
```

Then start the service and request one phrase:

```bash
curl -sS http://127.0.0.1:8880/health
curl -sS -X POST http://127.0.0.1:8880/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input": "verification", "response_format": "wav"}' \
  --output /tmp/kokoro-check.wav
file /tmp/kokoro-check.wav
```

The health response should report `"status": "ok"` and the output file should be
recognised as WAV audio.

## License

MIT for this service; see [LICENSE](LICENSE). The Kokoro-82M model weights and
the `kokoro` Python package are distributed separately under their own licences
(Apache-2.0 for the model).
