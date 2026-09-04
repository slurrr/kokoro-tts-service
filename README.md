# Kokoro TTS service

Provider-owned Kokoro HTTP service. The shared voice loop sends only the
completed-audio transport fields (`input` and `response_format: "wav"`); this
service applies the active profile when no provider-native voice, speed, or
profile is supplied.

## Run

The existing launcher controls the runtime environment and keeps the default
CPU/GPU behavior:

```bash
kokoro-tts-serve --bg       # add --gpu for CUDA
```

For a direct foreground run:

```bash
uv run --project ~/code/dev/kokoro-tts-service \
  uvicorn service:app --host 127.0.0.1 --port 8880
```

Important environment variables remain provider-local:

- `KOKORO_MODEL`
- `KOKORO_LANG`
- `KOKORO_VOICE`
- `KOKORO_SPEED`
- `KOKORO_DEVICE`
- `KOKORO_WARMUP` (`1` by default)
- `KOKORO_PROFILE_STORE` (defaults to `~/runs/kokoro-tts/profiles.json`)

## Provider management

The profile store is not read or written by `voice-agentd`, `the-stack`, or
Agent TUI. Use the provider API or the Voice Lab at `/lab/`:

```bash
curl http://127.0.0.1:8880/v1/audio/profiles
curl http://127.0.0.1:8880/v1/audio/defaults
curl -X PUT http://127.0.0.1:8880/v1/audio/profiles/<name> \
  -H 'content-type: application/json' \
  -d '{"voice":"af_heart","speed":1.0}'
curl -X POST http://127.0.0.1:8880/v1/audio/profiles/<name>/activate
```

`~/runs/kokoro-tts/profiles.json` is an existing provider data store and is
preserved in place. Profile voice values support Kokoro voice IDs, OpenAI
aliases, equal blends (`voice_a,voice_b`), and weighted blends
(`voice_a:70,voice_b:30`). The service owns validation and translation of
these values.

## Compatibility and transport APIs

The service retains the `/health`, `/v1/models`, `/v1/audio/models`,
`/v1/audio/voices`, `/v1/audio/voice-map`, `/v1/audio/defaults`,
`/v1/audio/profiles`, and `/v1/audio/speech` APIs, plus the Azure-compatible
route. `GET /v1/audio/capabilities` advertises completed WAV transport and no
native stream.

A request with only `input` and `response_format: "wav"` uses the active
profile. Provider-native requests may supply `profile`, `voice`, or `speed`
for explicit provider management/testing. The loop and generic clients never
send those fields.

## Migration check

Run this before and after service changes. It copies the real store to a
temporary path, exercises profile discovery, active-profile synthesis
resolution, and profile activation, then verifies that the real store is
byte-for-byte unchanged and all profile records survive:

```bash
uv run --project ~/code/dev/kokoro-tts-service \
  python scripts/check-profile-migration.py
```
