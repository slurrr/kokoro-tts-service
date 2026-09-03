from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from huggingface_hub import list_repo_files
from kokoro import KPipeline


DEFAULT_MODEL = os.getenv("KOKORO_MODEL", "hexgrad/Kokoro-82M")
DEFAULT_LANG = os.getenv("KOKORO_LANG", "a")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
DEFAULT_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))
DEFAULT_DEVICE = os.getenv("KOKORO_DEVICE", "cpu")

# Open WebUI often tries OpenAI defaults like "alloy".
VOICE_ALIASES = {
    # OpenAI-label -> actual Kokoro voice ID (user-selected female voices)
    "alloy": "af_heart",
    "echo": "af_bella",
    "fable": "af_nicole",
    "onyx": "af_aoede",
    "nova": "af_kore",
    "shimmer": "af_sarah",
}


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str | None = None
    response_format: Literal["wav", "mp3"] = "wav"
    speed: float | None = None
    profile: str | None = None


app = FastAPI(title="Kokoro TTS API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: KPipeline | None = None
_cached_voices: list[str] | None = None
DEFAULT_WARMUP = os.getenv("KOKORO_WARMUP", "1").strip().lower() not in {"0", "false", "no", "off"}

PROFILE_STORE_PATH = Path(os.getenv("KOKORO_PROFILE_STORE", str(Path.home() / "runs" / "kokoro-tts" / "profiles.json")))


def _default_profile_store() -> dict[str, Any]:
    return {
        "active": "default",
        "profiles": {
            "default": {
                "voice": DEFAULT_VOICE,
                "speed": DEFAULT_SPEED,
            }
        },
    }


def _load_profile_store() -> dict[str, Any]:
    try:
        if PROFILE_STORE_PATH.exists():
            data = json.loads(PROFILE_STORE_PATH.read_text())
            if isinstance(data, dict) and isinstance(data.get("profiles"), dict):
                return data
    except Exception:
        pass
    data = _default_profile_store()
    _save_profile_store(data)
    return data


def _save_profile_store(data: dict[str, Any]) -> None:
    PROFILE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_STORE_PATH.write_text(json.dumps(data, indent=2))


def _resolve_profile_values(profile_name: str | None) -> tuple[str, float, str]:
    store = _load_profile_store()
    active = str(store.get("active") or "default")
    use = str(profile_name or active or "default")
    profiles = store.get("profiles") if isinstance(store.get("profiles"), dict) else {}
    selected = profiles.get(use) if isinstance(profiles, dict) else None
    if not isinstance(selected, dict):
        selected = profiles.get(active) if isinstance(profiles, dict) else None
    if not isinstance(selected, dict):
        selected = {"voice": DEFAULT_VOICE, "speed": DEFAULT_SPEED}
    voice = str(selected.get("voice") or DEFAULT_VOICE)
    try:
        speed = float(selected.get("speed") if selected.get("speed") is not None else DEFAULT_SPEED)
    except Exception:
        speed = DEFAULT_SPEED
    return voice, speed, use


def get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code=DEFAULT_LANG, repo_id=DEFAULT_MODEL, device=DEFAULT_DEVICE)
    return _pipeline


def _warmup_pipeline() -> None:
    if not DEFAULT_WARMUP:
        return
    synthesize("warmup", DEFAULT_VOICE, DEFAULT_SPEED)


def get_available_voices() -> list[str]:
    global _cached_voices
    if _cached_voices is None:
        files = list_repo_files(DEFAULT_MODEL)
        _cached_voices = sorted(
            f.split("/")[-1].replace(".pt", "")
            for f in files
            if f.startswith("voices/") and f.endswith(".pt")
        )
    return _cached_voices


def _resolve_voice_spec(voice: str) -> str:
    """Resolve aliases + optional weighted mix syntax into pipeline-compatible voice string.

    Supports:
      - "af_heart"
      - "af_heart,af_bella" (equal mix; native Kokoro behavior)
      - "af_heart:70,af_bella:30" (weighted mix; approximated via repetition)
    """
    raw = (voice or "").strip()
    if not raw:
        return DEFAULT_VOICE

    known_voices = set(get_available_voices())

    # Weighted syntax: v1:w1,v2:w2
    if ":" in raw and "," in raw:
        parts: list[tuple[str, float]] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                continue
            v, w = item.split(":", 1)
            v = VOICE_ALIASES.get(v.strip(), v.strip())
            try:
                weight = float(w.strip())
            except ValueError:
                continue
            if weight <= 0:
                continue
            if v in known_voices:
                parts.append((v, weight))

        if parts:
            total = sum(w for _v, w in parts)
            if total <= 0:
                return DEFAULT_VOICE
            expanded: list[str] = []
            # Quantize to ~100 slots for stable, low-overhead approximation.
            for v, w in parts:
                slots = max(1, int(round((w / total) * 100)))
                expanded.extend([v] * slots)
            return ",".join(expanded)

    # Equal-mix syntax: v1,v2,v3
    if "," in raw:
        valid = []
        for item in raw.split(","):
            v = VOICE_ALIASES.get(item.strip(), item.strip())
            if v in known_voices:
                valid.append(v)
        return ",".join(valid) if valid else DEFAULT_VOICE

    # Single voice
    one = VOICE_ALIASES.get(raw, raw)
    return one if one in known_voices else DEFAULT_VOICE


def synthesize(text: str, voice: str, speed: float) -> bytes:
    if not text.strip():
        raise HTTPException(status_code=400, detail="input must not be empty")

    pipeline = get_pipeline()
    selected_voice = _resolve_voice_spec(voice)

    try:
        stream = pipeline(text, voice=selected_voice, speed=speed)
        chunks: list[np.ndarray] = []
        sample_rate = 24000

        for _g, _p, audio in stream:
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            chunks.append(arr)

        if not chunks:
            raise RuntimeError("no audio generated")

        audio_all = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, audio_all, samplerate=sample_rate, format="WAV")
        return buf.getvalue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"kokoro generation failed (voice={selected_voice}): {e}") from e


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "lang": DEFAULT_LANG,
        "default_voice": DEFAULT_VOICE,
        "device": DEFAULT_DEVICE,
        "voice_count": len(get_available_voices()),
        "loaded": _pipeline is not None,
    }


@app.on_event("startup")
def startup() -> None:
    _warmup_pipeline()


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": "kokoro", "object": "model", "owned_by": "local"}]}


@app.get("/v1/audio/capabilities")
def audio_capabilities() -> dict[str, Any]:
    return {
        "provider": "kokoro",
        "name": "Kokoro",
        "protocol_version": 1,
        "default_voice": DEFAULT_VOICE,
        "speech_endpoint": "/v1/audio/speech",
        "native_stream": {
            "available": False,
            "endpoint": "",
            "cancel_endpoint": "",
            "protocol": "",
        },
    }


@app.get("/v1/audio/voice-map")
def voice_map() -> dict[str, Any]:
    return {
        "openai_to_kokoro": VOICE_ALIASES,
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/v1/audio/models")
def audio_models() -> dict[str, Any]:
    return {"data": ["kokoro"]}


@app.get("/v1/audio/defaults")
def audio_defaults() -> dict[str, Any]:
    store = _load_profile_store()
    active = str(store.get("active") or "default")
    profiles = store.get("profiles") if isinstance(store.get("profiles"), dict) else {}
    current = profiles.get(active, {}) if isinstance(profiles, dict) else {}
    return {
        "active_profile": active,
        "voice": current.get("voice", DEFAULT_VOICE),
        "speed": current.get("speed", DEFAULT_SPEED),
    }


@app.get("/v1/audio/profiles")
def audio_profiles() -> dict[str, Any]:
    store = _load_profile_store()
    return {
        "active": store.get("active", "default"),
        "profiles": store.get("profiles", {}),
    }


@app.put("/v1/audio/profiles/{name}")
async def upsert_audio_profile(name: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="profile payload must be object")
    voice = str(payload.get("voice") or "").strip()
    if not voice:
        raise HTTPException(status_code=400, detail="voice is required")
    try:
        speed = float(payload.get("speed") if payload.get("speed") is not None else DEFAULT_SPEED)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid speed: {exc}") from exc

    store = _load_profile_store()
    profiles = store.get("profiles") if isinstance(store.get("profiles"), dict) else {}
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[name] = {"voice": voice, "speed": speed}
    store["profiles"] = profiles
    _save_profile_store(store)
    return {"ok": True, "name": name, "profile": profiles[name]}


@app.post("/v1/audio/profiles/{name}/activate")
def activate_audio_profile(name: str) -> dict[str, Any]:
    store = _load_profile_store()
    profiles = store.get("profiles") if isinstance(store.get("profiles"), dict) else {}
    if name not in profiles:
        raise HTTPException(status_code=404, detail=f"profile not found: {name}")
    store["active"] = name
    _save_profile_store(store)
    return {"ok": True, "active": name}


@app.get("/v1/audio/voices")
def audio_voices() -> Any:
    voices = get_available_voices()
    # Return multiple schema variants for UI compatibility.
    return {
        "voices": voices,
        "data": [{"id": v, "name": v, "object": "voice"} for v in voices],
        "items": voices,
    }


@app.get("/cognitiveservices/voices/list")
def azure_voices_list() -> Any:
    voices = get_available_voices()
    return [
        {
            "Name": v,
            "ShortName": v,
            "DisplayName": v,
            "Locale": "en-US",
            "Gender": "Female" if v.startswith(("af_", "bf_", "ef_", "ff_", "hf_", "if_", "jf_", "pf_", "zf_")) else "Male",
            "VoiceType": "Neural",
        }
        for v in voices
    ]


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    p_voice, p_speed, _name = _resolve_profile_values(req.profile)
    voice = (req.voice or p_voice or DEFAULT_VOICE)
    speed = float(req.speed if req.speed is not None else p_speed)
    wav_bytes = synthesize(req.input, voice, speed)
    return Response(content=wav_bytes, media_type="audio/wav")


# Compatibility route for clients trying Azure-style pathing.
@app.post("/cognitiveservices/v1")
async def azure_tts_compat(request: Request):
    ctype = request.headers.get("content-type", "")
    body: dict[str, Any] = {}
    if "application/json" in ctype:
        raw = await request.json()
        if isinstance(raw, dict):
            body = raw

    profile = request.query_params.get("profile") or body.get("profile")
    p_voice, p_speed, _name = _resolve_profile_values(str(profile) if profile else None)

    if "application/json" in ctype:
        text = str(body.get("input") or body.get("text") or "")
        voice = str(body.get("voice") or p_voice or DEFAULT_VOICE)
        try:
            speed = float(body.get("speed") if body.get("speed") is not None else p_speed)
        except Exception:
            speed = p_speed
    else:
        # Minimal SSML/plain-text fallback.
        text = (await request.body()).decode("utf-8", errors="ignore")
        voice = p_voice
        speed = p_speed

    wav_bytes = synthesize(text, voice, float(speed))
    return Response(content=wav_bytes, media_type="audio/wav")


# Better error visibility in logs and client responses.
@app.exception_handler(Exception)
async def unhandled_exception(_request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": f"unhandled server error: {exc}"})
