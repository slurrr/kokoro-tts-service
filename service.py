from __future__ import annotations

import io
import os
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
    voice: str = DEFAULT_VOICE
    response_format: Literal["wav", "mp3"] = "wav"
    speed: float = DEFAULT_SPEED


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


def get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code=DEFAULT_LANG, repo_id=DEFAULT_MODEL, device=DEFAULT_DEVICE)
    return _pipeline


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


def synthesize(text: str, voice: str, speed: float) -> bytes:
    if not text.strip():
        raise HTTPException(status_code=400, detail="input must not be empty")

    pipeline = get_pipeline()
    selected_voice = VOICE_ALIASES.get(voice, voice)
    known_voices = set(get_available_voices())
    if selected_voice not in known_voices:
        # Don't fail hard for unsupported voices from OpenAI defaults.
        selected_voice = DEFAULT_VOICE

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
    }


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
    wav_bytes = synthesize(req.input, req.voice, req.speed)
    return Response(content=wav_bytes, media_type="audio/wav")


# Compatibility route for clients trying Azure-style pathing.
@app.post("/cognitiveservices/v1")
async def azure_tts_compat(request: Request):
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        body = await request.json()
        text = str(body.get("input") or body.get("text") or "")
        voice = str(body.get("voice") or DEFAULT_VOICE)
        speed = float(body.get("speed") or DEFAULT_SPEED)
    else:
        # Minimal SSML/plain-text fallback.
        text = (await request.body()).decode("utf-8", errors="ignore")
        voice = DEFAULT_VOICE
        speed = DEFAULT_SPEED

    wav_bytes = synthesize(text, voice, speed)
    return Response(content=wav_bytes, media_type="audio/wav")


# Better error visibility in logs and client responses.
@app.exception_handler(Exception)
async def unhandled_exception(_request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": f"unhandled server error: {exc}"})
