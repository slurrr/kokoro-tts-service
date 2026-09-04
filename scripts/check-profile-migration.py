#!/usr/bin/env python3
"""Verify that the provider-owned profile store survives service migration.

The real store is only read. A temporary copy is used to exercise the service
profile APIs and default resolution, so this check cannot rewrite the user's
profile data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_STORE = Path.home() / "runs" / "kokoro-tts" / "profiles.json"


def read_store(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        raise TypeError(f"invalid profile store: {path}")
    return raw, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args()

    source_raw, source = read_store(args.store)
    source_profiles = source["profiles"]
    source_active = str(source.get("active") or "default")
    if not source_profiles or source_active not in source_profiles:
        raise AssertionError("profile store has no valid active profile")

    with tempfile.TemporaryDirectory(prefix="kokoro-profile-migration-") as tmp:
        temp_store = Path(tmp) / "profiles.json"
        temp_store.write_bytes(source_raw)
        os.environ["KOKORO_PROFILE_STORE"] = str(temp_store)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import service

        assert service.PROFILE_STORE_PATH == temp_store
        route_paths = {route.path for route in service.app.routes}
        assert {
            "/health",
            "/v1/models",
            "/v1/audio/models",
            "/v1/audio/voices",
            "/v1/audio/voice-map",
            "/v1/audio/defaults",
            "/v1/audio/profiles",
            "/v1/audio/speech",
        } <= route_paths
        assert service.models()["data"][0]["id"] == "kokoro"
        assert service.audio_models()["data"] == ["kokoro"]
        assert service.audio_profiles() == {"active": source_active, "profiles": source_profiles}
        defaults = service.audio_defaults()
        active_profile = source_profiles[source_active]
        assert defaults["active_profile"] == source_active
        assert defaults["voice"] == active_profile["voice"]
        assert defaults["speed"] == active_profile["speed"]
        # Read-only discovery/default calls must not rewrite even formatting.
        assert temp_store.read_bytes() == source_raw

        generated: list[tuple[str, float]] = []

        def fake_synthesize(text: str, voice: str, speed: float) -> bytes:
            assert text == "migration check"
            generated.append((voice, speed))
            return b"RIFF migration-check"

        service.synthesize = fake_synthesize
        service.speech(service.SpeechRequest(input="migration check", response_format="wav"))
        assert generated[-1] == (active_profile["voice"], float(active_profile["speed"]))

        old_voices = service.get_available_voices
        service.get_available_voices = lambda: ["af_bella", "af_heart", "af_nicole"]
        try:
            assert service._resolve_voice_spec("alloy") == "af_heart"
            assert service._resolve_voice_spec("af_heart,af_bella") == "af_heart,af_bella"
            weighted = service._resolve_voice_spec("af_heart:70,af_bella:30")
            assert weighted.count("af_heart") == 70
            assert weighted.count("af_bella") == 30
        finally:
            service.get_available_voices = old_voices

        other_names = [name for name in source_profiles if name != source_active]
        if other_names:
            other = other_names[0]
            service.activate_audio_profile(other)
            after_raw, after = read_store(temp_store)
            assert after["active"] == other
            # Activation changes only the selected profile pointer; all
            # existing profile records remain byte-for-byte equivalent as data.
            assert after["profiles"] == source_profiles
            other_profile = source_profiles[other]
            service.speech(service.SpeechRequest(input="migration check", response_format="wav"))
            assert generated[-1] == (other_profile["voice"], float(other_profile["speed"]))
            assert after_raw != source_raw  # active selection was intentionally changed

    assert args.store.read_bytes() == source_raw
    print(
        "profile migration ok: "
        f"active={source_active} profiles={','.join(source_profiles)} "
        "source_unchanged=yes"
    )


if __name__ == "__main__":
    main()
