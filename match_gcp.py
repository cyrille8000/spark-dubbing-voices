import requests
import base64
import json
import numpy as np
import librosa
import soundfile as sf
import io
from pydub import AudioSegment
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from trim_silences import detect_silences, _apply_trim
import os
import warnings
import sys

warnings.filterwarnings('ignore')

API_KEY = os.environ.get("GCP_TTS_API_KEY", "")
TTS_URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={API_KEY}"
VOICES_URL = f"https://texttospeech.googleapis.com/v1/voices?key={API_KEY}"

# Mapping nos codes langue -> codes GCP
LANG_MAP = {
    "en": "en-US", "fr": "fr-FR", "es": "es-ES", "de": "de-DE",
    "it": "it-IT", "pt": "pt-BR", "ja": "ja-JP", "ko": "ko-KR",
    "cmn": "cmn-CN", "ar": "ar-XA", "hi": "hi-IN", "ru": "ru-RU",
    "tr": "tr-TR", "nl": "nl-NL", "pl": "pl-PL", "id": "id-ID",
    "uk": "uk-UA", "vi": "vi-VN", "th": "th-TH", "ro": "ro-RO",
    "el": "el-GR", "cs": "cs-CZ", "fi": "fi-FI", "bg": "bg-BG",
    "da": "da-DK", "he": "he-IL", "ms": "ms-MY", "fa": None,
    "sk": "sk-SK", "sv": "sv-SE", "hr": None, "fil": "fil-PH",
    "hu": "hu-HU", "nb": "nb-NO", "sl": None, "ca": "ca-ES",
    "nn": None, "ta": "ta-IN", "af": "af-ZA"
}


def get_gcp_standard_voices():
    """Recupere toutes les voix Standard GCP groupees par prefixe langue (2-3 premieres lettres)."""
    r = requests.get(VOICES_URL)
    r.raise_for_status()
    voices = r.json()["voices"]
    standard = [v for v in voices if "Standard" in v["name"]]

    by_prefix = {}
    for v in standard:
        for lc in v["languageCodes"]:
            # Prefixe = partie avant le premier tiret (fr, en, cmn, fil, etc.)
            prefix = lc.split("-")[0]
            if prefix not in by_prefix:
                by_prefix[prefix] = []
            # Stocker le nom + le languageCode pour l'appel API
            by_prefix[prefix].append({"name": v["name"], "languageCode": lc})
    return by_prefix


def generate_tts(text, voice_name, lang_code):
    """Genere un sample TTS et retourne l'audio en bytes MP3."""
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": lang_code, "name": voice_name},
        "audioConfig": {"audioEncoding": "MP3", "sampleRateHertz": 24000}
    }
    r = requests.post(TTS_URL, json=payload)
    if r.status_code != 200:
        return None
    return base64.b64decode(r.json()["audioContent"])


def get_trimmed_duration(mp3_bytes):
    """Calcule la duree trimmee d'un fichier audio MP3 en bytes."""
    # Sauver temporairement pour librosa
    audio_seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    wav_buf = io.BytesIO()
    audio_seg.export(wav_buf, format="wav")
    wav_buf.seek(0)

    audio, sr = librosa.load(wav_buf, sr=None, mono=True)
    duration, silences = detect_silences(audio=audio, sr=sr)

    long = [s for s in silences if (s[2] - s[1]) > 0.05]
    if long:
        audio = _apply_trim(audio, sr, silences, 0.05)

    return len(audio) / sr


def get_file_duration(file_path):
    """Duree d'un fichier MP3 local."""
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    return len(audio) / sr


def process_voice_lang(voice_name, lang, text, gcp_voices, our_duration):
    """Pour une voix+langue, genere tous les samples GCP (toutes variantes) et retourne le meilleur match."""
    # Prefix langue : fr, en, cmn, fil, etc.
    prefix = lang if lang in gcp_voices else None
    if not prefix:
        return None

    candidates = gcp_voices[prefix]
    best_match = None
    best_diff = float("inf")
    best_gcp_dur = 0

    for candidate in candidates:
        gcp_voice = candidate["name"]
        lang_code = candidate["languageCode"]
        mp3_bytes = generate_tts(text, gcp_voice, lang_code)
        if mp3_bytes is None:
            continue

        try:
            trimmed_dur = get_trimmed_duration(mp3_bytes)
            diff = abs(trimmed_dur - our_duration)
            if diff < best_diff:
                best_diff = diff
                best_match = gcp_voice
                best_gcp_dur = trimmed_dur
        except Exception:
            continue

    return {
        "gcp_voice": best_match,
        "duration_diff": round(best_diff, 3) if best_match else None,
        "our_duration": round(our_duration, 3),
        "gcp_duration": round(best_gcp_dur, 3) if best_match else None,
    }


def main():
    root = Path(__file__).parent

    with open(root / "voices.json", encoding="utf-8") as f:
        voices_data = json.load(f)
    with open(root / "texts.json", encoding="utf-8") as f:
        texts_data = json.load(f)

    voices = voices_data["voices"]

    print("Recuperation des voix GCP Standard...", flush=True)
    gcp_voices = get_gcp_standard_voices()
    total_gcp = sum(len(v) for v in gcp_voices.values())
    print(f"Voix GCP Standard: {total_gcp} dans {len(gcp_voices)} langues", flush=True)

    # Precalculer les durees de nos fichiers
    print("Calcul des durees de nos fichiers...", flush=True)
    our_durations = {}
    for vname, vdata in voices.items():
        for lang, fpath in vdata["files"].items():
            full_path = root / fpath
            if full_path.exists():
                our_durations[(vname, lang)] = get_file_duration(str(full_path))

    print(f"Nos fichiers: {len(our_durations)}", flush=True)

    # Preparer les taches
    tasks = []
    for vname, vdata in voices.items():
        # Recuperer le texte depuis texts.json (le nom est capitalise)
        text_key = vname.capitalize() if vname.capitalize() in texts_data else vname
        if text_key not in texts_data:
            # Chercher case-insensitive
            for k in texts_data:
                if k.lower() == vname.lower():
                    text_key = k
                    break

        for lang, fpath in vdata["files"].items():
            if (vname, lang) not in our_durations:
                continue

            text = texts_data.get(text_key, {}).get(lang)
            if not text:
                continue

            tasks.append((vname, lang, text, our_durations[(vname, lang)]))

    print(f"Taches a traiter: {len(tasks)}", flush=True)
    print("-" * 60, flush=True)

    results = {}
    done = 0

    def run_task(args):
        vname, lang, text, our_dur = args
        result = process_voice_lang(vname, lang, text, gcp_voices, our_dur)
        return vname, lang, result

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(run_task, t): t for t in tasks}
        for future in as_completed(futures):
            vname, lang, result = future.result()
            done += 1

            if vname not in results:
                results[vname] = {}
            results[vname][lang] = result

            if result and result["gcp_voice"]:
                print(f"[{done}/{len(tasks)}] {vname}/{lang}: {result['gcp_voice']} (diff={result['duration_diff']}s)", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {vname}/{lang}: no match", flush=True)

    # Sauvegarder
    output_path = root / "gcp_matches.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"Resultats sauvegardes dans {output_path}")


if __name__ == "__main__":
    main()
