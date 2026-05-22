"""
Trouve, pour chaque (voix, langue), la voix EDGE TTS dont la duree (trimmee)
se rapproche le plus de notre voix, sur le MEME texte.

IMPORTANT : ce script NE regenere PAS nos fichiers audio Gemini/OpenAI.
  - Les durees de nos voix sont LUES depuis les fichiers locaux (gratuit).
  - Seuls des samples EDGE TTS sont generes (gratuit, service public Microsoft).

Pour chaque (voix, langue) declaree dans voices.json :
  1. our_duration = duree trimmee de notre fichier local (male|female/<lang>/<voice>.mp3)
  2. Pour chaque voix Edge de la langue : genere le MEME texte (texts.json),
     trim les silences, mesure la duree.
  3. Retient la voix Edge avec le plus petit |edge_dur - our_duration|.

Sorties :
  - gcp_matches.json          (par voix -> langue ; champ gcp_voice = voix Edge)
  - gcp_matches_ranked.json   ({ lang: { gemini: [...], openai: [...] } })
  - edge_duration_cache.json  (cache resume : (edge_voice|text_hash) -> duree)

Le champ reste nomme `gcp_voice` pour ne pas casser le frontend qui le lit ;
sa valeur est desormais une voix Edge TTS.

Variables d'environnement :
  WORKERS       (defaut 10)
  MAX_RETRIES   (defaut 5)

Flags CLI :
  --voice <name>   Filtre sur une voix
  --lang <code>    Filtre sur une langue
  --limit N        Limite le nombre de (voix,langue) traitees (tests)
"""

import os
import sys
import ssl
import json
import io
import time
import asyncio
import hashlib
import warnings
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa
from pydub import AudioSegment

import edge_tts
import edge_tts.voices as _ev
import edge_tts.communicate as _ec

from trim_silences import detect_silences, _apply_trim

warnings.filterwarnings("ignore")

# --- Fix SSL : utiliser le cert store Windows (inclut la CA du proxy local) ---
_SSL_CTX = ssl.create_default_context()
_ev._SSL_CTX = _SSL_CTX
_ec._SSL_CTX = _SSL_CTX

WORKERS = int(os.environ.get("WORKERS", "10"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
MAX_SILENCE = 0.05
CACHE_PATH = "edge_duration_cache.json"

# Nos codes langue -> prefixe Edge (Edge utilise zh pour le mandarin)
LANG_ALIAS = {"cmn": "zh"}

_cache_lock = threading.Lock()
_cache = {}
_cache_dirty = 0


def _text_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_cache(root):
    global _cache
    p = root / CACHE_PATH
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    print(f"Cache charge: {len(_cache)} entrees", flush=True)


def save_cache(root):
    p = root / CACHE_PATH
    tmp = p.with_suffix(p.suffix + ".tmp")
    with _cache_lock:
        snapshot = dict(_cache)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False)
    os.replace(tmp, p)


def get_edge_voices_by_prefix():
    """Recupere toutes les voix Edge groupees par prefixe de langue."""
    async def _list():
        return await edge_tts.list_voices()
    voices = asyncio.run(_list())
    by_prefix = {}
    for v in voices:
        prefix = v["Locale"].split("-")[0]
        by_prefix.setdefault(prefix, []).append(v["ShortName"])
    return by_prefix


def get_trimmed_duration_local(file_path):
    """Duree trimmee d'un fichier MP3 local (lecture seule, pas d'API)."""
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    _, silences = detect_silences(audio=audio, sr=sr)
    long = [s for s in silences if (s[2] - s[1]) > MAX_SILENCE]
    if long:
        audio = _apply_trim(audio, sr, silences, MAX_SILENCE)
    return len(audio) / sr


def edge_synth_bytes(text, edge_voice):
    """Genere un sample Edge TTS et retourne les bytes MP3 (run un loop asyncio dedie)."""
    async def _run():
        communicate = edge_tts.Communicate(text, edge_voice)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)
    return asyncio.run(_run())


def edge_trimmed_duration(text, edge_voice):
    """Duree trimmee d'un sample Edge, avec cache + retry."""
    key = f"{edge_voice}|{_text_hash(text)}"
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    backoff = 1.5
    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            mp3 = edge_synth_bytes(text, edge_voice)
            if not mp3:
                raise RuntimeError("empty audio")
            seg = AudioSegment.from_mp3(io.BytesIO(mp3))
            wav = io.BytesIO()
            seg.export(wav, format="wav")
            wav.seek(0)
            audio, sr = librosa.load(wav, sr=None, mono=True)
            _, silences = detect_silences(audio=audio, sr=sr)
            long = [s for s in silences if (s[2] - s[1]) > MAX_SILENCE]
            if long:
                audio = _apply_trim(audio, sr, silences, MAX_SILENCE)
            dur = len(audio) / sr

            global _cache_dirty
            with _cache_lock:
                _cache[key] = dur
                _cache_dirty += 1
            return dur
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 20)
    raise RuntimeError(f"edge synth failed for {edge_voice}: {last_err}")


def process_voice_lang(our_dur, text, edge_candidates):
    """Retourne le meilleur match Edge (duree la plus proche)."""
    best_voice = None
    best_diff = float("inf")
    best_edge_dur = 0.0

    for edge_voice in edge_candidates:
        try:
            edge_dur = edge_trimmed_duration(text, edge_voice)
        except Exception:
            continue
        diff = abs(edge_dur - our_dur)
        if diff < best_diff:
            best_diff = diff
            best_voice = edge_voice
            best_edge_dur = edge_dur

    if not best_voice:
        return None

    ratio = best_edge_dur / our_dur if our_dur > 0 else None
    return {
        "gcp_voice": best_voice,  # nom conserve pour compat frontend ; valeur = voix Edge
        "duration_diff": round(best_diff, 3),
        "our_duration": round(our_dur, 3),
        "edge_duration": round(best_edge_dur, 3),
        "ratio": round(ratio, 3) if ratio is not None else None,
        "duree_sans_silence": round(our_dur, 3),
    }


def parse_args(argv):
    voice_filter = lang_filter = None
    limit = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--voice" and i + 1 < len(argv):
            voice_filter = argv[i + 1]; i += 1
        elif a == "--lang" and i + 1 < len(argv):
            lang_filter = argv[i + 1]; i += 1
        elif a == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1]); i += 1
        else:
            print(f"Argument inconnu: {a}", file=sys.stderr); sys.exit(2)
        i += 1
    return voice_filter, lang_filter, limit


def main():
    root = Path(__file__).parent
    voice_filter, lang_filter, limit = parse_args(sys.argv)

    with open(root / "voices.json", encoding="utf-8") as f:
        voices_data = json.load(f)
    with open(root / "texts.json", encoding="utf-8") as f:
        texts_data = json.load(f)

    voices = voices_data["voices"]

    print("Recuperation des voix Edge...", flush=True)
    edge_by_prefix = get_edge_voices_by_prefix()
    total_edge = sum(len(v) for v in edge_by_prefix.values())
    print(f"Voix Edge: {total_edge} dans {len(edge_by_prefix)} prefixes", flush=True)

    load_cache(root)

    # Precalcul des durees de NOS fichiers (lecture locale, gratuit)
    print("Calcul des durees de nos fichiers (local, sans API)...", flush=True)
    our_durations = {}
    for vname, vdata in voices.items():
        if voice_filter and vname.lower() != voice_filter.lower():
            continue
        for lang, fpath in vdata["files"].items():
            if lang_filter and lang != lang_filter:
                continue
            full = root / fpath
            if full.exists():
                our_durations[(vname, lang)] = get_trimmed_duration_local(str(full))
    print(f"Nos fichiers mesures: {len(our_durations)}", flush=True)

    # Construction des taches
    tasks = []
    for vname, vdata in voices.items():
        if voice_filter and vname.lower() != voice_filter.lower():
            continue
        # texte (lookup case-insensitive)
        text_key = vname if vname in texts_data else None
        if text_key is None:
            for k in texts_data:
                if k.lower() == vname.lower():
                    text_key = k
                    break
        for lang, fpath in vdata["files"].items():
            if lang_filter and lang != lang_filter:
                continue
            if (vname, lang) not in our_durations:
                continue
            text = (texts_data.get(text_key, {}) or {}).get(lang)
            if not text:
                continue
            edge_prefix = LANG_ALIAS.get(lang, lang)
            edge_candidates = edge_by_prefix.get(edge_prefix, [])
            if not edge_candidates:
                continue
            tasks.append((vname, lang, text, our_durations[(vname, lang)], edge_candidates))

    if limit:
        tasks = tasks[:limit]

    total_calls = sum(len(t[4]) for t in tasks)
    print(f"Taches (voix,langue): {len(tasks)}", flush=True)
    print(f"Appels Edge max (hors cache): {total_calls}", flush=True)
    print(f"Workers: {WORKERS}", flush=True)
    print("-" * 60, flush=True)

    results = {}
    done = 0

    def run_task(args):
        vname, lang, text, our_dur, cands = args
        r = process_voice_lang(our_dur, text, cands)
        return vname, lang, r

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            vname, lang, r = fut.result()
            done += 1
            results.setdefault(vname, {})[lang] = r
            if r and r["gcp_voice"]:
                print(f"[{done}/{len(tasks)}] {vname}/{lang}: {r['gcp_voice']} "
                      f"(diff={r['duration_diff']}s, ratio={r['ratio']})", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {vname}/{lang}: no match", flush=True)

            # Sauvegarde periodique du cache
            global _cache_dirty
            if _cache_dirty >= 100:
                save_cache(root)
                with _cache_lock:
                    _cache_dirty = 0

    save_cache(root)

    # --- edge_matches.json (detail par voix -> langue) ---
    # NB: on n'ecrase PAS gcp_matches.json (conserve la version GCP d'origine).
    out_path = root / "edge_matches.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- gcp_matches_ranked.json ({ lang: { gemini:[...], openai:[...] } }) ---
    voice_provider = {vn: vd.get("provider", "gemini") for vn, vd in voices.items()}
    ranked = {}
    for vname, by_lang in results.items():
        provider = voice_provider.get(vname, "gemini")
        for lang, r in by_lang.items():
            if not r or not r.get("gcp_voice"):
                continue
            entry = {
                "voice": vname,
                "gcp_voice": r["gcp_voice"],
                "duration_diff": r["duration_diff"],
                "ratio": r["ratio"],
                "duree_sans_silence": r["duree_sans_silence"],
            }
            ranked.setdefault(lang, {}).setdefault(provider, []).append(entry)

    for lang, by_prov in ranked.items():
        for prov in by_prov:
            by_prov[prov].sort(key=lambda x: x["duration_diff"])

    provider_order = ["gemini", "openai"]
    ranked_sorted = {}
    for lang in sorted(ranked.keys()):
        ranked_sorted[lang] = {
            p: ranked[lang][p] for p in provider_order if p in ranked[lang]
        }

    ranked_path = root / "edge_matches_ranked.json"
    with open(ranked_path, "w", encoding="utf-8") as f:
        json.dump(ranked_sorted, f, indent=2, ensure_ascii=False)

    print("-" * 60)
    print("Resultats sauvegardes :")
    print(f"  - {out_path}")
    print(f"  - {ranked_path}")
    print(f"  - cache: {root / CACHE_PATH} ({len(_cache)} entrees)")


if __name__ == "__main__":
    main()
