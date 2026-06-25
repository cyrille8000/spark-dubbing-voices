"""
Regenere tous les samples audio Gemini TTS (MP3 bruts, sans trim).

Pour chaque (voice, lang) declare dans voices.json :
  1. Recupere le texte depuis texts.json
  2. Appelle l'API Gemini (model GEMINI_TTS_MODEL, defaut gemini-3.1-flash-tts-preview)
  3. Decode le PCM 16-bit 24kHz mono retourne par l'API
  4. Encode en MP3 192k et ecrit a la destination indiquee dans voices.json
     (ex: female/en/zephyr.mp3) en ECRASANT le fichier existant.

Le trim des silences est une etape separee : lancer
`python trim_silences.py` ensuite pour produire les fichiers trimmes.

Authentification (deux modes, OAuth prioritaire) :
  - OAuth via service account : GEMINI_SA_KEY (JSON inline) + GCP_AI_STUDIO_PROJECT
  - API key : GEMINI_API_KEY

Autres variables d'environnement :
  GEMINI_TTS_MODEL      (defaut gemini-3.1-flash-preview-tts)
  WORKERS               (defaut 8)
  MAX_RETRIES           (defaut 6)

Flags CLI :
  --skip-existing       N'ecrase pas les fichiers deja presents (idempotent)
  --limit N             Ne traite que N taches (utile pour tests)
  --voice <Name>        Filtre sur une voix (ex: --voice Puck)
  --lang <code>         Filtre sur une langue (ex: --lang en)
"""

import os
import sys
import json
import time
import base64
import threading
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import ssl
import requests
import certifi
from pydub import AudioSegment
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

warnings.filterwarnings("ignore")


# TLS : magasin de certs de l'OS en plus de certifi — derriere un proxy/AV qui
# intercepte le HTTPS, le CA du proxy est dans le store Windows mais pas dans
# certifi ("unable to get local issuer certificate").
class _OSStoreAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **kw):
        ctx = ssl.create_default_context(cafile=certifi.where())
        try:
            ctx.load_default_certs()
        except Exception:
            pass
        self.poolmanager = PoolManager(
            num_pools=connections, maxsize=maxsize, block=block, ssl_context=ctx
        )


SESSION = requests.Session()
SESSION.mount("https://", _OSStoreAdapter())

API_KEY = os.environ.get("GEMINI_API_KEY", "")
SA_KEY_JSON = os.environ.get("GEMINI_SA_KEY", "")
SA_PROJECT = os.environ.get("GCP_AI_STUDIO_PROJECT", "") or (
    os.environ.get("GCP_AI_STUDIO_PROJECTS", "").split(",")[0].strip()
)
MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
WORKERS = int(os.environ.get("WORKERS", "20"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "6"))

ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
)

USE_OAUTH = bool(SA_KEY_JSON)
_oauth_creds = None
_oauth_lock = threading.Lock()


def _get_oauth_token():
    """Mint/refresh un OAuth token via le service account (cache thread-safe)."""
    global _oauth_creds
    with _oauth_lock:
        if _oauth_creds is None:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request as GoogleAuthRequest

            sa_info = json.loads(SA_KEY_JSON)
            _oauth_creds = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/generative-language"],
            )
        if not _oauth_creds.valid:
            from google.auth.transport.requests import Request as GoogleAuthRequest

            _oauth_creds.refresh(GoogleAuthRequest())
        return _oauth_creds.token


def _build_request():
    """Construit (url, headers) selon le mode d'auth."""
    if USE_OAUTH:
        token = _get_oauth_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if SA_PROJECT:
            headers["x-goog-user-project"] = SA_PROJECT
        return ENDPOINT, headers
    else:
        url = f"{ENDPOINT}?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        return url, headers

STYLE_PREFIX = (
    "Read aloud with a natural, expressive tone, adapting emotion and pacing to the content:"
)

# Instruction de langue explicite pour les langues que Gemini risque de mal
# identifier d'apres le seul texte (cantonais lu en mandarin, nynorsk, etc.).
LANG_INSTRUCTION = {
    "yue": "Speak the following entirely in Cantonese (Hong Kong Yue Chinese), not Mandarin.",
    "fa": "Speak the following entirely in Persian (Farsi).",
    "hr": "Speak the following entirely in Croatian.",
    "sl": "Speak the following entirely in Slovenian.",
    "nn": "Speak the following entirely in Norwegian (Nynorsk).",
}

PCM_SAMPLE_RATE = 24_000
PCM_SAMPLE_WIDTH = 2  # 16-bit
PCM_CHANNELS = 1


def call_gemini_tts(text, voice_name, lang=None):
    """Appelle l'API Gemini TTS et retourne les bytes PCM bruts (16-bit 24kHz mono)."""
    instruction = LANG_INSTRUCTION.get(lang)
    prompt = f"{instruction} {STYLE_PREFIX} {text}" if instruction else f"{STYLE_PREFIX} {text}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}
            },
        },
    }

    backoff = 2.0
    last_err = None
    for _ in range(MAX_RETRIES):
        url, headers = _build_request()
        try:
            r = SESSION.post(url, headers=headers, json=body, timeout=180)
        except requests.RequestException as e:
            last_err = f"network: {e}"
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 30)
            continue

        if r.status_code == 200:
            data = r.json()
            try:
                audio_b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
                return base64.b64decode(audio_b64)
            except (KeyError, IndexError):
                # finishReason "OTHER" + content {} -> flakiness modele, on retry
                last_err = f"no audio in response: {str(data)[:200]}"
                time.sleep(backoff)
                backoff = min(backoff * 1.7, 30)
                continue

        if r.status_code in (400, 404):
            raise RuntimeError(f"permanent {r.status_code}: {r.text[:300]}")

        if r.status_code in (429, 500, 502, 503, 504):
            last_err = f"HTTP {r.status_code}"
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(float(retry_after))
                except ValueError:
                    time.sleep(backoff)
            else:
                time.sleep(backoff)
            backoff = min(backoff * 1.7, 30)
            continue

        # autres status -> retry quelques fois
        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        time.sleep(backoff)
        backoff = min(backoff * 1.7, 30)

    raise RuntimeError(f"max retries exhausted; last={last_err}")


def pcm_to_mp3(pcm_bytes, out_path):
    """Encode le PCM brut en MP3 192k a la destination. Retourne la duree."""
    seg = AudioSegment(
        data=pcm_bytes,
        sample_width=PCM_SAMPLE_WIDTH,
        frame_rate=PCM_SAMPLE_RATE,
        channels=PCM_CHANNELS,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    seg.export(tmp_path, format="mp3", bitrate="192k")
    os.replace(tmp_path, out_path)

    return len(seg) / 1000.0


def process_one(args):
    voice_name, lang, text, file_rel, root, skip_existing = args
    out_path = root / file_rel
    if skip_existing and out_path.exists():
        return ("skip", voice_name, lang, file_rel, None, None)
    try:
        pcm = call_gemini_tts(text, voice_name, lang)
        dur = pcm_to_mp3(pcm, out_path)
        return ("ok", voice_name, lang, file_rel, dur, None)
    except Exception as e:
        return ("err", voice_name, lang, file_rel, str(e), None)


def build_tasks(voices_data, texts_data, root, voice_filter, lang_filter, limit):
    tasks = []
    for vname, vdata in voices_data["voices"].items():
        # Ne traiter QUE les voix Gemini : voices.json contient aussi des voix
        # openai (Azure) et minimax (Replicate) qu'il ne faut pas synthetiser
        # via l'API Gemini. (Absence de provider = gemini, retro-compat.)
        if vdata.get("provider", "gemini") != "gemini":
            continue
        if voice_filter and vname.lower() != voice_filter.lower():
            continue

        # Lookup case-insensitive du texte
        text_key = vname if vname in texts_data else None
        if text_key is None:
            for k in texts_data:
                if k.lower() == vname.lower():
                    text_key = k
                    break
        if text_key is None:
            print(f"WARN: pas de texts pour la voix {vname}", file=sys.stderr)
            continue

        for lang, file_rel in vdata["files"].items():
            if lang_filter and lang != lang_filter:
                continue
            text = texts_data[text_key].get(lang)
            if not text:
                continue
            tasks.append((vname, lang, text, file_rel, root, None))

    if limit:
        tasks = tasks[:limit]
    return tasks


def parse_args(argv):
    skip_existing = False
    voice_filter = None
    lang_filter = None
    limit = None

    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--skip-existing":
            skip_existing = True
        elif a == "--voice" and i + 1 < len(argv):
            voice_filter = argv[i + 1]
            i += 1
        elif a == "--lang" and i + 1 < len(argv):
            lang_filter = argv[i + 1]
            i += 1
        elif a == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
            i += 1
        else:
            print(f"Argument inconnu: {a}", file=sys.stderr)
            sys.exit(2)
        i += 1

    return skip_existing, voice_filter, lang_filter, limit


def main():
    if not USE_OAUTH and not API_KEY:
        print(
            "ERROR: defini soit GEMINI_SA_KEY (+ GCP_AI_STUDIO_PROJECT) "
            "soit GEMINI_API_KEY",
            file=sys.stderr,
        )
        sys.exit(1)

    skip_existing, voice_filter, lang_filter, limit = parse_args(sys.argv)

    root = Path(__file__).parent
    with open(root / "voices.json", encoding="utf-8") as f:
        voices_data = json.load(f)
    with open(root / "texts.json", encoding="utf-8") as f:
        texts_data = json.load(f)

    raw_tasks = build_tasks(voices_data, texts_data, root, voice_filter, lang_filter, limit)
    tasks = [(vn, lg, tx, fr, rt, skip_existing) for (vn, lg, tx, fr, rt, _) in raw_tasks]

    print(f"Modele       : {MODEL}")
    print(f"Auth         : {'OAuth (SA)' if USE_OAUTH else 'API key'}")
    print(f"Workers      : {WORKERS}")
    print(f"Skip existing: {skip_existing}")
    print(f"Style prefix : {STYLE_PREFIX}")
    print(f"Taches       : {len(tasks)}")
    print("-" * 70, flush=True)

    ok = err = skip = 0
    errs = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_one, t): (t[0], t[1]) for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            status, vname, lang, file_rel, a, _ = fut.result()
            if status == "ok":
                ok += 1
                print(
                    f"[{i}/{len(tasks)}] OK  {vname}/{lang}  "
                    f"{a:.2f}s  ({file_rel})",
                    flush=True,
                )
            elif status == "skip":
                skip += 1
                print(f"[{i}/{len(tasks)}] SKIP {vname}/{lang} (existe deja)", flush=True)
            else:
                err += 1
                errs.append((vname, lang, a))
                print(f"[{i}/{len(tasks)}] ERR {vname}/{lang}: {a}", flush=True)

    elapsed = time.time() - t_start
    print("-" * 70)
    print(f"OK: {ok}  SKIP: {skip}  ERR: {err}  (en {elapsed:.1f}s)")
    if errs:
        print("\nErreurs (max 30 affichees):")
        for vn, lg, msg in errs[:30]:
            print(f"  {vn}/{lg}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
