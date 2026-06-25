"""
Genere les samples audio des voix MiniMax via Replicate (modele
`minimax/speech-2.8-turbo`).

Les voix MiniMax NE sont PAS multilingues : chaque voix appartient a une seule
langue (prefixe du voice_id, ex `French_CasualMan`). Dans voices.json chaque
voix `provider == "minimax"` a donc exactement une entree dans `files`.

Pour chaque voix :
  1. Recupere le texte depuis texts.json (cle = voice_id, une seule langue)
  2. Appelle Replicate (Prefer: wait) avec :
        text, voice_id, channel=mono, emotion=auto, audio_format=wav,
        language_boost=<valeur stockee dans voices.json>
  3. Telecharge l'URL de sortie (wav) et l'ecrit a la destination indiquee
     dans voices.json (ex: minimax/fr/French_CasualMan.wav) en ECRASANT.

Variables d'environnement :
  REPLICATE_API_TOKEN   (requis)
  WORKERS               (defaut 6)
  MAX_RETRIES           (defaut 6)
  POLL_TIMEOUT          (defaut 300, secondes max d'attente d'une prediction)

Flags CLI :
  --skip-existing       N'ecrase pas les fichiers deja presents (idempotent)
  --limit N             Ne traite que N taches (utile pour tests)
  --voice <id>          Filtre sur un voice_id (ex: --voice French_CasualMan)
  --lang <code>         Filtre sur une langue (ex: --lang fr)
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import ssl
import requests
import certifi
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

warnings.filterwarnings("ignore")


# --- TLS : magasin de certificats de l'OS EN PLUS de certifi ---
# Derriere un proxy/antivirus qui intercepte le HTTPS, le CA du proxy est dans
# le magasin Windows mais pas dans certifi -> "unable to get local issuer
# certificate". On combine les deux trust stores. (OpenSSL ne verifie pas la
# revocation, donc pas de CRYPT_E_NO_REVOCATION_CHECK comme avec Schannel.)
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

API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
WORKERS = int(os.environ.get("WORKERS", "6"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "6"))
POLL_TIMEOUT = float(os.environ.get("POLL_TIMEOUT", "300"))

MODEL = "minimax/speech-2.8-turbo"
PREDICT_URL = f"https://api.replicate.com/v1/models/{MODEL}/predictions"

# Parametres fixes demandes (cf. cahier des charges)
CHANNEL = "mono"
EMOTION = "auto"
AUDIO_FORMAT = "wav"

# language_boost MiniMax par code langue. Le boost suit la LANGUE du fichier
# (et non la langue native de la voix) : les voix anglaises cross-lingual
# servent les langues sans voix dediee via le boost de la langue cible.
LANG_TO_BOOST = {
    "en": "English", "cmn": "Chinese", "ja": "Japanese", "ko": "Korean",
    "es": "Spanish", "pt": "Portuguese", "fr": "French", "id": "Indonesian",
    "de": "German", "ru": "Russian", "it": "Italian", "nl": "Dutch",
    "vi": "Vietnamese", "ar": "Arabic", "tr": "Turkish", "uk": "Ukrainian",
    "af": "Afrikaans", "bg": "Bulgarian", "ca": "Catalan", "cs": "Czech",
    "da": "Danish", "el": "Greek", "fi": "Finnish", "fil": "Filipino",
    "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian", "ms": "Malay",
    "nb": "Norwegian", "pl": "Polish", "ro": "Romanian", "sk": "Slovak",
    "sv": "Swedish", "ta": "Tamil", "th": "Thai",
}

TERMINAL = {"succeeded", "failed", "canceled"}


def _headers(wait=False):
    h = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    if wait:
        h["Prefer"] = "wait"
    return h


def create_prediction(text, voice_id, language_boost):
    """Cree une prediction (Prefer: wait) et retourne le JSON de la prediction."""
    body = {
        "input": {
            "text": text,
            "voice_id": voice_id,
            "channel": CHANNEL,
            "emotion": EMOTION,
            "audio_format": AUDIO_FORMAT,
            "language_boost": language_boost,
        }
    }
    r = SESSION.post(PREDICT_URL, headers=_headers(wait=True), json=body, timeout=120)
    return r


def poll_prediction(get_url):
    """Poll une prediction jusqu'a un etat terminal ou POLL_TIMEOUT."""
    deadline = time.time() + POLL_TIMEOUT
    while True:
        r = SESSION.get(get_url, headers=_headers(), timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"poll HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if data.get("status") in TERMINAL:
            return data
        if time.time() > deadline:
            raise RuntimeError(f"poll timeout (status={data.get('status')})")
        time.sleep(1.5)


def synth_one(text, voice_id, language_boost):
    """Retourne (audio_bytes) ou leve une exception apres retries."""
    backoff = 2.0
    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            r = create_prediction(text, voice_id, language_boost)
        except requests.RequestException as e:
            last_err = f"network: {e}"
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 30)
            continue

        if r.status_code in (200, 201):
            data = r.json()
            # Avec Prefer: wait la prediction peut ne pas etre encore terminee.
            if data.get("status") not in TERMINAL:
                get_url = (data.get("urls") or {}).get("get")
                if not get_url:
                    last_err = f"no get url; status={data.get('status')}"
                    time.sleep(backoff)
                    backoff = min(backoff * 1.7, 30)
                    continue
                data = poll_prediction(get_url)

            status = data.get("status")
            if status == "succeeded":
                out = data.get("output")
                # output = URL string (schema: type string / uri)
                if isinstance(out, list):
                    out = out[0] if out else None
                if not out:
                    last_err = "succeeded but empty output"
                    time.sleep(backoff)
                    backoff = min(backoff * 1.7, 30)
                    continue
                ar = SESSION.get(out, timeout=180)
                if ar.status_code == 200 and ar.content:
                    return ar.content
                last_err = f"download HTTP {ar.status_code}"
                time.sleep(backoff)
                backoff = min(backoff * 1.7, 30)
                continue

            # failed / canceled
            last_err = f"prediction {status}: {str(data.get('error'))[:200]}"
            # erreurs modele -> rarement transitoires, mais on retente qqs fois
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 30)
            continue

        if r.status_code == 422:
            # input invalide -> permanent, inutile de retenter
            raise RuntimeError(f"permanent 422: {r.text[:300]}")

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

        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        time.sleep(backoff)
        backoff = min(backoff * 1.7, 30)

    raise RuntimeError(f"max retries exhausted; last={last_err}")


def write_audio(audio_bytes, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    os.replace(tmp_path, out_path)


def process_one(args):
    vid, voice_id, lang, language_boost, text, file_rel, root, skip_existing = args
    out_path = root / file_rel
    if skip_existing and out_path.exists():
        return ("skip", vid, lang, file_rel, None)
    try:
        audio = synth_one(text, voice_id, language_boost)
        write_audio(audio, out_path)
        return ("ok", vid, lang, file_rel, None)
    except Exception as e:
        return ("err", vid, lang, file_rel, str(e))


def lookup_text(texts_data, vid, lang):
    entry = texts_data.get(vid)
    if entry is None:
        # fallback case-insensitive
        for k in texts_data:
            if k.lower() == vid.lower():
                entry = texts_data[k]
                break
    if entry is None:
        return None
    return entry.get(lang)


def build_tasks(voices_data, texts_data, root, voice_filter, lang_filter, limit, skip_existing):
    tasks = []
    for vid, vdata in voices_data["voices"].items():
        if vdata.get("provider") != "minimax":
            continue
        if voice_filter and vid.lower() != voice_filter.lower():
            continue

        voice_id = vdata.get("minimax_voice_id", vid)

        for lang, file_rel in vdata["files"].items():
            if lang_filter and lang != lang_filter:
                continue
            # language_boost derive de la LANGUE du fichier (pas de la voix) :
            # une meme voix anglaise sert plusieurs langues (cross-lingual).
            language_boost = LANG_TO_BOOST.get(lang, vdata.get("language_boost", "None"))
            text = lookup_text(texts_data, vid, lang)
            if not text:
                print(f"WARN: pas de texte pour {vid}/{lang}", file=sys.stderr)
                continue
            tasks.append((vid, voice_id, lang, language_boost, text, file_rel, root, skip_existing))

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
    if not API_TOKEN:
        print("ERROR: REPLICATE_API_TOKEN non defini", file=sys.stderr)
        sys.exit(1)

    skip_existing, voice_filter, lang_filter, limit = parse_args(sys.argv)

    root = Path(__file__).parent
    with open(root / "voices.json", encoding="utf-8") as f:
        voices_data = json.load(f)
    with open(root / "texts.json", encoding="utf-8") as f:
        texts_data = json.load(f)

    tasks = build_tasks(
        voices_data, texts_data, root,
        voice_filter, lang_filter, limit, skip_existing,
    )

    print(f"Modele       : {MODEL}")
    print(f"Params       : channel={CHANNEL} emotion={EMOTION} audio_format={AUDIO_FORMAT}")
    print(f"Workers      : {WORKERS}")
    print(f"Skip existing: {skip_existing}")
    print(f"Taches       : {len(tasks)}")
    print("-" * 70, flush=True)

    ok = err = skip = 0
    errs = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_one, t): (t[0], t[2]) for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            status, vid, lang, file_rel, payload = fut.result()
            if status == "ok":
                ok += 1
                print(f"[{i}/{len(tasks)}] OK  {vid}/{lang}  ({file_rel})", flush=True)
            elif status == "skip":
                skip += 1
                print(f"[{i}/{len(tasks)}] SKIP {vid}/{lang} (existe deja)", flush=True)
            else:
                err += 1
                errs.append((vid, lang, payload))
                print(f"[{i}/{len(tasks)}] ERR {vid}/{lang}: {payload}", flush=True)

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
