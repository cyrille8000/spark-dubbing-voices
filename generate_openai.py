"""
Genere les samples audio des voix OpenAI via l'API OpenAI Speech
(modele gpt-4o-mini-tts) pour les 40 langues du repo.

Pour chaque (voice, lang) declaree dans voices.json (provider == "openai") :
  1. Recupere le texte depuis texts.json (cle = nom de voix lowercase, ex "alloy")
  2. Genere via POST /v1/audio/speech (voice = champ `openai_voice`)
  3. Ecrit le MP3 a la destination indiquee dans voices.json
     (ex: female/fr/alloy.mp3) en ECRASANT le fichier existant.

Le trim des silences est une etape separee (trim_silences.py --in-place).

MOTEUR : jusqu'au 2026-08-10 ces echantillons sortaient d'Azure Speech
(voix en-US-XxxTurboMultilingualNeural, qui imitaient les voix OpenAI). La gamme
lite du produit appelle desormais OpenAI EN DIRECT, donc les echantillons aussi :
un extrait doit faire entendre le moteur qui doublera reellement la video.
La langue n'est pas un parametre de l'API — le modele la deduit du texte.

Variables d'environnement :
  OPENAI_API_KEY  (requis)
  OPENAI_TTS_MODEL (defaut: gpt-4o-mini-tts)
  WORKERS         (defaut: 8)
  MAX_RETRIES     (defaut: 5)

Flags CLI :
  --skip-existing       N'ecrase pas les fichiers deja presents
  --limit N             Ne traite que N taches (utile pour tests)
  --voice <name>        Filtre sur une voix (ex: --voice alloy)
  --lang <code>         Filtre sur une langue (ex: --lang fr)
"""

import os
import sys
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
WORKERS = int(os.environ.get("WORKERS", "8"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))

API_URL = "https://api.openai.com/v1/audio/speech"

# Erreurs transitoires : 429 (rate limit) et 5xx. Les limites du compte
# (5 000 RPM / 600 000 TPM) sont tres au-dessus de ce que ce script demande,
# donc un 429 ici signale surtout une rafale ; un backoff court suffit.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def synth_to_mp3(openai_voice, text, out_path):
    """Synthese OpenAI -> MP3 24 kHz mono. Ecriture atomique via .tmp.

    Retourne (True, None) ou (False, message). Ne leve pas : l'appelant
    decide du retry.
    """
    body = json.dumps({
        "model": MODEL,
        "voice": openai_voice,
        "input": text,
        "response_format": "mp3",
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            audio = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        return False, f"HTTP {e.code}: {detail}"
    except Exception as e:  # noqa: BLE001 - reseau/TLS/timeout
        return False, f"{type(e).__name__}: {e}"

    if len(audio) < 1024:
        return False, f"reponse trop courte ({len(audio)}B)"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(audio)
    os.replace(tmp_path, out_path)
    return True, None


def call_with_retries(openai_voice, text, out_path):
    """Wrapper avec retry exponentiel sur erreur transitoire."""
    backoff = 1.5
    last_err = None
    for _ in range(MAX_RETRIES):
        ok, err = synth_to_mp3(openai_voice, text, out_path)
        if ok:
            return True, None
        last_err = err or "unknown"
        retryable = any(f"HTTP {code}" in last_err for code in RETRYABLE_STATUS) or any(
            k in last_err.lower() for k in ("timeout", "connection", "remotedisconnected", "ssl")
        )
        if not retryable:
            return False, last_err
        time.sleep(backoff)
        backoff = min(backoff * 1.7, 20)
    return False, f"max retries exhausted; last={last_err}"


def process_one(args):
    voice_name, openai_voice, lang, text, file_rel, root, skip_existing = args
    out_path = root / file_rel
    if skip_existing and out_path.exists():
        return ("skip", voice_name, lang, file_rel, None)

    ok, err = call_with_retries(openai_voice, text, out_path)
    if ok:
        return ("ok", voice_name, lang, file_rel, None)
    return ("err", voice_name, lang, file_rel, err)


def build_tasks(voices_data, texts_data, root, voice_filter, lang_filter, limit, skip_existing):
    tasks = []
    for vname, vdata in voices_data["voices"].items():
        if vdata.get("provider") != "openai":
            continue
        if voice_filter and vname.lower() != voice_filter.lower():
            continue

        # `openai_voice` = l'identifiant de voix de l'API (alloy, ash, ballad...).
        # Repli sur le nom du catalogue : les deux coincident aujourd'hui.
        openai_voice = vdata.get("openai_voice") or vname.lower()

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
            tasks.append((vname, openai_voice, lang, text, file_rel, root, skip_existing))

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
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY non defini", file=sys.stderr)
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
            status, vname, lang, file_rel, payload = fut.result()
            if status == "ok":
                ok += 1
                print(f"[{i}/{len(tasks)}] OK  {vname}/{lang}  ({file_rel})", flush=True)
            elif status == "skip":
                skip += 1
                print(f"[{i}/{len(tasks)}] SKIP {vname}/{lang} (existe deja)", flush=True)
            else:
                err += 1
                errs.append((vname, lang, payload))
                print(f"[{i}/{len(tasks)}] ERR {vname}/{lang}: {payload}", flush=True)

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
