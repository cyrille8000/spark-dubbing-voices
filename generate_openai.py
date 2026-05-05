"""
Genere les samples audio des 8 voix style OpenAI via Azure Speech HD
(modeles XxxTurboMultilingualNeural) pour les 35 langues du repo.

Pour chaque (voice, lang) declaree dans voices.json (provider == "openai") :
  1. Recupere le texte depuis texts.json (cle = nom de voix lowercase, ex "alloy")
  2. Genere via Azure Speech avec un SSML force xml:lang=<code> (multilingue)
  3. Ecrit le MP3 a la destination indiquee dans voices.json
     (ex: openai/fr/alloy.mp3) en ECRASANT le fichier existant.

Le trim des silences est une etape separee (trim_silences.py --in-place).

Variables d'environnement :
  AZURE_SPEECH_KEY     (requis)
  AZURE_SPEECH_REGION  (defaut: eastus)
  WORKERS              (defaut: 8)
  MAX_RETRIES          (defaut: 5)

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
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import azure.cognitiveservices.speech as speechsdk

warnings.filterwarnings("ignore")

SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")
WORKERS = int(os.environ.get("WORKERS", "8"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))

# Mapping nos codes langue -> codes Azure xml:lang
LANG_MAP = {
    "en": "en-US", "fr": "fr-FR", "es": "es-ES", "de": "de-DE",
    "it": "it-IT", "pt": "pt-BR", "ja": "ja-JP", "ko": "ko-KR",
    "cmn": "zh-CN", "ar": "ar-EG", "hi": "hi-IN", "ru": "ru-RU",
    "tr": "tr-TR", "nl": "nl-NL", "pl": "pl-PL", "id": "id-ID",
    "uk": "uk-UA", "vi": "vi-VN", "th": "th-TH", "ro": "ro-RO",
    "el": "el-GR", "cs": "cs-CZ", "fi": "fi-FI", "bg": "bg-BG",
    "da": "da-DK", "he": "he-IL", "ms": "ms-MY", "sk": "sk-SK",
    "sv": "sv-SE", "fil": "fil-PH", "hu": "hu-HU", "nb": "nb-NO",
    "ca": "ca-ES", "ta": "ta-IN", "af": "af-ZA",
}


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )


def synth_to_mp3(azure_voice, xml_lang, text, out_path):
    """Synthese SSML -> MP3 24kHz 160kbps mono.

    Synthese en memoire (audio_config=None) pour eviter que le SDK Azure
    garde le file handle ouvert sous Windows. Ecriture atomique via .tmp.
    """
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3
    )

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,
    )

    safe_text = _xml_escape(text)
    ssml = (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{xml_lang}">'
        f'<voice name="{azure_voice}">{safe_text}</voice>'
        f'</speak>'
    )

    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            f.write(result.audio_data)
        os.replace(tmp_path, out_path)
        return True, None

    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        msg = f"{details.reason}"
        if details.reason == speechsdk.CancellationReason.Error:
            msg += f": {details.error_details}"
        return False, msg

    return False, f"unexpected reason: {result.reason}"


def call_with_retries(azure_voice, xml_lang, text, out_path):
    """Wrapper avec retry exponentiel sur erreur transitoire."""
    backoff = 1.5
    last_err = None
    for _ in range(MAX_RETRIES):
        ok, err = synth_to_mp3(azure_voice, xml_lang, text, out_path)
        if ok:
            return True, None
        last_err = err or "unknown"
        # Throttling Azure / 5xx -> retry
        msg = (last_err or "").lower()
        retryable = any(k in msg for k in (
            "throttl", "timeout", "503", "500", "429",
            "connectionerror", "network", "temporary",
        ))
        if not retryable:
            return False, last_err
        time.sleep(backoff)
        backoff = min(backoff * 1.7, 20)
    return False, f"max retries exhausted; last={last_err}"


def process_one(args):
    voice_name, azure_voice, lang, text, file_rel, root, skip_existing = args
    out_path = root / file_rel
    if skip_existing and out_path.exists():
        return ("skip", voice_name, lang, file_rel, None)

    xml_lang = LANG_MAP.get(lang)
    if not xml_lang:
        return ("err", voice_name, lang, file_rel, f"no xml:lang mapping for {lang}")

    ok, err = call_with_retries(azure_voice, xml_lang, text, out_path)
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

        azure_voice = vdata.get("azure_voice")
        if not azure_voice:
            print(f"WARN: pas de azure_voice pour {vname}", file=sys.stderr)
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
            tasks.append((vname, azure_voice, lang, text, file_rel, root, skip_existing))

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
    if not SPEECH_KEY:
        print("ERROR: AZURE_SPEECH_KEY non defini", file=sys.stderr)
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

    print(f"Region       : {SPEECH_REGION}")
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
