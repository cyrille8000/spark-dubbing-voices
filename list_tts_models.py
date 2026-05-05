"""Liste les modeles TTS Gemini disponibles via l'API ListModels."""
import os, sys, json, requests, threading

SA_KEY_JSON = os.environ.get("GEMINI_SA_KEY", "")
SA_PROJECT = os.environ.get("GCP_AI_STUDIO_PROJECT", "") or (
    os.environ.get("GCP_AI_STUDIO_PROJECTS", "").split(",")[0].strip()
)

if not SA_KEY_JSON:
    print("GEMINI_SA_KEY missing", file=sys.stderr)
    sys.exit(1)

from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest

creds = service_account.Credentials.from_service_account_info(
    json.loads(SA_KEY_JSON),
    scopes=["https://www.googleapis.com/auth/generative-language"],
)
creds.refresh(GoogleAuthRequest())

headers = {"Authorization": f"Bearer {creds.token}"}
if SA_PROJECT:
    headers["x-goog-user-project"] = SA_PROJECT

r = requests.get(
    "https://generativelanguage.googleapis.com/v1beta/models",
    headers=headers,
    timeout=60,
)
r.raise_for_status()
data = r.json()
models = data.get("models", [])

# Filtrer les modeles susceptibles de faire du TTS / audio
print(f"Total modeles: {len(models)}\n")
print("--- Modeles avec 'tts' ou 'audio' dans le nom ---")
for m in models:
    name = m.get("name", "")
    methods = m.get("supportedGenerationMethods", [])
    if "tts" in name.lower() or "audio" in name.lower() or "speech" in name.lower():
        print(f"  {name}  methods={methods}")

print("\n--- Tous les modeles 'flash' ---")
for m in models:
    name = m.get("name", "")
    if "flash" in name.lower():
        print(f"  {name}")
