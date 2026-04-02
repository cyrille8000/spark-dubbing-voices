import requests
import base64
import json

import os
API_KEY = os.environ.get("GCP_TTS_API_KEY", "")
URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={API_KEY}"

payload = {
    "input": {
        "text": "Bonjour, ceci est un test de synthèse vocale avec Google Cloud."
    },
    "voice": {
        "languageCode": "fr-FR",
        "name": "fr-FR-Standard-A",
        "ssmlGender": "FEMALE"
    },
    "audioConfig": {
        "audioEncoding": "MP3",
        "sampleRateHertz": 24000
    }
}

response = requests.post(URL, json=payload)

if response.status_code == 200:
    audio_content = base64.b64decode(response.json()["audioContent"])
    with open("test_gcp.mp3", "wb") as f:
        f.write(audio_content)
    print(f"Audio genere: test_gcp.mp3 ({len(audio_content)} bytes)")
else:
    print(f"Erreur {response.status_code}: {response.text}")
