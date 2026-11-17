"""Télécharge les fichiers bruts du défi Zindi dans data/raw/.

Nécessite un fichier .env à la racine du projet avec :
    ZINDI_USERNAME=...
    ZINDI_PASSWORD=...
    ZINDI_CHALLENGE_ID=financial-stress-prediction-challenge-2026-09-01

Usage :
    python src/download_data.py

Note : la fonction download_dataset() du package `zindi` (v. installée) envoie un
paramètre `data={"auth_token": ...}` en plus du header sur une requête GET, ce qui
fait échouer le téléchargement avec un 403 côté API Zindi. On réutilise donc l'API
bas niveau du package (signin, select_a_challenge) mais on télécharge les fichiers
nous-mêmes avec une requête GET standard.
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from zindi.user import Zindian

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
BASE_API = "https://api.zindi.africa/v1/competitions"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def download_file(url, destination, auth_token):
    response = requests.get(
        url, headers={**HEADERS, "auth_token": auth_token}, stream=True
    )
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(destination, "wb") as f, tqdm(
        desc=destination.name, total=total, unit="o", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    load_dotenv(ROOT_DIR / ".env")

    username = os.environ.get("ZINDI_USERNAME")
    password = os.environ.get("ZINDI_PASSWORD")
    challenge_id = os.environ.get(
        "ZINDI_CHALLENGE_ID",
        "financial-stress-prediction-challenge-2026-09-01",
    )

    if not username or not password:
        sys.exit(
            "ZINDI_USERNAME et ZINDI_PASSWORD doivent être définis dans .env "
            "(voir .env.example)."
        )

    user = Zindian(username=username, fixed_password=password)
    challenge_result = user.select_a_challenge(challenge_id=challenge_id)

    auth_token = requests.post(
        "https://api.zindi.africa/v1/auth/signin",
        data={"username": username, "password": password},
        headers=HEADERS,
    ).json()["data"]["auth_token"]

    datafiles = challenge_result["challenge"].get("datafiles", [])
    if not datafiles:
        sys.exit("Aucun fichier de données trouvé pour ce défi.")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    for item in datafiles:
        filename = item["filename"]
        url = f"{BASE_API}/{challenge_id}/files/{filename}"
        destination = RAW_DATA_DIR / filename
        download_file(url, destination, auth_token)
        downloaded_files.append(str(destination))

    print(f"\n{len(downloaded_files)} fichier(s) téléchargé(s) dans {RAW_DATA_DIR} :")
    for f in downloaded_files:
        print(f" - {f}")


if __name__ == "__main__":
    main()
