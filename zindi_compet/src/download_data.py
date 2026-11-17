"""Downloads the raw files of the Zindi challenge into data/raw/.

Requires a .env file at the project root with:
    ZINDI_USERNAME=...
    ZINDI_PASSWORD=...
    ZINDI_CHALLENGE_ID=financial-stress-prediction-challenge-2026-09-01

Usage:
    python src/download_data.py

Note: the download_dataset() function of the `zindi` package (installed version) sends a
`data={"auth_token": ...}` parameter in addition to the header on a GET request, which
makes the download fail with a 403 from the Zindi API. We therefore reuse the
package's low-level API (signin, select_a_challenge) but download the files
ourselves with a standard GET request.
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
            "ZINDI_USERNAME and ZINDI_PASSWORD must be set in .env "
            "(see .env.example)."
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
        sys.exit("No data files found for this challenge.")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    for item in datafiles:
        filename = item["filename"]
        url = f"{BASE_API}/{challenge_id}/files/{filename}"
        destination = RAW_DATA_DIR / filename
        download_file(url, destination, auth_token)
        downloaded_files.append(str(destination))

    print(f"\n{len(downloaded_files)} file(s) downloaded to {RAW_DATA_DIR}:")
    for f in downloaded_files:
        print(f" - {f}")


if __name__ == "__main__":
    main()
