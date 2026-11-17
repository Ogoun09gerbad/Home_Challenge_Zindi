"""Build kaggle/climx_native.ipynb: one self-contained notebook embedding this folder's code.

Run after any change to the code: python climx_emulator/build_kaggle_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "kaggle" / "climx_native.ipynb"
MODULES = ["common.py", "model.py", "native.py", "make_submission.py", "run_native.py"]
OFFICIAL_COMMIT = "c1b2eab8c9f79bda04c6455dc4745335dc4dc5ab"
PINS = """xarray==2024.7.0
dask==2024.8.1
numpy==2.1.1
pandas==2.2.3
scipy==1.13.1
netcdf4==1.7.1
h5netcdf==1.3.0
cftime==1.6.4
bottleneck
pyarrow==17.0.0
cf-xarray==0.10.8
xclim==0.59.1
pint-xarray
huggingface_hub
tqdm
"""

INTRO = """# ClimX : émulateur par analogues forcés, pleine résolution

Ce notebook télécharge les données 192×288, en extrait les cellules terrestres, émule SSP2-4.5 et écrit
des fichiers de soumission de plus en plus précis (12, puis 24, puis 48 réalisations).

**Avant de lancer**
1. Le notebook doit être rattaché à la compétition `climx-main-track`.
2. Panneau *Settings* : **Internet ON**, *Accelerator* : **None**.
3. Cliquez sur **Save Version → Save & Run All** : l'exécution continue une fois le navigateur fermé.

**Après l'exécution**, ouvrez l'onglet *Output* de la version enregistrée et soumettez le fichier
`submission_native_k48_m*.parquet` le plus grand (`m` est le nombre de réalisations).

Si l'exécution s'arrête en route, les fichiers déjà écrits restent valables.
"""

SETTINGS = """# Réglages. Les valeurs par défaut conviennent à une première exécution.
STAGES = [12, 24, 48]      # réalisations : un fichier de soumission à chaque étape
N_ANALOGUES = 48           # années candidates par année cible
RUNS = ["hist", "ssp370", "ssp126", "ssp585"]  # ordre de priorité si le disque manque
RUN_VALIDATION = True      # à la fin, score local avec SSP3-7.0 exclu, si le temps le permet
VALIDATION_REALIZATIONS = 8
TIME_BUDGET_HOURS = 11.0   # marge sous la limite de session de Kaggle
"""

CHECK = """import os, shutil, signal, subprocess, sys, time, glob, json, urllib.request
from pathlib import Path

START = time.time()
WORK = Path("/kaggle/temp/climx"); WORK.mkdir(parents=True, exist_ok=True)   # hors de la sortie (limitée en taille)
OUTPUT = Path("/kaggle/working/outputs"); OUTPUT.mkdir(parents=True, exist_ok=True)
CODE = Path("/kaggle/temp/code"); CODE.mkdir(parents=True, exist_ok=True)

free = shutil.disk_usage(WORK).free / 1e9
ram = int(open("/proc/meminfo").read().split()[1]) / 1e6
WORKERS = os.cpu_count() or 4
print(f"Python {sys.version.split()[0]} | {WORKERS} CPU | RAM {ram:.0f} GB | disque libre {free:.0f} GB")
# Pic : magasin complet (35 Go) + un fichier source (33 Go). Avec moins, les derniers scénarios seront ignorés.
if free < 70:
    print(f"ATTENTION : {free:.0f} GB libres pour 70 GB souhaités ; les derniers scénarios seront ignorés.")
assert free > 55, "Disque insuffisant pour l'historique et un scénario."
if ram < 12:
    WORKERS = 2
try:
    urllib.request.urlopen("https://huggingface.co", timeout=30)
except Exception as error:
    raise RuntimeError("Internet inaccessible : activez Internet ON dans Settings, puis relancez.") from error
print("Internet OK")
"""

SETUP = f"""# Environnement isolé : Python 3.11 et les versions du dépôt officiel (l'index officiel dépend de xclim 0.59.1).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
VENV = Path("/kaggle/temp/venv")
subprocess.run(UV + ["venv", str(VENV), "--python", "3.11", "--clear"], check=True)
PY = str(VENV / "bin" / "python")
(CODE / "requirements.txt").write_text({PINS!r})
subprocess.run(UV + ["pip", "install", "-q", "--python", PY, "-r", str(CODE / "requirements.txt")], check=True)

# Code officiel (indices, métrique, format), à un commit fixé.
REPO = Path("/kaggle/temp/repo_ClimX")
if not (REPO / "rnNSE.py").exists():
    REPO.mkdir(parents=True, exist_ok=True)
    for command in (["init", "-q"],
                    ["fetch", "-q", "--depth", "1", "https://github.com/IPL-UV/ClimX.git", "{OFFICIAL_COMMIT}"],
                    ["checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(["git", "-C", str(REPO)] + command, check=True)

# Données lite (état forcé et choix des analogues) : fournies par la compétition, sinon Hugging Face.
ENV = dict(os.environ, CLIMX_REPO=str(REPO), CLIMX_WORK=str(WORK), CLIMX_OUT=str(OUTPUT), HF_HUB_DISABLE_XET="1")
found = glob.glob("/kaggle/input/**/*subsampled_16_test_forcings_ssp245.nc", recursive=True)
if found:
    ENV["CLIMX_LITE"] = str(Path(found[0]).parent)
else:
    print("Données lite non montées : téléchargement depuis Hugging Face.")
    script = "from huggingface_hub import snapshot_download as s; s('isp-uv-es/ClimX', repo_type='dataset', local_dir='/kaggle/temp/lite_data', allow_patterns=['lite/*'])"
    subprocess.run([PY, "-c", script], check=True, env=ENV)
    ENV["CLIMX_LITE"] = "/kaggle/temp/lite_data/lite"
print("Données lite :", ENV["CLIMX_LITE"])
print(subprocess.run([PY, "-c", "import xclim, xarray, numpy; print('xclim', xclim.__version__, 'xarray', xarray.__version__, 'numpy', numpy.__version__)"], capture_output=True, text=True, check=True).stdout)
"""

DRIVER = """SCRIPT = str(CODE / "run_native.py")
RUN_CWD = Path("/kaggle/temp/cwd"); RUN_CWD.mkdir(exist_ok=True)
LOG = OUTPUT / "run_log.txt"

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def run(*args):
    \"\"\"Lance une étape avec journal en direct. Renvoie False si elle échoue ou dépasse le temps restant.\"\"\"
    limit = remaining()
    if limit <= 60:
        print("Temps écoulé : étape ignorée :", args)
        return False
    command = [PY, "-u", SCRIPT, *map(str, args)]
    print(f"\\n=== {' '.join(map(str, args))} ({limit / 3600:.1f} h restantes) ===", flush=True)
    started = time.time()
    process = subprocess.Popen(command, cwd=RUN_CWD, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    tail = []
    with open(LOG, "a") as log:
        try:
            for line in process.stdout:
                log.write(line); log.flush()
                tail = (tail + [line])[-30:]
                if "Warning" not in line and "warn" not in line:
                    print(line, end="", flush=True)
                if time.time() - started > limit:
                    raise TimeoutError
            code = process.wait()
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            print("Temps écoulé : étape arrêtée.")
            return False
    print(f"=== terminé en {(time.time() - started) / 60:.0f} min, code {code} ===", flush=True)
    if code != 0:
        print("".join(tail))
    return code == 0
"""

FETCH = """assert run("fetch", "--runs", *RUNS), "Échec du téléchargement ou de l'extraction : voir le journal ci-dessus."
store = WORK / "store_native"
for name in RUNS:
    meta = store / name / "meta.json"
    print(name, json.loads(meta.read_text()) if meta.exists() else "non extrait")
print(f"Disque libre : {shutil.disk_usage(WORK).free / 1e9:.0f} GB ; temps écoulé : {(time.time() - START) / 3600:.1f} h")
"""

SUBMIT = """written = []
for realizations in STAGES:
    if not run("submit", "--realizations", realizations, "--n-analogues", N_ANALOGUES, "--workers", WORKERS):
        break
    written.append(realizations)
    print("Fichiers prêts :", sorted(p.name for p in OUTPUT.glob("*.parquet")))
assert written, "Aucun fichier de soumission écrit : voir le journal ci-dessus."
"""

VALIDATE = """# Score local en pleine résolution, avec SSP3-7.0 exclu : il ne prévoit pas le score Kaggle, mais détaille les indices.
if RUN_VALIDATION:
    run("validate", "--holdout", "ssp370", "--realizations", VALIDATION_REALIZATIONS, "--n-analogues", N_ANALOGUES, "--workers", WORKERS)
"""

SUMMARY = """for path in sorted(OUTPUT.iterdir()):
    print(f"{path.name:50s} {path.stat().st_size / 1e6:9.1f} MB")
for path in sorted(OUTPUT.glob("native_validation_*.json")):
    scores = json.loads(path.read_text())["scores"]
    print("\\n" + path.name, f"\\n  global {scores['overall']:+.4f}")
    print("  " + "  ".join(f"{k} {v:+.2f}" for k, v in scores.items() if k != "overall"))
print(f"\\nDurée totale : {(time.time() - START) / 3600:.1f} h")
print("À soumettre : onglet Output, le fichier submission_native avec le plus grand m.")
"""


def cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    return base if kind == "markdown" else {**base, "outputs": [], "execution_count": None}


def module_cell(name: str) -> dict:
    """A cell that writes one project file, without any character the raw string could not hold."""
    text = (HERE / name).read_text(encoding="utf8")
    return cell("code", f"(CODE / {name!r}).write_text({text!r}, encoding='utf8')\nprint({name!r}, 'écrit')\n")


def main() -> None:
    cells = [
        cell("markdown", INTRO),
        cell("code", SETTINGS),
        cell("code", CHECK),
        cell("markdown", "## Code du projet\nLes modules sont écrits tels quels depuis le projet local.\n"),
        *[module_cell(name) for name in MODULES],
        cell("markdown", "## Environnement\n"),
        cell("code", SETUP),
        cell("code", DRIVER),
        cell("markdown", "## Données pleine résolution\nTéléchargement, contrôle d'empreinte, extraction, suppression de la source, fichier par fichier.\n"),
        cell("code", FETCH),
        cell("markdown", "## Émulation et fichiers de soumission\n"),
        cell("code", SUBMIT),
        cell("markdown", "## Validation\n"),
        cell("code", VALIDATE),
        cell("code", SUMMARY),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.parent.mkdir(exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"{TARGET} ({TARGET.stat().st_size / 1e3:.0f} kB, {len(cells)} cellules)")


if __name__ == "__main__":
    main()
