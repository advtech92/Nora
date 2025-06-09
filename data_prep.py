#!/usr/bin/env python3
"""
data_prep.py

1) Attempts to download Cornell Movie-Dialogs via ConvoKit (key: "movie-corpus").
   - If ConvoKit/Unsloth fails, falls back to manual ZIP download/extraction.

2) Attempts to download PersonaChat via Hugging Face Datasets:
   - First tries "persona_chat" (older key).
   - If that fails, tries "conv_ai_2" (alias).
   - Catches any exception to skip gracefully.

3) Writes each utterance to:
     data/conversational/cornell_movie_dialogs.txt
     data/conversational/persona_chat.txt

After running, you’ll have:
    data/
    ├── books/                     (your original Gutenberg .txt files)
    └── conversational/
        ├── cornell_movie_dialogs.txt
        └── persona_chat.txt

Then retrain or fine-tune Nora on data/books/ + data/conversational/.
"""

import os
import sys
import zipfile
import tempfile
import urllib.request
from pathlib import Path

# === 1) Attempt to import ConvoKit for Cornell Movie-Dialogs ===
USE_CONVOKIT = True
try:
    from convokit import Corpus, download as convokit_download
except ImportError:
    USE_CONVOKIT = False

# === 2) Attempt to import Hugging Face Datasets ===
HAS_DATASETS = True
try:
    from datasets import load_dataset
except ImportError:
    HAS_DATASETS = False

# Directory for conversational data
CONV_DIR = Path("data/conversational")
CONV_DIR.mkdir(parents=True, exist_ok=True)

# Official ZIP URL (fallback) for Cornell Movie-Dialogs
CORNELL_ZIP_URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"


def install_package(pkg_name: str):
    """
    Installs a Python package using the same Python interpreter,
    wrapping the path in quotes to handle spaces.
    """
    python_executable = sys.executable
    command = f"\"{python_executable}\" -m pip install {pkg_name}"
    print(f"[data_prep] Installing package: {pkg_name}")
    os.system(command)


def prepare_cornell_via_convokit(output_path: str) -> bool:
    """
    Try to download Cornell Movie-Dialogs via ConvoKit (key: "movie-corpus").
    Returns True if successful, False otherwise.
    """
    if not USE_CONVOKIT:
        print("[data_prep] ConvoKit not installed; skipping ConvoKit path.")
        return False

    print("[data_prep] Attempting to download Cornell Movie-Dialogs via ConvoKit...")
    try:
        corpus = Corpus(filename=convokit_download("movie-corpus"))
        with open(output_path, "w", encoding="utf-8") as fout:
            for utt in corpus.iter_utterances():
                text = utt.text.strip()
                if text:
                    fout.write(text.replace("\n", " ") + "\n")
        print(f"[data_prep] Wrote Cornell Movie-Dialogs to {output_path} (via ConvoKit).")
        return True

    except NotImplementedError as nie:
        # Typically due to Unsloth error if GPU unsupported
        print("[data_prep] ConvoKit raised NotImplementedError (Unsloth/GPU issue).")
        print(f"[data_prep] Error: {nie}")
        return False

    except Exception as e:
        print("[data_prep] ConvoKit path failed with exception:", file=sys.stderr)
        print(e, file=sys.stderr)
        return False


def prepare_cornell_manual(output_path: str):
    """
    Fallback: Download Cornell ZIP manually, extract movie_lines.txt,
    and write all utterances to output_path.
    """
    print("[data_prep] Falling back to manual download of Cornell Movie-Dialogs...")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "cornell.zip")
        try:
            print(f"[data_prep] Downloading from {CORNELL_ZIP_URL} ...")
            urllib.request.urlretrieve(CORNELL_ZIP_URL, zip_path)
        except Exception as e:
            print(f"[data_prep] Error downloading Cornell corpus: {e}", file=sys.stderr)
            return

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                member_name = None
                for name in z.namelist():
                    if name.endswith("movie_lines.txt"):
                        member_name = name
                        break
                if member_name is None:
                    print("[data_prep] movie_lines.txt not found in ZIP.", file=sys.stderr)
                    return
                z.extract(member_name, path=tmpdir)
                extracted_path = os.path.join(tmpdir, member_name)
        except Exception as e:
            print(f"[data_prep] Error extracting ZIP: {e}", file=sys.stderr)
            return

        try:
            with open(extracted_path, "r", encoding="iso-8859-1", errors="ignore") as fin, open(
                output_path, "w", encoding="utf-8"
            ) as fout:
                for line in fin:
                    parts = line.split(" +++$+++ ")
                    if len(parts) == 5:
                        text = parts[-1].strip()
                        if text:
                            fout.write(text.replace("\n", " ") + "\n")
        except Exception as e:
            print(f"[data_prep] Error parsing movie_lines.txt: {e}", file=sys.stderr)
            return

    print(f"[data_prep] Wrote Cornell Movie-Dialogs to {output_path} (manual).")


def prepare_personachat(output_path: str):
    """
    Attempt to download PersonaChat via Hugging Face Datasets.
    Tries "persona_chat" and then "conv_ai_2". Catches any exception.
    """
    if not HAS_DATASETS:
        install_package("datasets")
        global load_dataset
        from datasets import load_dataset
        # Now we have it
    for dataset_key in ["persona_chat", "conv_ai_2"]:
        try:
            print(f"[data_prep] Attempting to load '{dataset_key}' via Hugging Face Datasets...")
            if dataset_key == "conv_ai_2":
                dataset = load_dataset(dataset_key, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_key)
            print(f"[data_prep] Successfully loaded '{dataset_key}'. Writing to {output_path}...")
            with open(output_path, "w", encoding="utf-8") as fout:
                if dataset_key == "persona_chat":
                    for split in ["train", "valid"]:
                        for conv in dataset[split]:
                            for line in conv["dialog"]:
                                text = line.strip()
                                if text:
                                    fout.write(text.replace("\n", " ") + "\n")
                else:  # conv_ai_2
                    for split in ["train", "valid"]:
                        for item in dataset[split]:
                            # conv_ai_2 has a field named "dialog"
                            if "dialog" in item:
                                for line in item["dialog"]:
                                    text = line.strip()
                                    if text:
                                        fout.write(text.replace("\n", " ") + "\n")
                            elif "utterance" in item:
                                text = item["utterance"].strip()
                                if text:
                                    fout.write(text.replace("\n", " ") + "\n")
            print(f"[data_prep] Finished writing PersonaChat ({dataset_key}) to {output_path}.")
            return
        except Exception as e:
            print(f"[data_prep] Failed '{dataset_key}': {e}", file=sys.stderr)
            # Try next key

    print("[data_prep] Could not load PersonaChat under any key. Skipping PersonaChat.", file=sys.stderr)


def main():
    cornell_path = CONV_DIR / "cornell_movie_dialogs.txt"
    persona_path = CONV_DIR / "persona_chat.txt"

    # 1) Prepare Cornell Movie-Dialogs: try ConvoKit, then manual
    if not cornell_path.is_file():
        ok = prepare_cornell_via_convokit(str(cornell_path))
        if not ok:
            prepare_cornell_manual(str(cornell_path))
    else:
        print(f"[data_prep] Skipping Cornell: '{cornell_path}' already exists.")

    # 2) Prepare PersonaChat
    if not persona_path.is_file():
        prepare_personachat(str(persona_path))
    else:
        print(f"[data_prep] Skipping PersonaChat: '{persona_path}' already exists.")

    print("[data_prep] All done. You can now include data/conversational/ in Nora's training.")


if __name__ == "__main__":
    main()
