# -*- coding: utf-8 -*-
"""
data.py - shared helper that loads the WikiText-2 text
=======================================================
Purpose: return the evaluation text used by the measurement scripts.

Why a separate file: Hugging Face renamed its dataset repos. "wikitext" used
to work on its own, but newer huggingface_hub versions require the
"namespace/name" form and the old call raises HfUriError. The candidates
below are tried in order.

There is NO SILENT FALLBACK. If none of them works, the script stops.

Inputs : a local .txt file (optional), otherwise the WikiText-2 test split
         from the Hugging Face hub.
Outputs: the text as a string (nothing is written to disk).
Run    : python data.py   (connection test; prints basic text statistics)
         In other scripts: from data import wikitext
"""
import os

CANDIDATES = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("mindchain/wikitext2",  None),
    ("wikitext",             "wikitext-2-raw-v1"),   # for older versions
]


def wikitext(path=None, min_chars=100_000):
    """Return the text. If path is given (e.g. a Turkish text), read that file instead."""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        print(f"text : {path} ({len(text):,} characters)")
        return _validate(text, min_chars)

    from datasets import load_dataset
    errors = []
    for repo, config in CANDIDATES:
        try:
            ds = (load_dataset(repo, config, split="test")
                  if config else load_dataset(repo, split="test"))
            text = "\n\n".join(ds["text"])
            print(f"text : {repo} ({len(text):,} characters)")
            return _validate(text, min_chars)
        except Exception as e:
            errors.append(f"  {repo}: {type(e).__name__}: {str(e)[:160]}")

    raise RuntimeError(
        "Could not open any data source. Tried:\n" + "\n".join(errors) +
        "\n\nPossible fixes:\n"
        "  1) pip install -U datasets huggingface_hub\n"
        "  2) Download the wikitext-2-raw test file from\n"
        "     https://huggingface.co/datasets/Salesforce/wikitext, save it as\n"
        "     data/text.txt, then point the script's TEXT_FILE setting to it.")


def _validate(text, min_chars):
    if len(text) < min_chars:
        raise RuntimeError(
            f"Text is too short ({len(text):,} characters, at least {min_chars:,} required). "
            "The data did not download correctly; stopping.")
    words = text.split()
    if len(set(words[:20_000])) < 2_000:
        raise RuntimeError(
            "Text looks too repetitive - it may be the wrong or a placeholder dataset; "
            "stopping.")
    return text


if __name__ == "__main__":
    print("data connection test\n" + "-"*40)
    text = wikitext()
    words = text.split()
    print(f"characters    : {len(text):,}")
    print(f"words         : {len(words):,}")
    print(f"distinct words: {len(set(words[:20_000])):,} (within the first 20,000)")
    print("\nfirst 300 characters:\n" + text[:300].replace("\n", " "))
    print("\n" + "-"*40 + "\nOK - data looks fine. You can now run 01_union.py.")
