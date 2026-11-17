"""Cleaning pipeline for lyrics (fixes the v1 "awayand" glued-word bug).

    python -m lyrics_v2.clean --market US

Reads  data/lyrics/lyrics_<market>_<year>.csv   (from lyrics_v2.collect)
Writes data/processed/clean_<market>.csv         (git-ignored)

Steps: strip Genius boilerplate -> join lines WITH a space -> drop non-English
-> de-duplicate (remix / re-entry versions) -> spaCy lemmatization with the
same POS filter as v1 (NOUN/ADJ/VERB/ADV) -> drop stopwords and empty docs.
"""
import argparse
import re
import sys

from .paths import DATA_DIR, LYRICS_DIR

PROCESSED_DIR = DATA_DIR / "processed"
KEEP_POS = {"NOUN", "ADJ", "VERB", "ADV"}
SPACY_MODEL = "en_core_web_sm"

# Vocal fillers / contractions remnants that carry no topic information.
LYRIC_STOPWORDS = {
    "oh", "ooh", "yeah", "yea", "uh", "huh", "hey", "ah", "aah", "la", "na", "woah", "whoa",
    "mm", "mmm", "ayy", "ay", "yo", "gonna", "wanna", "gotta", "ima", "nah", "baby",
}

_HEADER = re.compile(r"^.*?\bLyrics\b", re.S)  # "123 Contributors... Lyrics" preamble
_EMBED = re.compile(r"\d*\s*Embed\s*$")
_SECTION = re.compile(r"\[[^\]]*\]")
_YOU_MIGHT = re.compile(r"You might also like")
_GLUED = re.compile(r"[a-z][A-Z]")
_TITLE_NOISE = re.compile(r"\s*[\(\[].*?[\)\]]|\s+-\s+.*$|\s+(feat|ft)\.?\s+.*$", re.I)


def join_lines(raw):
    """Join lyric lines with a single space (v1 joined without one -> 'awayand')."""
    return " ".join(part.strip() for part in re.split(r"[\r\n]+", raw) if part.strip())


def strip_boilerplate(raw):
    """Remove Genius preamble/trailer, section headers and ad strings."""
    text = raw
    head = text[:400]
    if "Lyrics" in head and "Contributor" in head:
        text = _HEADER.sub("", text, count=1)
    text = _SECTION.sub(" ", text)
    text = _YOU_MIGHT.sub(" ", text)
    text = _EMBED.sub("", text.strip())
    return text


def prepare_text(raw):
    """Raw Genius text -> single-line plain text (case preserved for glue checks)."""
    text = join_lines(strip_boilerplate(raw))
    return re.sub(r"\s+", " ", text).strip()


def has_glued_words(text):
    """True if text has lower->Upper transitions inside a token (e.g. 'awayAnd')."""
    return any(_GLUED.search(tok) for tok in text.split())


def normalize_title(title):
    """Canonical key for a song title: no remix/feat/version tags, lowercase."""
    t = _TITLE_NOISE.sub("", str(title))
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def is_english(text):
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    try:
        return detect(text) == "en"
    except Exception:  # noqa: BLE001 - too short / no features -> not usable
        return False


def lemmatize(doc_or_text, nlp=None):
    """Lemmas of NOUN/ADJ/VERB/ADV tokens, minus stopwords; returns a list."""
    doc = nlp(doc_or_text) if nlp is not None else doc_or_text
    out = []
    for tok in doc:
        if tok.pos_ not in KEEP_POS or tok.is_stop or not tok.is_alpha:
            continue
        lemma = tok.lemma_.lower()
        if len(lemma) < 3 or lemma in LYRIC_STOPWORDS:
            continue
        out.append(lemma)
    return out


def deduplicate(df):
    """Keep one row per (market, artist, normalized title); best rank/popularity wins."""
    df = df.copy()
    df["_key"] = df["track"].map(normalize_title)
    df["_artist"] = df["artist"].str.lower().str.strip()
    df["_pop"] = df["popularity"].astype(float)
    df = df.sort_values("_pop", ascending=False)
    df = df.drop_duplicates(subset=["market", "_artist", "_key"], keep="first")
    return df.drop(columns=["_key", "_artist", "_pop"]).sort_index()


def clean_frame(df, nlp, min_tokens=20):
    """Full cleaning of a lyrics DataFrame; returns (clean_df, stats dict)."""
    stats = {"input": len(df)}
    df = df.dropna(subset=["lyrics"]).copy()
    df["text"] = df["lyrics"].map(prepare_text)
    df = df[df["text"].str.len() > 0]
    stats["after_boilerplate"] = len(df)
    df = df[df["text"].map(is_english)]
    stats["after_language"] = len(df)
    df = deduplicate(df)
    stats["after_dedup"] = len(df)
    docs = nlp.pipe(df["text"].tolist(), batch_size=32)
    df["tokens"] = [" ".join(lemmatize(d)) for d in docs]
    df = df[df["tokens"].str.split().str.len() >= min_tokens]
    stats["after_min_tokens"] = len(df)
    return df.drop(columns=["lyrics"]), stats


def load_spacy():
    import spacy

    try:
        return spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    except OSError as exc:
        raise SystemExit(
            f"spaCy model {SPACY_MODEL} missing in this venv. "
            "Run: .venv\\Scripts\\python -m pip install -r v2/requirements.txt (see MK-004)."
        ) from exc


def main(argv=None):
    import pandas as pd

    ap = argparse.ArgumentParser(description="Clean lyrics for topic modeling")
    ap.add_argument("--market", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--tag", help="Output slice name (defaults to --market, e.g. US_2016_2022)")
    args = ap.parse_args(argv)

    files = [LYRICS_DIR / f"lyrics_{args.market.upper()}_{y}.csv" for y in args.years]
    files = [f for f in files if f.exists()]
    if not files:
        print(f"No lyrics files for {args.market} under {LYRICS_DIR}. "
              "Run `python -m lyrics_v2.collect` first.", file=sys.stderr)
        return 2
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    clean, stats = clean_frame(df, load_spacy())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tag = (args.tag or args.market).upper()
    out = PROCESSED_DIR / f"clean_{tag}.csv"
    clean.to_csv(out, index=False)
    print(f"{stats} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
