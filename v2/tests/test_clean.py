import re

import pandas as pd
import pytest

from lyrics_v2 import clean

RAW = """243 Contributors Some Song Lyrics
[Verse 1]
I walk away
And never look back
[Chorus]
Oh yeah, we are dancing
You might also like
Under the neon lights
12Embed"""


def test_join_lines_inserts_space_no_glued_words():
    text = clean.prepare_text(RAW)
    assert "away And" in text
    assert not clean.has_glued_words(text)


def test_v1_bug_is_detected():
    v1_style = "".join(l for l in RAW.split("\n"))  # joined without space
    assert clean.has_glued_words(v1_style)


def test_boilerplate_removed():
    text = clean.prepare_text(RAW)
    for junk in ("Contributors", "[Verse", "Chorus", "You might also like", "Embed"):
        assert junk not in text


def test_no_glued_regex_remnants_in_prepared_text():
    assert not re.search(r"[a-z][A-Z]", clean.prepare_text(RAW))


def test_normalize_title_collapses_variants():
    base = clean.normalize_title("Flowers")
    assert clean.normalize_title("Flowers - Remix") == base
    assert clean.normalize_title("Flowers (feat. X)") == base
    assert clean.normalize_title("FLOWERS") == base


def test_deduplicate_keeps_best_ranked():
    df = pd.DataFrame({
        "market": ["US", "US", "US"], "artist": ["A", "a", "B"],
        "track": ["Song", "Song (Remix)", "Other"], "popularity": [5, 9, 1],
    })
    out = clean.deduplicate(df)
    assert len(out) == 2
    assert out[out.artist.str.lower() == "a"].popularity.iloc[0] == 9


def test_english_detection():
    assert clean.is_english("I walk away and never look back at the neon lights tonight")
    assert not clean.is_english("Bu gece yıldızlar altında seninle dans etmek istiyorum sevgilim")


@pytest.fixture(scope="module")
def nlp():
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load(clean.SPACY_MODEL, disable=["ner", "parser"])
    except OSError:
        pytest.skip("spaCy model not installed")


def test_lemmatize_pos_filter_and_no_stopwords(nlp):
    doc = nlp(clean.prepare_text(RAW))
    tokens = clean.lemmatize(doc)
    assert tokens, "empty document"
    stop = nlp.Defaults.stop_words | clean.LYRIC_STOPWORDS
    assert not (set(tokens) & stop), "stopword leak"
    assert "the" not in tokens and "oh" not in tokens
    assert all(len(t) >= 3 for t in tokens)


def test_clean_frame_drops_empty_and_has_no_glue(nlp):
    long_text = "\n".join(["I walk away and never look back at the empty streets tonight"] * 8)
    df = pd.DataFrame({
        "year": [2023, 2023], "market": ["US", "US"], "artist": ["A", "B"],
        "track": ["One", "Two"], "popularity": [2, 1],
        "lyrics": [long_text, "[Intro]\nOh yeah"],
    })
    out, stats = clean.clean_frame(df, nlp)
    assert len(out) == 1
    assert (out["tokens"].str.split().str.len() > 0).all()
    assert not out["text"].map(clean.has_glued_words).any()
    assert stats["input"] == 2


def test_cli_missing_data_message(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(clean, "LYRICS_DIR", tmp_path)
    assert clean.main(["--market", "US"]) == 2
    assert "No lyrics files" in capsys.readouterr().err

