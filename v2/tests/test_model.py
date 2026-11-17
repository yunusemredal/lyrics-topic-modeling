import pandas as pd

from lyrics_v2 import model


def test_label_from_words_and_docs_skips_generic_terms():
    lbl = model.label_from_words_and_docs(
        ["know", "love", "night", "heart"],
        rep_docs=["dancing under neon midnight sky", "neon heart break"],
    )
    assert "know" not in lbl
    assert "love" in lbl and "night" in lbl


def test_fit_lda_and_nmf_on_synthetic_corpus():
    docs = [
        "love heart romance kiss forever embrace".split(),
        "money car diamond gold chain flex".split(),
        "party night club dance music drink".split(),
    ] * 8
    lda = model.fit_lda(docs, "US", num_topics=3, passes=5, top_n=5)
    nmf = model.fit_nmf(docs, "US", num_topics=3, passes=5, top_n=5)
    assert lda.num_topics == 3 and len(lda.topics) == 3
    assert nmf.num_topics == 3 and len(nmf.topics) == 3
    assert len(lda.doc_topics) == len(docs)
    assert all(len(w) == 5 for w in lda.topics.values())


def test_save_results_excludes_lyrics_text(tmp_path):
    res = model.TopicResult(
        "lda", "US", 2,
        topics={0: ["love", "heart"], 1: ["money", "gold"]},
        labels={0: "love / heart", 1: "money / gold"},
        doc_topics=[0, 1],
        topics_over_time=pd.DataFrame({"Topic": [0], "Words": ["love"], "Frequency": [1], "Timestamp": [2023], "Label": ["love / heart"]}),
    )
    df = pd.DataFrame({"year": [2023, 2024], "market": ["US", "US"], "artist": ["A", "B"], "track": ["T1", "T2"]})
    out_dir, m_dir = tmp_path / "results", tmp_path / "models"
    summary_path = model.save_results([res], df, "US", out_dir=out_dir, models_dir=m_dir)
    assert summary_path.exists()
    topics_csv = (out_dir / "topics_US_lda.csv").read_text(encoding="utf-8")
    assert "love / heart" in topics_csv and "lyrics" not in topics_csv
    assert (out_dir / "topics_over_time_US_bertopic.csv").exists() is False
    assert (out_dir / "topics_over_time_US_lda.csv").exists()
    assert (m_dir / "doc_topics_US.csv").exists()


def test_cli_missing_processed_data_returns_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(model, "PROCESSED_DIR", tmp_path)
    assert model.main(["--market", "ZZ", "--models", "lda"]) == 2
    assert "Cleaned data missing" in capsys.readouterr().err
