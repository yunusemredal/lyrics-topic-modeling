import pandas as pd

from lyrics_v2 import evaluate


def test_compute_diversity_bounds():
    assert evaluate.compute_diversity([["a", "b", "c"], ["d", "e", "f"]], top_n=3) == 1.0
    assert evaluate.compute_diversity([["a", "b"], ["a", "b"]], top_n=2) == 0.5
    assert evaluate.compute_diversity([]) == 0.0


def test_compute_coherence_returns_float():
    docs = [
        "love heart romance kiss".split(),
        "love heart dream night".split(),
        "money car gold chain".split(),
        "money car flex cash".split(),
    ] * 4
    c_v = evaluate.compute_coherence([["love", "heart", "kiss"], ["money", "car", "gold"]], docs)
    assert isinstance(c_v, float) and c_v > 0.0


def test_evaluate_market_reads_topics_and_writes_metrics(tmp_path):
    pdir, mdir, edir = tmp_path / "proc", tmp_path / "models", tmp_path / "eval"
    pdir.mkdir()
    mdir.mkdir()
    pd.DataFrame({"tokens": ["love heart kiss night", "money car gold cash"] * 6}).to_csv(
        pdir / "clean_US.csv", index=False
    )
    pd.DataFrame({
        "topic_id": [0, 1],
        "label": ["love / heart", "money / car"],
        "doc_count": [6, 6],
        "top_words": ["love, heart, kiss", "money, car, gold"],
    }).to_csv(mdir / "topics_US_lda.csv", index=False)
    pd.DataFrame({"model": ["lda"], "outliers": [0]}).to_csv(
        mdir / "model_summary_US.csv", index=False
    )

    out = evaluate.evaluate_market("US", models=("lda",), processed_dir=pdir, models_dir=mdir, eval_dir=edir)
    assert len(out) == 1
    assert out["c_v"].iloc[0] > 0.0
    assert out["topic_diversity"].iloc[0] == 1.0
    assert (edir / "metrics_US.csv").exists()


def test_scan_k_produces_csv_and_plot(tmp_path):
    pdir, edir = tmp_path / "proc", tmp_path / "eval"
    pdir.mkdir()
    pd.DataFrame({"tokens": ["love heart kiss night", "money car gold cash"] * 6}).to_csv(
        pdir / "clean_US.csv", index=False
    )
    df = evaluate.scan_k("US", k_values=[2, 3], passes=3, processed_dir=pdir, eval_dir=edir)
    assert set(df["model"]) == {"lda", "nmf"}
    assert (edir / "k_scan_US.csv").exists()
    assert (edir / "k_scan_US.png").exists()


def test_cli_missing_data_returns_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "PROCESSED_DIR", tmp_path)
    assert evaluate.main(["--market", "ZZ"]) == 2
    assert "Cleaned data missing" in capsys.readouterr().err
