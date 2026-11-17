import pandas as pd

from lyrics_v2 import compare


def test_compare_word_rates_detects_rising_and_falling():
    a = pd.DataFrame({"tokens": ["love love night heart"] * 5})
    b = pd.DataFrame({"tokens": ["money flex night car"] * 5})
    out = compare.compare_word_rates(a, b, min_rate=1.0)
    assert out.iloc[0]["diff_per_1k"] > 0
    assert out.iloc[-1]["diff_per_1k"] < 0
    assert "love" in set(out["word"]) and "money" in set(out["word"])


def test_compute_genre_proxy_shares_bounded():
    df = pd.DataFrame({
        "year": [2023, 2023, 2024],
        "tokens": ["fuck money flex", "love heart dream", "sun sky walk"],
    })
    out = compare.compute_genre_proxy(df, "US")
    assert len(out) == 2
    assert out[out["year"] == 2023]["hiphop_proxy_share"].iloc[0] == 0.5
    assert out[out["year"] == 2023]["romance_proxy_share"].iloc[0] == 0.5


def test_run_compare_missing_files_returns_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, "PROCESSED_DIR", tmp_path)
    assert compare.main([]) == 2
    assert "Missing cleaned files" in capsys.readouterr().err


def test_run_compare_writes_all_outputs(tmp_path):
    pdir, rdir = tmp_path / "proc", tmp_path / "results"
    pdir.mkdir()
    (rdir / "eval").mkdir(parents=True)
    (rdir / "models").mkdir(parents=True)

    pd.DataFrame({"year": [2020, 2021], "tokens": ["love heart night", "money flex car"]}).to_csv(
        pdir / "clean_US_2016_2022.csv", index=False
    )
    pd.DataFrame({"year": [2023, 2024], "tokens": ["fuck bitch night", "love dream sky"]}).to_csv(
        pdir / "clean_US.csv", index=False
    )
    pd.DataFrame({"year": [2023, 2024], "tokens": ["love heart away", "night sound light"]}).to_csv(
        pdir / "clean_GB.csv", index=False
    )
    for k in ("US_2016_2022", "US", "GB"):
        pd.DataFrame({
            "model": ["lda", "fastopic"], "market": [k, k],
            "num_topics": [15, 15], "outliers": [0, 0],
            "c_v": [0.35, 0.45], "topic_diversity": [0.5, 1.0],
        }).to_csv(rdir / "eval" / f"metrics_{k}.csv", index=False)
    for name in ("topics_over_time_US_2016_2022_bertopic.csv", "topics_over_time_US_bertopic.csv"):
        pd.DataFrame({
            "Topic": [0], "Words": ["love"], "Frequency": [5], "Timestamp": [2021], "Label": ["love / heart"]
        }).to_csv(rdir / "models" / name, index=False)

    out = compare.run_compare(processed_dir=pdir, results_dir=rdir)
    for fname in (
        "period_comparison_US.csv", "country_comparison_2023_2025.csv",
        "genre_proxy_summary.csv", "all_metrics_summary.csv",
        "topics_over_time_US_2016_2025.png", "model_comparison.png",
        "wordcloud_US_2016_2022.png", "wordcloud_US_2023_2025.png", "wordcloud_GB_2023_2025.png",
    ):
        assert (out / fname).exists(), fname
