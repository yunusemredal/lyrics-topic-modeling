"""Topic modeling pipeline: LDA, NMF, Top2Vec, BERTopic (primary) and FASTopic (2024).

    python -m lyrics_v2.model --market US
    python -m lyrics_v2.model --market US GB --models lda nmf top2vec bertopic fastopic

Reads  data/processed/clean_<market>.csv   (from lyrics_v2.clean)
Writes v2/results/models/topics_<market>_<model>.csv       (keywords + labels only)
       v2/results/models/topics_over_time_<market>_bertopic.csv
       v2/results/models/model_summary_<market>.csv
       data/models/doc_topics_<market>.csv                 (git-ignored)
"""
import argparse
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR

ALL_MODELS = ("lda", "nmf", "top2vec", "bertopic", "fastopic")
_STOP_LABEL = {
    "like", "just", "know", "want", "need", "make", "take", "come", "go", "get",
    "say", "tell", "see", "look", "feel", "think", "let", "way", "thing", "time",
    "good", "right", "try", "keep", "give", "put", "leave", "turn", "call",
}


@dataclass
class TopicResult:
    model_name: str
    market: str
    num_topics: int
    topics: dict
    labels: dict
    doc_topics: list
    outliers: int = 0
    runtime_sec: float = 0.0
    topics_over_time: pd.DataFrame | None = field(default=None, repr=False)


def label_from_words_and_docs(words, rep_docs=None, max_terms=3):
    """Build a concise human-readable theme label from top words + representative docs."""
    candidates = []
    for w in words:
        for tok in str(w).lower().split():
            if tok.isalpha() and len(tok) >= 3 and tok not in candidates:
                candidates.append(tok)
    if rep_docs:
        counts = Counter()
        for doc in rep_docs:
            for tok in re.findall(r"[a-z]{3,}", str(doc).lower()):
                if tok not in _STOP_LABEL:
                    counts[tok] += 1
        # Promote frequent non-generic terms from representative docs if top words are generic
        top_non_generic = [w for w in candidates if w not in _STOP_LABEL]
        for tok, _ in counts.most_common(5):
            if tok not in top_non_generic and tok not in candidates:
                candidates.append(tok)
    preferred = [w for w in candidates if w not in _STOP_LABEL]
    chosen = (preferred + [w for w in candidates if w not in preferred])[:max_terms]
    return " / ".join(chosen) if chosen else "misc"


def _gensim_corpus(tokenized_docs):
    from gensim import corpora

    id2word = corpora.Dictionary(tokenized_docs)
    corpus = [id2word.doc2bow(doc) for doc in tokenized_docs]
    return id2word, corpus


def fit_lda(tokenized_docs, market, num_topics=15, passes=50, random_state=100, top_n=10):
    from gensim.models import LdaModel

    t0 = time.perf_counter()
    id2word, corpus = _gensim_corpus(tokenized_docs)
    model = LdaModel(
        corpus=corpus, id2word=id2word, num_topics=num_topics,
        random_state=random_state, passes=passes,
    )
    topics = {
        tid: [w for w, _ in model.show_topic(tid, topn=top_n)]
        for tid in range(num_topics)
    }
    doc_topics = [
        max(model.get_document_topics(bow), key=lambda x: x[1])[0] if bow else -1
        for bow in corpus
    ]
    labels = {tid: label_from_words_and_docs(words) for tid, words in topics.items()}
    return TopicResult(
        "lda", market, num_topics, topics, labels, doc_topics,
        outliers=doc_topics.count(-1), runtime_sec=round(time.perf_counter() - t0, 2),
    )


def fit_nmf(tokenized_docs, market, num_topics=15, passes=50, random_state=100, top_n=10):
    from gensim.models import Nmf

    t0 = time.perf_counter()
    id2word, corpus = _gensim_corpus(tokenized_docs)
    model = Nmf(
        corpus=corpus, id2word=id2word, num_topics=num_topics,
        random_state=random_state, passes=passes,
    )
    topics = {
        tid: [w for w, _ in model.show_topic(tid, topn=top_n)]
        for tid in range(num_topics)
    }
    doc_topics = []
    for bow in corpus:
        dist = model.get_document_topics(bow)
        doc_topics.append(max(dist, key=lambda x: x[1])[0] if dist else -1)
    labels = {tid: label_from_words_and_docs(words) for tid, words in topics.items()}
    return TopicResult(
        "nmf", market, num_topics, topics, labels, doc_topics,
        outliers=doc_topics.count(-1), runtime_sec=round(time.perf_counter() - t0, 2),
    )


def fit_top2vec(docs, market, min_count=5, speed="learn", workers=4, top_n=10):
    from top2vec import Top2Vec

    t0 = time.perf_counter()
    model = None
    for mc in (min_count, 2, 1):
        try:
            model = Top2Vec(
                docs, embedding_model="doc2vec", min_count=mc,
                speed=speed, workers=workers, verbose=False,
            )
            break
        except ValueError:
            if mc == 1:
                raise
    topic_words, _, topic_nums = model.get_topics()
    topics = {int(tid): list(words[:top_n]) for tid, words in zip(topic_nums, topic_words)}
    doc_topics = [int(t) for t in model.doc_top]
    labels = {tid: label_from_words_and_docs(words) for tid, words in topics.items()}
    return TopicResult(
        "top2vec", market, len(topics), topics, labels, doc_topics,
        outliers=0, runtime_sec=round(time.perf_counter() - t0, 2),
    )


def fit_bertopic(
    docs, market, timestamps=None, min_topic_size=8,
    embedding_model="all-MiniLM-L6-v2", nr_topics="auto", top_n=10,
):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    t0 = time.perf_counter()
    n_neighbors = max(2, min(15, len(docs) - 1))
    n_components = max(2, min(5, len(docs) - 2))
    umap_model = UMAP(
        n_neighbors=n_neighbors, n_components=n_components,
        min_dist=0.0, metric="cosine", random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min(min_topic_size, max(2, len(docs) // 4)),
        metric="euclidean", cluster_selection_method="eom", prediction_data=True,
    )
    vec_model = CountVectorizer(stop_words="english", min_df=2 if len(docs) >= 20 else 1)
    model = BERTopic(
        language="english",
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vec_model,
        nr_topics=nr_topics,
        top_n_words=top_n,
        verbose=False,
    )
    raw_topics, _ = model.fit_transform(docs)
    doc_topics = [int(t) for t in raw_topics]
    if -1 in doc_topics and len(set(doc_topics) - {-1}) >= 1:
        try:
            reduced = model.reduce_outliers(docs, doc_topics, strategy="c-tf-idf")
            model.update_topics(docs, topics=reduced, vectorizer_model=vec_model)
            doc_topics = [int(t) for t in reduced]
        except Exception:  # noqa: BLE001 - keep original assignments on tiny corpora
            pass

    topics, labels = {}, {}
    for tid in sorted(set(doc_topics)):
        if tid == -1:
            continue
        words = [w for w, _ in (model.get_topic(tid) or [])[:top_n]]
        rep_docs = model.get_representative_docs(tid)
        topics[tid] = words
        labels[tid] = label_from_words_and_docs(words, rep_docs)

    tot_df = None
    if timestamps is not None and len(topics) > 0:
        tot_df = model.topics_over_time(docs, [int(y) for y in timestamps], topics=doc_topics)
        tot_df["Label"] = tot_df["Topic"].map(lambda t: labels.get(int(t), f"Topic {t}"))

    return TopicResult(
        "bertopic", market, len(topics), topics, labels, doc_topics,
        outliers=doc_topics.count(-1), runtime_sec=round(time.perf_counter() - t0, 2),
        topics_over_time=tot_df,
    )


def fit_fastopic(docs, market, num_topics=15, epochs=150, doc_embed_model="all-MiniLM-L6-v2", top_n=10):
    from fastopic import FASTopic

    t0 = time.perf_counter()
    model = FASTopic(
        num_topics=num_topics, num_top_words=top_n,
        doc_embed_model=doc_embed_model, verbose=False,
    )
    top_words, theta = model.fit_transform(docs, epochs=epochs)
    topics = {}
    for tid, entry in enumerate(top_words):
        words = entry.split()[:top_n] if isinstance(entry, str) else list(entry)[:top_n]
        topics[tid] = words
    doc_topics = [int(x) for x in np.asarray(theta).argmax(axis=1)]
    labels = {tid: label_from_words_and_docs(words) for tid, words in topics.items()}
    return TopicResult(
        "fastopic", market, len(topics), topics, labels, doc_topics,
        outliers=0, runtime_sec=round(time.perf_counter() - t0, 2),
    )


def save_results(results, df, market, out_dir=None, models_dir=None):
    """Save non-copyrighted topic tables under v2/results/models and assignments under data/models."""
    out_dir = Path(out_dir or (RESULTS_DIR / "models"))
    models_dir = Path(models_dir or MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    assign_df = df[["year", "market", "artist", "track"]].copy()
    summary_rows = []
    for res in results:
        counts = Counter(res.doc_topics)
        rows = [
            {
                "topic_id": tid,
                "label": res.labels.get(tid, f"Topic {tid}"),
                "doc_count": counts.get(tid, 0),
                "top_words": ", ".join(words),
            }
            for tid, words in sorted(res.topics.items())
        ]
        pd.DataFrame(rows).to_csv(out_dir / f"topics_{market}_{res.model_name}.csv", index=False)
        if res.topics_over_time is not None:
            res.topics_over_time.to_csv(
                out_dir / f"topics_over_time_{market}_{res.model_name}.csv", index=False
            )
        assign_df[f"topic_{res.model_name}"] = res.doc_topics
        summary_rows.append({
            "model": res.model_name,
            "market": market,
            "num_topics": res.num_topics,
            "outliers": res.outliers,
            "docs": len(res.doc_topics),
            "runtime_sec": res.runtime_sec,
        })

    summary_path = out_dir / f"model_summary_{market}.csv"
    new_summary = pd.DataFrame(summary_rows)
    if summary_path.exists():
        old_summary = pd.read_csv(summary_path)
        old_summary = old_summary[~old_summary["model"].isin(new_summary["model"])]
        new_summary = pd.concat([old_summary, new_summary], ignore_index=True)
    new_summary.to_csv(summary_path, index=False)

    assign_path = models_dir / f"doc_topics_{market}.csv"
    if assign_path.exists():
        old_assign = pd.read_csv(assign_path)
        if len(old_assign) == len(assign_df):
            for col in assign_df.columns:
                if col.startswith("topic_"):
                    old_assign[col] = assign_df[col]
            assign_df = old_assign
    assign_df.to_csv(assign_path, index=False)
    return summary_path


def run_market(market, models=ALL_MODELS, num_topics=15, processed_dir=None):
    pdir = Path(processed_dir or PROCESSED_DIR)
    path = pdir / f"clean_{market.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data missing: {path}. Run `python -m lyrics_v2.clean --market {market}` first."
        )
    df = pd.read_csv(path).dropna(subset=["tokens"]).reset_index(drop=True)
    docs = df["tokens"].astype(str).tolist()
    tokenized = [d.split() for d in docs]
    years = df["year"].tolist()

    results = []
    for name in models:
        if name == "lda":
            results.append(fit_lda(tokenized, market, num_topics=num_topics))
        elif name == "nmf":
            results.append(fit_nmf(tokenized, market, num_topics=num_topics))
        elif name == "top2vec":
            results.append(fit_top2vec(docs, market))
        elif name == "bertopic":
            results.append(fit_bertopic(docs, market, timestamps=years))
        elif name == "fastopic":
            results.append(fit_fastopic(docs, market, num_topics=num_topics))
        else:
            raise ValueError(f"Unknown model: {name}")
    save_results(results, df, market)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit 5 topic models on cleaned lyrics")
    ap.add_argument("--market", nargs="+", default=["US"], help="US, GB, ...")
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=list(ALL_MODELS))
    ap.add_argument("--num-topics", type=int, default=15, help="Topics for LDA/NMF/FASTopic")
    args = ap.parse_args(argv)

    for m in args.market:
        try:
            res = run_market(m.upper(), models=args.models, num_topics=args.num_topics)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 2
        for r in res:
            print(f"[{m.upper()}/{r.model_name}] topics={r.num_topics} outliers={r.outliers} ({r.runtime_sec}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
