"""Experiment Runner for Phase 4: Retrieval Experiments & Ablation Studies.

This script automates running the 60‑question golden set across multiple retrieval configurations
and stores a JSON report for each experiment under ``reports/``.

Configuration matrix (matching the design spec):

| ID | Chunking | Retriever | Reranker |
|----|----------|-----------|----------|
| EXP-01 | fixed | dense (Chroma) | No |
| EXP-02 | recursive | dense (Chroma) | No |
| EXP-03 | clause_aware | dense (Chroma) | No |
| EXP-04 | clause_aware | hybrid (BM25 + dense) | No |
| EXP-05 | clause_aware | hybrid (BM25 + dense) | Yes |
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any
import sys

# Ensure the repository root is on sys.path for package imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eval.eval_runner import EvaluationRunner


# Standalone Okapi BM25 implementation for zero-dependency hybrid retrieval
class SimpleBM25:
    """Zero-dependency Okapi BM25 implementation."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(x) for x in corpus) / max(1, self.corpus_size)
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = [len(x) for x in corpus]

        nd = {}
        for document in corpus:
            frequencies = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1

        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for q in query:
            if q not in self.idf:
                continue
            idf = self.idf[q]
            for i, doc_freq in enumerate(self.doc_freqs):
                if q in doc_freq:
                    freq = doc_freq[q]
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avgdl))
                    scores[i] += idf * (numerator / denominator)
        return scores


try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = SimpleBM25

# Optional cross‑encoder for reranking
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

_encoder_cache = {}


def build_bm25_index(vector_store) -> Dict[str, Any]:
    """Create a BM25 index over all chunk texts in the collection.

    Returns a dict with ``'bm25'`` (BM25 object), ``'ids'`` and ``'docs'``.
    """
    collection_data = vector_store.collection.get(include=["documents"])
    docs = collection_data.get("documents", [])
    ids = collection_data.get("ids", [])
    if not docs:
        print("[BM25] Warning: collection has no documents to index.")
        return {"bm25": BM25Okapi([[]]), "ids": [], "docs": []}
    tokenised = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenised)
    return {"bm25": bm25, "ids": ids, "docs": docs}


def hybrid_retrieval(query: str, vector_store, bm25_index, top_k: int = 5, strategy_filter: str = None) -> List[Dict[str, Any]]:
    """Combine dense and BM25 scores using Reciprocal Rank Fusion (RRF)."""
    # 1️⃣ Dense retrieval
    dense_hits = vector_store.query(query, top_k=100, strategy_filter=strategy_filter)
    dense_rank = {hit["chunk_id"]: rank + 1 for rank, hit in enumerate(dense_hits)}

    # 2️⃣ BM25 retrieval
    bm25 = bm25_index["bm25"]
    tokenised_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenised_query)
    bm25_rank = {cid: rank + 1 for rank, (score, cid) in enumerate(
        sorted(zip(bm25_scores, bm25_index["ids"]), key=lambda x: x[0], reverse=True)
    )}

    # 3️⃣ RRF fusion (k=60 as per standard practice)
    k = 60
    fused: Dict[str, float] = {}
    for cid, r in dense_rank.items():
        fused[cid] = fused.get(cid, 0) + 1 / (k + r)
    for cid, r in bm25_rank.items():
        fused[cid] = fused.get(cid, 0) + 1 / (k + r)

    # 4️⃣ Sort by fused score and retrieve full metadata from Chroma
    sorted_cids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    final_hits: List[Dict[str, Any]] = []
    # Theoretical maximum RRF score for k=60 when chunk is rank 1 in both dense and BM25:
    max_rrf = 2.0 / (k + 1.0)  # = 2 / 61 ≈ 0.032786885

    for cid, score in sorted_cids:
        entry = vector_store.collection.get(ids=[cid], include=["documents", "metadatas"])
        docs = entry.get("documents", [])
        metas = entry.get("metadatas", [])
        if docs and metas:
            # Normalize RRF score to [0, 1] range to align with cosine similarity
            norm_score = min(1.0, max(0.0, score / max_rrf))
            final_hits.append({
                "chunk_id": cid,
                "text": docs[0],
                "metadata": metas[0],
                "similarity_score": round(float(norm_score), 4),
                "raw_rrf_score": round(float(score), 6),
                "distance": 0.0,
            })
    return final_hits


def rerank_with_cross_encoder(query: str, hits: List[Dict[str, Any]], model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2") -> List[Dict[str, Any]]:
    """Re‑rank ``hits`` using a cross‑encoder model."""
    if not hits:
        return hits
    global _encoder_cache
    encoder = _encoder_cache.get(model_name)
    if encoder is None and CrossEncoder is not None:
        try:
            print(f"[Reranker] Loading CrossEncoder '{model_name}'...")
            encoder = CrossEncoder(model_name)
            _encoder_cache[model_name] = encoder
        except Exception as e:
            print(f"[Reranker] Warning: Failed to load '{model_name}': {e}. Continuing with RRF ranks.")
            encoder = None

    if encoder is not None:
        try:
            pairs = [(query, hit["text"]) for hit in hits]
            scores = encoder.predict(pairs)
            for hit, sc in zip(hits, scores):
                logit = float(sc)
                # Map cross-encoder logits (-inf, +inf) to [0, 1] via sigmoid
                bounded_logit = max(-50.0, min(50.0, logit))
                sig_score = 1.0 / (1.0 + math.exp(-bounded_logit))
                hit["similarity_score"] = round(sig_score, 4)
                hit["raw_logit"] = round(logit, 4)
            return sorted(hits, key=lambda x: x["similarity_score"], reverse=True)
        except Exception as e:
            print(f"[Reranker] Warning during scoring: {e}. Preserving RRF ranks.")
            return hits
    return hits


def run_all_experiments():
    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    matrix = [
        {"id": "exp_01_baseline", "strategy": "fixed", "collection": "rbi_fixed", "hybrid": False, "rerank": False},
        {"id": "exp_02_recursive", "strategy": "recursive", "collection": "rbi_recursive", "hybrid": False, "rerank": False},
        {"id": "exp_03_clause_aware", "strategy": "clause_aware", "collection": "rbi_clause_aware", "hybrid": False, "rerank": False},
        {"id": "exp_04_hybrid", "strategy": "clause_aware", "collection": "rbi_clause_aware", "hybrid": True, "rerank": False},
        {"id": "exp_05_hybrid_rerank", "strategy": "clause_aware", "collection": "rbi_clause_aware", "hybrid": True, "rerank": True},
    ]

    master_rows = []
    for cfg in matrix:
        print(f"\n=== Running {cfg['id']} (strategy={cfg['strategy']}, collection={cfg['collection']}, hybrid={cfg['hybrid']}, rerank={cfg['rerank']}) ===\n")

        # Instantiate runner pointed at the experiment's target collection
        runner = EvaluationRunner(collection_name=cfg["collection"])

        # Prepare BM25 index on the current collection if hybrid is requested
        bm25_index = build_bm25_index(runner.vector_store) if cfg["hybrid"] else None

        # Build query routing function
        if cfg["hybrid"]:
            def query_fn(question: str, top_k: int = 5, _runner=runner, _bm25=bm25_index, _rerank=cfg["rerank"]):
                hits = hybrid_retrieval(question, _runner.vector_store, _bm25, top_k=top_k)
                if _rerank:
                    hits = rerank_with_cross_encoder(question, hits)
                return hits
        else:
            def query_fn(question: str, top_k: int = 5, _runner=runner):
                return _runner.vector_store.query(question, top_k=top_k)

        report_path = reports_dir / f"{cfg['id']}.json"
        runner.run_evaluation(
            output_report_path=str(report_path),
            query_function=query_fn,
        )

        # Load aggregate metrics for master table
        with open(report_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        agg_ret = payload.get("retrieval_metrics", {})
        agg_gen = payload.get("generation_metrics", {})
        agg_lat = payload.get("aggregate_latency", {})
        tok_usage = payload.get("token_usage", {})

        in_tok = tok_usage.get("input_tokens", 0)
        out_tok = tok_usage.get("output_tokens", 0)
        cost_usd = (in_tok / 1e6) * 0.75 + (out_tok / 1e6) * 3.75
        cost_inr = cost_usd * 85.0

        p95_lat = agg_lat.get("p95_s", "—")
        lat_str = f"{p95_lat:.2f}s" if isinstance(p95_lat, (int, float)) else str(p95_lat)
        cost_str = f"₹{cost_inr:.2f}" if (in_tok or out_tok) else "—"

        master_rows.append({
            "id": cfg["id"],
            "hit@3": agg_ret.get("mean_hit_at_3", 0),
            "mrr": agg_ret.get("mrr", 0),
            "faithfulness": agg_gen.get("mean_faithfulness", 0),
            "latency_s": lat_str,
            "cost_₹": cost_str,
        })

    # Write master markdown
    md_lines = [
        "# Master Results Table (Phase 4)",
        "",
        "| Experiment | Hit@3 | MRR | Faithfulness | Latency (s) | Cost per 60 Queries (₹) |",
        "|------------|------|-----|--------------|------------|--------------------------|",
    ]
    for row in master_rows:
        md_lines.append(
            f"| {row['id']} | {row['hit@3']:.2f} | {row['mrr']:.4f} | {row['faithfulness']:.2f} | {row['latency_s']} | {row['cost_₹']} |"
        )
    master_md = repo_root / "master_results.md"
    master_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nMaster results written to {master_md}\n")


if __name__ == "__main__":
    run_all_experiments()
