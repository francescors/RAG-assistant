"""Évalue l'assistant sur eval/questions.json.

Métriques de recherche (gratuites, sans LLM) :
  - hit@k      : la bonne source est-elle dans les k passages retrouvés ?
  - MRR        : rang moyen (inversé) de la bonne source
  - abstention : pour les questions hors sujet, on ne retrouve rien

Avec --with-llm (et ANTHROPIC_API_KEY) :
  - exactitude : la réponse contient-elle les mots-clés attendus ?
  - refus      : le LLM répond-il « je ne sais pas » aux questions hors sujet ?

Usage :
  python -m eval.run_eval                       # retriever défini par RETRIEVER
  python -m eval.run_eval --retriever embeddings
  python -m eval.run_eval --compare             # tfidf vs embeddings (tableau)
  python -m eval.run_eval --retriever embeddings --sweep   # aide à choisir le seuil
"""

import argparse
import json
import sys
from pathlib import Path

from app import config
from app.ingest import load_chunks
from app.llm import NO_ANSWER, generate_answer
from app.retriever import build_retriever

QUESTIONS = Path(__file__).parent / "questions.json"


def evaluate(retriever, questions: list[dict], k: int, min_score: float | None = None) -> dict:
    """Calcule hit@k, MRR et abstention. Retourne aussi le détail par question."""
    hits_ok, rr_sum, answerable, abstained, unanswerable = 0, 0.0, 0, 0, 0
    details = []
    for q in questions:
        hits = retriever.search(q["question"], k=k, min_score=min_score)
        sources = [h.chunk.source for h in hits]
        expected = q["expected_source"]
        if expected is None:
            unanswerable += 1
            ok = not hits
            abstained += ok
        else:
            answerable += 1
            ok = expected in sources
            hits_ok += ok
            if ok:
                rr_sum += 1 / (sources.index(expected) + 1)
        details.append({"q": q, "hits": hits, "ok": ok})
    return {
        "hit_rate": hits_ok / answerable,
        "mrr": rr_sum / answerable,
        "abstention": abstained / unanswerable if unanswerable else 1.0,
        "details": details,
    }


def build(kind: str, chunks):
    return build_retriever(chunks, kind, config.min_score(kind))


def print_row(label: str, m: dict) -> None:
    print(f"{label:<12} hit@k {m['hit_rate']:>5.0%}   MRR {m['mrr']:.2f}   "
          f"abstention {m['abstention']:>5.0%}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--retriever", choices=["tfidf", "embeddings"], default=config.retriever_kind())
    parser.add_argument("--compare", action="store_true", help="tfidf vs embeddings")
    parser.add_argument("--sweep", action="store_true", help="essaie plusieurs seuils de score")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--min-hit-rate", type=float, default=0.85)
    parser.add_argument("--min-abstention", type=float, default=0.5)
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    chunks = load_chunks(config.docs_dir(), config.chunk_size(), config.chunk_overlap())

    if args.compare:
        # k=1 est la mesure la plus exigeante : la bonne source doit arriver en premier.
        for kind in ("tfidf", "embeddings"):
            retriever = build(kind, chunks)
            for k in sorted({1, args.k}):
                print_row(f"{kind} k={k}", evaluate(retriever, questions, k))
        return 0

    retriever = build(args.retriever, chunks)

    if args.sweep:
        print(f"Retriever : {args.retriever} (seuil par défaut : {retriever.min_score})")
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
            print_row(f"seuil {threshold:.2f}", evaluate(retriever, questions, args.k, threshold))
        return 0

    use_llm = args.with_llm and config.anthropic_api_key() is not None
    if args.with_llm and not use_llm:
        print("⚠️  --with-llm ignoré : ANTHROPIC_API_KEY absente.")

    result = evaluate(retriever, questions, args.k)
    kw_ok, refused = 0, 0
    for d in result["details"]:
        q = d["q"]
        tag = "hors sujet" if q["expected_source"] is None else q["expected_source"]
        print(f"{'OK ' if d['ok'] else 'KO '} [{tag}] {q['question']}")
        if use_llm:
            answer, _ = generate_answer(q["question"], d["hits"])
            if q["expected_source"] is None:
                refused += NO_ANSWER[:15].lower() in answer.lower()
            else:
                kw_ok += all(k.lower() in answer.lower() for k in q["keywords"])

    print(f"\nRetriever    : {args.retriever}")
    print(f"hit@{args.k}        : {result['hit_rate']:.0%}")
    print(f"MRR          : {result['mrr']:.2f}")
    print(f"abstention   : {result['abstention']:.0%}")
    if use_llm:
        n_ans = sum(1 for q in questions if q["expected_source"])
        n_oos = len(questions) - n_ans
        print(f"exactitude   : {kw_ok}/{n_ans}")
        print(f"refus LLM    : {refused}/{n_oos}")

    if result["hit_rate"] < args.min_hit_rate or result["abstention"] < args.min_abstention:
        print("❌ Seuils non atteints")
        return 1
    print("✅ Seuils atteints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
