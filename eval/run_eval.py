"""Évalue l'assistant sur eval/questions.json.

Métriques sans LLM (gratuites, utilisées en CI) :
  - hit@k      : la bonne source est-elle dans les k passages retrouvés ?
  - MRR        : rang moyen (inversé) de la bonne source
  - abstention : pour les questions hors sujet, on ne retrouve rien

Avec --with-llm (et ANTHROPIC_API_KEY) :
  - exactitude : la réponse contient-elle les mots-clés attendus ?
  - refus      : le LLM répond-il « je ne sais pas » aux questions hors sujet ?

Usage : python -m eval.run_eval [--k 3] [--with-llm]
"""

import argparse
import json
import sys
from pathlib import Path

from app import config
from app.ingest import load_chunks
from app.llm import NO_ANSWER, generate_answer
from app.retriever import Retriever

QUESTIONS = Path(__file__).parent / "questions.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--min-hit-rate", type=float, default=0.85)
    parser.add_argument("--min-abstention", type=float, default=0.5)
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    chunks = load_chunks(config.docs_dir(), config.chunk_size(), config.chunk_overlap())
    retriever = Retriever(chunks, min_score=config.min_score())

    use_llm = args.with_llm and config.anthropic_api_key() is not None
    if args.with_llm and not use_llm:
        print("⚠️  --with-llm ignoré : ANTHROPIC_API_KEY absente.")

    hits_ok, rr_sum, answerable = 0, 0.0, 0
    abstained, unanswerable = 0, 0
    kw_ok, refused = 0, 0

    for q in questions:
        hits = retriever.search(q["question"], k=args.k)
        sources = [h.chunk.source for h in hits]
        expected = q["expected_source"]
        if expected is None:
            unanswerable += 1
            ok = not hits
            abstained += ok
            line = f"{'OK ' if ok else 'KO '} [hors sujet] {q['question']}"
            if use_llm:
                answer, _ = generate_answer(q["question"], hits)
                refused += NO_ANSWER[:15].lower() in answer.lower()
        else:
            answerable += 1
            ok = expected in sources
            hits_ok += ok
            if ok:
                rr_sum += 1 / (sources.index(expected) + 1)
            line = f"{'OK ' if ok else 'KO '} [{expected}] {q['question']}"
            if use_llm:
                answer, _ = generate_answer(q["question"], hits)
                kw_ok += all(k.lower() in answer.lower() for k in q["keywords"])
        print(line)

    hit_rate = hits_ok / answerable
    abstention = abstained / unanswerable if unanswerable else 1.0
    print(f"\nhit@{args.k}      : {hit_rate:.0%} ({hits_ok}/{answerable})")
    print(f"MRR          : {rr_sum / answerable:.2f}")
    print(f"abstention   : {abstention:.0%} ({abstained}/{unanswerable})")
    if use_llm:
        print(f"exactitude   : {kw_ok / answerable:.0%} ({kw_ok}/{answerable})")
        print(f"refus LLM    : {refused}/{unanswerable}")

    if hit_rate < args.min_hit_rate or abstention < args.min_abstention:
        print("❌ Seuils non atteints")
        return 1
    print("✅ Seuils atteints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
