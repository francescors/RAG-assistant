  ![CI](https://github.com/francescors/rag-assistant/actions/workflows/ci.yml/badge.svg)
# RAG Assistant

Assistant de questions-réponses sur des documents, exposé via une API REST.
Il retrouve les passages pertinents (RAG), génère une réponse avec un LLM (Claude)
et cite ses sources. Le projet inclut tests, évaluation, Docker et CI.

> Les documents d'exemple (`data/docs/`) décrivent une entreprise fictive, NimbusDrive.
> Remplace-les par tes propres fichiers `.md` / `.txt` pour l'utiliser sur autre chose.

## Architecture

```
question ──► FastAPI (/ask) ──► Retriever (TF-IDF, top-k) ──► LLM (Claude) ──► réponse + sources
                                      ▲
              data/docs/*.md ─► découpage en chunks (au démarrage)
```

| Fichier | Rôle |
|---|---|
| `app/ingest.py` | Lecture des documents, découpage en chunks avec chevauchement |
| `app/retriever.py` | Index TF-IDF, similarité cosinus, seuil de score minimal |
| `app/llm.py` | Prompt, appel à l'API Anthropic, mode sans clé (extractif) |
| `app/main.py` | API : `POST /ask`, `GET /health`, validation, gestion d'erreurs, logs |
| `eval/run_eval.py` | Évaluation sur un jeu de questions (hit@k, MRR, abstention, exactitude) |
| `tests/` | Tests unitaires et d'API (aucun appel réseau) |

## Lancer en local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env        # puis renseigne ANTHROPIC_API_KEY (optionnel)
export $(grep -v '^#' .env | xargs)

uvicorn app.main:app --reload
```

Sans clé API, l'API fonctionne en mode `extractive` (elle renvoie le meilleur passage).
Avec une clé, elle utilise le LLM (`mode: "llm"`).

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Combien coûte l'\''offre Pro ?"}'
```

Documentation interactive : http://localhost:8000/docs

## Tests et évaluation

```bash
ruff check .                 # lint
pytest -q                    # tests
python -m eval.run_eval      # évaluation de la recherche (gratuit)
python -m eval.run_eval --with-llm   # + exactitude des réponses (nécessite une clé)
```

L'évaluation mesure : **hit@k** (la bonne source est-elle retrouvée ?), **MRR**,
**abstention** (rien n'est retrouvé pour une question hors sujet) et, avec le LLM,
**exactitude** (la réponse contient les mots-clés attendus) et **refus** (« je ne sais pas »).
La CI échoue si les seuils ne sont pas atteints : une régression de qualité est donc détectée
comme une régression de code.

## Docker

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 --env-file .env rag-assistant
```

## CI (GitHub Actions)

À chaque push et pull request : lint, tests, évaluation, puis build de l'image Docker
(`.github/workflows/ci.yml`).

## Choix techniques

- **TF-IDF plutôt que des embeddings** : simple, rapide, sans modèle à télécharger, déterministe
  (donc testable en CI). Limite : ne comprend pas les synonymes.
- **Mode sans clé** : permet la démo, les tests et la CI sans coût ni secret.
- **Prompt défensif** : le contexte est balisé et traité comme une donnée (anti prompt injection),
  et le modèle doit refuser de répondre hors du contexte.
- **Secrets** : uniquement via variables d'environnement ; `.env` est dans `.gitignore`.
- **Conteneur** : utilisateur non-root, cache des dépendances, healthcheck.

## Pistes d'amélioration

- Embeddings (ex. `sentence-transformers`) + base vectorielle (FAISS, Chroma, pgvector), puis
  comparaison hit@k / MRR avec TF-IDF grâce au jeu d'évaluation existant.
- Recherche hybride (TF-IDF + embeddings) et reranking.
- Jeu d'évaluation plus grand, évaluation de la fidélité (le LLM invente-t-il ?).
- Streaming des réponses, cache, limitation de débit, authentification.
- Déploiement (Render, Fly.io, Cloud Run) et lien de démo.
