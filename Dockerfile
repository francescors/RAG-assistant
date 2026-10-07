FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    RETRIEVER=embeddings \
    FASTEMBED_CACHE_PATH=/opt/fastembed
WORKDIR /srv

# Les dépendances d'abord : cette couche est mise en cache tant que requirements.txt ne change pas.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharge le modèle d'embeddings pendant le build : le conteneur démarre vite
# et n'a pas besoin d'accès à Internet pour la recherche.
RUN python -c "from fastembed import TextEmbedding; \
TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY app ./app
COPY data ./data

# Ne pas tourner en root dans le conteneur.
RUN useradd --create-home appuser && chown -R appuser /opt/fastembed
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
