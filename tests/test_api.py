import pytest
from fastapi.testclient import TestClient

from app import main
from app.llm import NO_ANSWER


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # jamais d'appel réseau
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_returns_answer_and_sources(client):
    r = client.post("/ask", json={"question": "Où sont hébergés les serveurs ?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "extractive"
    assert "Francfort" in body["answer"]
    assert body["sources"][0]["source"] == "securite.md"


def test_out_of_scope_question(client):
    r = client.post("/ask", json={"question": "Quelle est la capitale de l'Australie ?"})
    assert r.json()["answer"] == NO_ANSWER
    assert r.json()["sources"] == []


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "ab"}, {"question": "x" * 501}, {"question": "ok ok", "top_k": 99}],
)
def test_invalid_input_is_rejected(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_llm_failure_returns_502(client, monkeypatch):
    import anthropic
    import httpx

    def boom(*args, **kwargs):
        raise anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com")
        )

    monkeypatch.setattr(main, "generate_answer", boom)
    r = client.post("/ask", json={"question": "Combien coûte l'offre Pro ?"})
    assert r.status_code == 502
