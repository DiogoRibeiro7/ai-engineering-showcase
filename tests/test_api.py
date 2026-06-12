from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_engineering_showcase.api import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AI_SHOWCASE_INDEX_PATH", str(tmp_path / "vector_store.json"))
    monkeypatch.setenv("AI_SHOWCASE_CONVERSATION_STORE_PATH", str(tmp_path / "conversations"))
    return TestClient(create_app())


def test_query_response_exposes_tool_metadata(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={"question": "What is the overall sentiment distribution?", "top_k": 3},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["tool_run"]["tool_name"] == "sentiment_summary"
    assert result["tool_run"]["status"] == "ok"
    assert result["tool_run"]["output"]["total_records"] > 0
    assert "Tool insight (sentiment_summary):" in result["answer"]


def test_query_response_without_tool_keeps_plain_rag(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={"question": "Why are enterprise customers unhappy with onboarding?", "top_k": 3},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["tool_run"] is None
    assert result["citations"]


def test_chat_creates_and_continues_a_conversation(client: TestClient) -> None:
    first = client.post(
        "/chat",
        json={"message": "Why are enterprise customers unhappy with onboarding?", "top_k": 3},
    )
    assert first.status_code == 200
    conversation_id = first.json()["conversation_id"]
    assert conversation_id
    assert first.json()["result"]["citations"]

    second = client.post(
        "/chat",
        json={"message": "What about pricing?", "conversation_id": conversation_id, "top_k": 3},
    )
    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id
    diagnostics = second.json()["result"]["diagnostics"]
    assert diagnostics["query_rewritten"] is True
    assert "onboarding" in diagnostics["retrieval_question"].lower()

    conversation = client.get(f"/conversations/{conversation_id}")
    assert conversation.status_code == 200
    turns = conversation.json()["turns"]
    assert [turn["user_message"] for turn in turns] == [
        "Why are enterprise customers unhappy with onboarding?",
        "What about pricing?",
    ]
    assert turns[0]["retrieved_document_ids"]


def test_chat_conversations_are_isolated(client: TestClient) -> None:
    first = client.post("/chat", json={"message": "Why is onboarding slow?"})
    second = client.post("/chat", json={"message": "Which integrations were requested?"})
    first_id = first.json()["conversation_id"]
    second_id = second.json()["conversation_id"]
    assert first_id != second_id
    first_turns = client.get(f"/conversations/{first_id}").json()["turns"]
    second_turns = client.get(f"/conversations/{second_id}").json()["turns"]
    assert len(first_turns) == 1
    assert len(second_turns) == 1
    assert first_turns[0]["user_message"] != second_turns[0]["user_message"]


def test_get_unknown_conversation_returns_404(client: TestClient) -> None:
    response = client.get("/conversations/does-not-exist")
    assert response.status_code == 404


def test_chat_with_invalid_conversation_id_returns_400(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "Why is onboarding slow?", "conversation_id": "bad id!"},
    )
    assert response.status_code == 400
    assert "invalid conversation_id" in response.json()["detail"]
