from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_engineering_showcase.api import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AI_SHOWCASE_INDEX_PATH", str(tmp_path / "vector_store.json"))
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
