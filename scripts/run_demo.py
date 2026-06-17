"""Run a local end-to-end demo."""

from __future__ import annotations

from pathlib import Path

from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.factory import build_agent, build_index


def main() -> None:
    """Build the index and run a demo query."""
    data_path = Path("data/sample_feedback.csv")
    index_path = Path(".artifacts/vector_store.json")
    build_index(data_path, index_path, embedding_dim=512)

    settings = Settings(data_path=data_path, index_path=index_path, embedding_dim=512)
    agent = build_agent(settings)
    answer = agent.answer("Why are enterprise customers unhappy with onboarding?", top_k=4)
    print(answer.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
