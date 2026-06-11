from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ai_engineering_showcase.cli import app

runner = CliRunner()


def test_evaluate_command_writes_structured_report(tmp_path: Path) -> None:
    output = tmp_path / "evaluation_report.json"
    index_path = tmp_path / "vector_store.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--queries",
            "examples/queries.jsonl",
            "--output",
            str(output),
            "--index-path",
            str(index_path),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["total_cases"] == 5
    assert report["top_k"] == 4
    for metric in ("precision_at_k", "recall_at_k", "mean_reciprocal_rank", "context_hit_rate"):
        assert 0.0 <= report["retrieval"][metric] <= 1.0
    for metric in ("keyword_coverage", "groundedness", "refusal_correctness"):
        assert 0.0 <= report["answers"][metric] <= 1.0
    assert len(report["cases"]) == 5
    # The stdout report matches the file, so the command is scriptable.
    assert '"total_cases": 5' in result.output


def test_evaluate_command_creates_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "dir" / "report.json"
    index_path = tmp_path / "vector_store.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--queries",
            "examples/queries.jsonl",
            "--output",
            str(output),
            "--index-path",
            str(index_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()


def test_evaluate_command_fails_on_invalid_queries_file(tmp_path: Path) -> None:
    queries = tmp_path / "bad.jsonl"
    queries.write_text("not json\n", encoding="utf-8")
    index_path = tmp_path / "vector_store.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--queries",
            str(queries),
            "--output",
            str(tmp_path / "report.json"),
            "--index-path",
            str(index_path),
        ],
    )
    assert result.exit_code != 0
