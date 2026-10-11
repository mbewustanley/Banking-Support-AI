"""
Load and evaluate generation_eval.json
run each question through the generator
compare returned sources with expected sources documents
check expected facts and abstentionbehavior
save a json report containing per-case results abd aggregate metrics
"""


"""Evaluate grounded answer generation against a golden dataset."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.generation import GroundedAnswerGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_PATH = PROJECT_ROOT / "evals" / "generation_eval.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "generation_eval_report.json"


class GenerationEvaluator:
    """Run and evaluate a grounded answer generator against golden cases."""

    def __init__(
        self,
        generator: GroundedAnswerGenerator | None = None,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        report_path: str | Path = DEFAULT_REPORT_PATH,
    ) -> None:
        self.generator = generator or GroundedAnswerGenerator()
        self.dataset_path = Path(dataset_path)
        self.report_path = Path(report_path)
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> dict[str, Any]:
        """Load the dataset and validate its basic structure."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {self.dataset_path}"
            )

        with self.dataset_path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)

        if not isinstance(dataset, dict):
            raise ValueError("Evaluation dataset must be a JSON object")

        cases = dataset.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("Evaluation dataset must contain a non-empty cases list")

        seen_ids: set[str] = set()

        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("Each evaluation case must be a JSON object")

            required_fields = {
                "id",
                "question",
                "expected_facts",
                "expected_sources",
                "should_abstain",
            }
            missing = required_fields - case.keys()

            if missing:
                raise ValueError(
                    f"Evaluation case is missing fields: {sorted(missing)}"
                )

            case_id = case["id"]

            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError("Each evaluation case must have a non-empty string ID")

            if case_id in seen_ids:
                raise ValueError(f"Duplicate evaluation case ID: {case_id}")

            seen_ids.add(case_id)

            if not isinstance(case["question"], str) or not case["question"].strip():
                raise ValueError(f"{case_id}: question must be a non-empty string")

            if not isinstance(case["expected_facts"], list):
                raise ValueError(f"{case_id}: expected_facts must be a list")

            if not isinstance(case["expected_sources"], list):
                raise ValueError(f"{case_id}: expected_sources must be a list")

            if not isinstance(case["should_abstain"], bool):
                raise ValueError(f"{case_id}: should_abstain must be a boolean")

        return dataset

    @staticmethod
    def _normalise(text: str) -> str:
        """Normalize text for simple, case-insensitive phrase checks."""
        return " ".join(text.casefold().split())

    @staticmethod
    def _source_filename(source: str) -> str:
        """Return a source's filename to tolerate different path prefixes."""
        return Path(source.replace("\\", "/")).name.casefold()

    @staticmethod
    def _valid_citations(
        answer: str,
        retrieval_results: list[dict[str, Any]],
    ) -> tuple[bool, list[str], list[str]]:
        """Check that cited IDs exist in the current retrieval results."""
        import re

        cited_ids = re.findall(r"\[(S\d+)\]", answer)
        available_ids = {
            item.get("citation_id")
            for item in retrieval_results
            if isinstance(item, dict)
        }

        invalid_ids = sorted(set(cited_ids) - available_ids)

        # An abstention or a clearly stated evidence gap need not cite a source.
        valid = not invalid_ids

        return valid, sorted(set(cited_ids)), invalid_ids

    def _evaluate_case(self, case: dict[str, Any]) -> dict[str, Any]:
        """Run one case and evaluate observable generation behavior."""
        case_id = case["id"]

        try:
            result = self.generator.answer(case["question"])
            answer = result.answer
            abstained = result.abstained
            sources = result.sources
            retrieval_results = result.retrieval_results

            normalized_answer = self._normalise(answer)

            expected_facts = case["expected_facts"]

            # Baseline phrase matching: each expected fact must occur
            # as a complete normalized phrase in the answer.
            fact_checks = [
                {
                    "fact": fact,
                    "matched": self._normalise(fact) in normalized_answer,
                }
                for fact in expected_facts
            ]

            source_names = {
                self._source_filename(str(source.get("source", "")))
                for source in sources
                if isinstance(source, dict)
            }

            expected_sources = {
                self._source_filename(str(source))
                for source in case["expected_sources"]
            }

            matched_sources = sorted(source_names & expected_sources)
            missing_sources = sorted(expected_sources - source_names)

            citations_valid, cited_ids, invalid_ids = self._valid_citations(
                answer,
                retrieval_results,
            )

            abstention_correct = abstained == case["should_abstain"]
            facts_passed = all(item["matched"] for item in fact_checks)

            # A case with no expected facts passes the phrase check by
            # definition. This does not establish semantic correctness.
            source_coverage_passed = (
                not expected_sources or not missing_sources
            )

            passed = (
                abstention_correct
                and facts_passed
                and source_coverage_passed
                and citations_valid
            )

            return {
                "case_id": case_id,
                "category": case.get("category", "uncategorized"),
                "question": case["question"],
                "answer": answer,
                "expected_abstention": case["should_abstain"],
                "actual_abstention": abstained,
                "abstention_correct": abstention_correct,
                "fact_checks": fact_checks,
                "facts_passed": facts_passed,
                "expected_sources": sorted(expected_sources),
                "matched_sources": matched_sources,
                "missing_sources": missing_sources,
                "source_coverage_passed": source_coverage_passed,
                "cited_ids": cited_ids,
                "invalid_citation_ids": invalid_ids,
                "citations_valid": citations_valid,
                "passed": passed,
                "error": None,
            }

        except Exception as exc:
            return {
                "case_id": case_id,
                "category": case.get("category", "uncategorized"),
                "question": case.get("question"),
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def evaluate(self) -> dict[str, Any]:
        """Evaluate all cases and return an aggregate report."""
        case_results = [
            self._evaluate_case(case)
            for case in self.dataset["cases"]
        ]

        total = len(case_results)
        passed = sum(result["passed"] for result in case_results)
        failed = total - passed

        def rate(key: str) -> float:
            eligible = [
                result for result in case_results if key in result
            ]
            if not eligible:
                return 0.0
            return round(
                sum(bool(result[key]) for result in eligible) / len(eligible),
                4,
            )

        report = {
            "dataset_name": self.dataset.get(
                "dataset_name", "unnamed_generation_eval"
            ),
            "dataset_version": self.dataset.get("version"),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_method": {
                "fact_check": "Exact normalized phrase matching",
                "citation_check": "Citation IDs must exist in retrieval results",
                "source_check": "Expected source filename coverage",
                "limitation": (
                    "Passing deterministic checks does not prove semantic "
                    "faithfulness or claim-level citation support."
                ),
            },
            "summary": {
                "total_cases": total,
                "passed_cases": passed,
                "failed_cases": failed,
                "pass_rate": round(passed / total, 4) if total else 0.0,
                "abstention_accuracy": rate("abstention_correct"),
                "fact_check_pass_rate": rate("facts_passed"),
                "source_coverage_pass_rate": rate("source_coverage_passed"),
                "citation_validity_rate": rate("citations_valid"),
            },
            "results": case_results,
        }

        return report

    def save_report(self, report: dict[str, Any]) -> Path:
        """Save the report as formatted JSON."""
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        with self.report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        return self.report_path


if __name__ == "__main__":
    evaluator = GenerationEvaluator()
    evaluation_report = evaluator.evaluate()
    report_path = evaluator.save_report(evaluation_report)

    summary = evaluation_report["summary"]

    print("=" * 60)
    print("GENERATION EVALUATION")
    print("=" * 60)
    print(f"Dataset: {evaluation_report['dataset_name']}")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed_cases']}")
    print(f"Failed: {summary['failed_cases']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Abstention accuracy: {summary['abstention_accuracy']:.1%}")
    print(f"Fact-check pass rate: {summary['fact_check_pass_rate']:.1%}")
    print(f"Source coverage: {summary['source_coverage_pass_rate']:.1%}")
    print(f"Citation validity: {summary['citation_validity_rate']:.1%}")
    print(f"\nReport saved to: {report_path}")
