import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali.db import Database
from tali.patches import (
    PatchProposal,
    PatchReviewResult,
    parse_patch_proposal,
    review_patch,
    store_patch_proposal,
    _build_review_prompt,
    _parse_review_result,
)


@dataclass
class FakeLLMResponse:
    content: str
    model: str = "fake"


class FakeLLM:
    """Minimal fake LLM that returns a canned review response."""

    def __init__(self, content: str) -> None:
        self._content = content

    def generate(self, prompt: str) -> FakeLLMResponse:
        return FakeLLMResponse(content=self._content)


class PatchReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tali.db"
        self.db = Database(self.db_path)
        self.db.initialize()
        self.proposal = PatchProposal(
            title="Add safety hook",
            rationale="Improve hook safety checks",
            files=["src/tali/hooks/safety.py"],
            diff_text="--- a/README.md\n+++ b/README.md\n@@\n+Safety note\n",
            tests=["pytest tests/test_hooks.py"],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_review_approved(self) -> None:
        llm = FakeLLM(json.dumps({"approved": True, "issues": []}))
        result = review_patch(llm, self.proposal)
        self.assertIsInstance(result, PatchReviewResult)
        self.assertTrue(result.approved)
        self.assertEqual(result.issues, [])
        self.assertIn("safety reviewer", result.reviewer_prompt)

    def test_review_rejected_with_issues(self) -> None:
        llm = FakeLLM(json.dumps({
            "approved": False,
            "issues": ["Modifies core runtime", "Missing test coverage"],
        }))
        result = review_patch(llm, self.proposal)
        self.assertFalse(result.approved)
        self.assertEqual(len(result.issues), 2)
        self.assertIn("Modifies core runtime", result.issues)

    def test_review_invalid_json(self) -> None:
        llm = FakeLLM("This is not JSON at all")
        result = review_patch(llm, self.proposal)
        self.assertFalse(result.approved)
        self.assertIn("invalid json", result.issues[0])

    def test_review_empty_response(self) -> None:
        llm = FakeLLM("")
        result = review_patch(llm, self.proposal)
        self.assertFalse(result.approved)
        self.assertIn("empty review response", result.issues[0])

    def test_parse_review_result_approved(self) -> None:
        text = json.dumps({"approved": True, "issues": []})
        result = _parse_review_result(text, "test prompt")
        self.assertTrue(result.approved)

    def test_parse_review_result_not_object(self) -> None:
        text = json.dumps([1, 2, 3])
        result = _parse_review_result(text, "test prompt")
        self.assertFalse(result.approved)
        self.assertIn("must be object", result.issues[0])

    def test_build_review_prompt_includes_diff(self) -> None:
        prompt = _build_review_prompt(self.proposal)
        self.assertIn("Safety note", prompt)
        self.assertIn("Add safety hook", prompt)
        self.assertIn("safety reviewer", prompt.lower())

    def test_store_and_review_patch_in_db(self) -> None:
        proposal_id = store_patch_proposal(self.db, self.proposal)
        row = self.db.fetch_patch_proposal(proposal_id)
        self.assertEqual(row["status"], "proposed")
        self.assertIsNone(row["review_json"])

        # Store review
        review_data = {"approved": True, "issues": [], "reviewer_prompt": "test"}
        self.db.update_patch_review(
            proposal_id=proposal_id,
            review_json=json.dumps(review_data),
            status="proposed",
        )
        row = self.db.fetch_patch_proposal(proposal_id)
        self.assertIsNotNone(row["review_json"])
        parsed = json.loads(row["review_json"])
        self.assertTrue(parsed["approved"])

    def test_review_failed_status_in_db(self) -> None:
        proposal_id = store_patch_proposal(self.db, self.proposal)
        review_data = {
            "approved": False,
            "issues": ["Unsafe shell command"],
            "reviewer_prompt": "test",
        }
        self.db.update_patch_review(
            proposal_id=proposal_id,
            review_json=json.dumps(review_data),
            status="review_failed",
        )
        row = self.db.fetch_patch_proposal(proposal_id)
        self.assertEqual(row["status"], "review_failed")


if __name__ == "__main__":
    unittest.main()
