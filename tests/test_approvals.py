import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tali.approvals as approvals_module
from tali.approvals import ApprovalManager


class ApprovalManagerTests(unittest.TestCase):
    def test_auto_approve_all_bypasses_prompt(self) -> None:
        approvals = ApprovalManager(mode="auto_approve_all")
        outcome = approvals.resolve(
            prompt_fn=lambda _: "n",
            tool_name="shell.run",
            signature=None,
            requires_approval=True,
            reason="testing",
            details=None,
        )
        self.assertTrue(outcome.approved)
        self.assertEqual(outcome.approval_mode, "auto")

    def test_prompt_can_enable_auto_approve_all(self) -> None:
        approvals_module.questionary = None
        approvals = ApprovalManager(mode="prompt")
        outcome = approvals.resolve(
            prompt_fn=lambda _: "u",
            tool_name="shell.run",
            signature=None,
            requires_approval=True,
            reason="testing",
            details=None,
        )
        self.assertTrue(outcome.approved)
        self.assertEqual(approvals.mode, "auto_approve_all")


if __name__ == "__main__":
    unittest.main()
