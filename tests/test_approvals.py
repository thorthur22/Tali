import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tali.approvals import ApprovalManager


def test_auto_approve_all_bypasses_requires_approval() -> None:
    approvals = ApprovalManager(mode="auto_approve_all")
    result = approvals.resolve(
        prompt_fn=lambda _: "n",
        tool_name="fs.write",
        signature="fs.write:/tmp/example",
        requires_approval=True,
        reason="risk",
    )
    assert result.approved is True
    assert result.approval_mode == "auto"
