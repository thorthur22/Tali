from __future__ import annotations

from tali.models import RetrievalBundle


def format_retrieval_context(bundle: RetrievalBundle) -> str:
    parts: list[str] = []
    parts.append("[Active Commitments]")
    if bundle.commitments:
        for commitment in bundle.commitments:
            parts.append(f"- ({commitment.status}) {commitment.description}")
    else:
        parts.append("- None")
    parts.append("\n[Relevant Facts]")
    if bundle.facts:
        for fact in bundle.facts:
            parts.append(
                f"- {fact.statement} (id={fact.id}, provenance={fact.provenance_type.value}, confidence={fact.confidence:.2f})"
            )
    else:
        parts.append("- None")
    parts.append("\n[Recent Episodes]")
    if bundle.episodes:
        for episode in bundle.episodes:
            parts.append(f"- {episode.timestamp}: {episode.user_input} -> {episode.agent_output}")
    else:
        parts.append("- None")

    parts.append("\n[Recent Reflections]")
    if getattr(bundle, "reflections", None):
        for ref in bundle.reflections:
            status = "success" if ref.success else "failed"
            parts.append(
                f"- {ref.timestamp} ({status}): worked={ref.what_worked} | failed={ref.what_failed} | next={ref.next_time}"
            )
    else:
        parts.append("- None")
    parts.append("\n[Preferences]")
    if bundle.preferences:
        for pref in bundle.preferences:
            parts.append(
                f"- {pref.key} = {pref.value} (provenance={pref.provenance_type.value}, confidence={pref.confidence:.2f})"
            )
    else:
        parts.append("- None")
    parts.append("\n[Skills]")
    if bundle.skills:
        for skill in bundle.skills:
            parts.append(f"- {skill}")
    else:
        parts.append("- None")
    return "\n".join(parts)
