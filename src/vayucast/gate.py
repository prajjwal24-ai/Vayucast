from __future__ import annotations

from pydantic import BaseModel


class GateDecision(BaseModel):
    passed: bool
    reason: str
    challenger_mae: float
    baseline_mae: float
    champion_mae: float | None
    improvement_over_baseline_pct: float
    delta_vs_champion_pct: float | None


def evaluate_quality_gate(
    challenger_mae: float,
    baseline_mae: float,
    champion_mae: float | None,
    max_regression_pct: float = 5.0,
) -> GateDecision:
    """Dual quality gate: beat persistence baseline AND stay within relative tolerance."""
    if baseline_mae <= 0:
        raise ValueError("baseline_mae must be positive.")

    improvement_pct = ((baseline_mae - challenger_mae) / baseline_mae) * 100.0
    delta_vs_champion: float | None = None

    if challenger_mae >= baseline_mae:
        return GateDecision(
            passed=False,
            reason=(
                f"Failed baseline gate: challenger MAE ({challenger_mae:.2f}) "
                f"did not beat persistence baseline ({baseline_mae:.2f})."
            ),
            challenger_mae=round(challenger_mae, 4),
            baseline_mae=round(baseline_mae, 4),
            champion_mae=round(champion_mae, 4) if champion_mae is not None else None,
            improvement_over_baseline_pct=round(improvement_pct, 2),
            delta_vs_champion_pct=None,
        )

    if champion_mae is not None and champion_mae > 0:
        delta_vs_champion = ((challenger_mae - champion_mae) / champion_mae) * 100.0
        max_allowed = champion_mae * (1.0 + max_regression_pct / 100.0)
        if challenger_mae > max_allowed:
            return GateDecision(
                passed=False,
                reason=(
                    f"Failed relative gate: challenger MAE ({challenger_mae:.2f}) regressed "
                    f"+{delta_vs_champion:.2f}% vs champion ({champion_mae:.2f}), "
                    f"exceeding +{max_regression_pct:.1f}% limit."
                ),
                challenger_mae=round(challenger_mae, 4),
                baseline_mae=round(baseline_mae, 4),
                champion_mae=round(champion_mae, 4),
                improvement_over_baseline_pct=round(improvement_pct, 2),
                delta_vs_champion_pct=round(delta_vs_champion, 2),
            )

    reason = (
        f"Passed baseline gate (-{improvement_pct:.2f}% vs persistence) "
        + (
            "and relative gate vs champion."
            if champion_mae is not None
            else "(initial bootstrap promotion)."
        )
    )
    return GateDecision(
        passed=True,
        reason=reason,
        challenger_mae=round(challenger_mae, 4),
        baseline_mae=round(baseline_mae, 4),
        champion_mae=round(champion_mae, 4) if champion_mae is not None else None,
        improvement_over_baseline_pct=round(improvement_pct, 2),
        delta_vs_champion_pct=round(delta_vs_champion, 2)
        if delta_vs_champion is not None
        else None,
    )