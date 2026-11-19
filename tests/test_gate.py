from __future__ import annotations

import pytest

from vayucast.gate import evaluate_quality_gate


def test_gate_passes_bootstrap_when_beating_baseline() -> None:
    decision = evaluate_quality_gate(
        challenger_mae=14.95,
        baseline_mae=16.11,
        champion_mae=None,
        max_regression_pct=5.0,
    )
    assert decision.passed is True
    assert decision.delta_vs_champion_pct is None
    assert decision.improvement_over_baseline_pct > 0


def test_gate_fails_when_worse_than_persistence_baseline() -> None:
    decision = evaluate_quality_gate(
        challenger_mae=17.0,
        baseline_mae=16.11,
        champion_mae=18.0,
        max_regression_pct=5.0,
    )
    assert decision.passed is False
    assert "baseline" in decision.reason.lower()


def test_gate_passes_within_relative_regression_tolerance() -> None:
    # 15.5 is +3.33% vs champion 15.0 (within +5% limit) and still beats baseline 16.5
    decision = evaluate_quality_gate(
        challenger_mae=15.5,
        baseline_mae=16.5,
        champion_mae=15.0,
        max_regression_pct=5.0,
    )
    assert decision.passed is True
    assert decision.delta_vs_champion_pct == pytest.approx(3.33, abs=0.05)


def test_gate_fails_when_exceeding_relative_regression_tolerance() -> None:
    # 16.0 is +6.67% vs champion 15.0 (exceeds +5% limit) even though it beats baseline 17.0
    decision = evaluate_quality_gate(
        challenger_mae=16.0,
        baseline_mae=17.0,
        champion_mae=15.0,
        max_regression_pct=5.0,
    )
    assert decision.passed is False
    assert "relative" in decision.reason.lower()


def test_gate_rejects_non_positive_baseline() -> None:
    with pytest.raises(ValueError, match="positive"):
        evaluate_quality_gate(challenger_mae=10.0, baseline_mae=0.0, champion_mae=None)