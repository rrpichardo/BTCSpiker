"""Deterministic Staging qualification gates for completed candidates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateEvidence:
    coverage_days: float
    quote_trade_coverage: bool
    folds_won: int
    bootstrap_lower: float
    brier_ratio: float
    event_f1_delta: float
    final_pr_auc_delta: float
    p95_latency_ms: float
    deployable: bool
    parity_passed: bool


@dataclass(frozen=True)
class QualificationResult:
    passed: bool
    reasons: tuple[str, ...]


def qualify(evidence: CandidateEvidence) -> QualificationResult:
    """Apply every Staging gate, retaining stable failure codes in gate order."""
    checks = {
        "coverage_under_thirty_days": evidence.coverage_days >= 30.0,
        "quote_trade_coverage_missing": evidence.quote_trade_coverage,
        "fewer_than_four_folds_won": evidence.folds_won >= 4,
        "bootstrap_lower_not_positive": evidence.bootstrap_lower > 0,
        "brier_regression_over_five_percent": evidence.brier_ratio <= 1.05,
        "event_f1_regressed": evidence.event_f1_delta >= 0,
        "final_pr_auc_not_improved": evidence.final_pr_auc_delta > 0,
        "p95_latency_over_800ms": evidence.p95_latency_ms <= 800,
        "feature_set_not_deployable": evidence.deployable,
        "feature_parity_failed": evidence.parity_passed,
    }
    reasons = tuple(reason for reason, passed in checks.items() if not passed)
    return QualificationResult(passed=not reasons, reasons=reasons)
