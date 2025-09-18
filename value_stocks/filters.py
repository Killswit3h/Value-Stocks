"""Filter logic for candidate screening."""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

from .models import CandidateMetrics, FilterResult


def determine_drop_trigger(
    metrics: CandidateMetrics,
    expanded: bool = False,
) -> Tuple[bool, str | None, List[str]]:
    """Determine if the stock satisfies the drop condition."""

    triggered_reasons: List[str] = []
    trigger_name: str | None = None
    passed = False

    if metrics.change_1d is not None and metrics.change_1d <= -0.40:
        passed = True
        trigger_name = "1-day −40%"
        triggered_reasons.append(f"1d move {metrics.change_1d:.2%}")

    drawdown_threshold = -0.40 if not expanded else -0.40
    day_change_threshold = -0.08 if not expanded else -0.04

    if (
        metrics.distance_from_52w_high is not None
        and metrics.distance_from_52w_high <= drawdown_threshold
        and metrics.change_1d is not None
        and metrics.change_1d <= day_change_threshold
    ):
        passed = True
        if trigger_name is None:
            trigger_name = "52w −40% + down day"
        triggered_reasons.append(
            f"52w drawdown {metrics.distance_from_52w_high:.2%}, 1d {metrics.change_1d:.2%}"
        )

    return passed, trigger_name, triggered_reasons


def passes_value_filters(
    metrics: CandidateMetrics,
    config: Dict[str, float],
    run_date: date,
) -> FilterResult:
    """Apply the hard survivability filters."""

    reasons: List[str] = []

    if metrics.trading_halted:
        reasons.append("Trading halted today")

    if not metrics.operating_cash_flow_positive:
        reasons.append("Negative operating cash flow")

    max_net_debt = config.get("max_net_debt_ebitda", 4)
    if metrics.net_debt_to_ebitda is None:
        reasons.append("Missing net debt/EBITDA")
    elif metrics.net_debt_to_ebitda > max_net_debt and metrics.net_debt_to_ebitda > 0:
        reasons.append("Net debt/EBITDA above limit")

    min_interest = config.get("min_interest_coverage", 3)
    if metrics.interest_coverage is None or metrics.interest_coverage < min_interest:
        reasons.append("Interest coverage below threshold")

    allowed_low_margin = set(config.get("allow_low_margin_sectors", []))
    min_margin = config.get("min_gross_margin", 0.15)
    if metrics.gross_margin is None:
        reasons.append("Missing gross margin")
    else:
        if metrics.sector not in allowed_low_margin and metrics.gross_margin < min_margin:
            reasons.append("Gross margin below threshold")

    earnings_buffer = int(config.get("earnings_days_buffer", 2))
    if metrics.earnings_date is not None:
        delta_days = (metrics.earnings_date - run_date).days
        if abs(delta_days) <= earnings_buffer:
            reasons.append("Within earnings blackout window")

    flagged_window_days = int(config.get("flagged_headline_window_days", 3))
    if metrics.headline_flags and flagged_window_days >= 0:
        reasons.append("Recent headline risk")

    if metrics.sec_delinquent:
        reasons.append("SEC delinquency flagged")

    if metrics.going_concern:
        reasons.append("Going concern risk")

    passed = len(reasons) == 0
    return FilterResult(passed=passed, reasons=reasons)


def earnings_window_status(metrics: CandidateMetrics, config: Dict[str, float], run_date: date) -> str:
    """Return the status for the earnings window select field."""

    buffer_days = int(config.get("earnings_days_buffer", 2))
    window_days = int(config.get("earnings_window_days", 10))
    if metrics.earnings_date is None:
        return "Safe"
    delta_days = (metrics.earnings_date - run_date).days
    if abs(delta_days) <= buffer_days:
        return "Inside 2d"
    if abs(delta_days) <= window_days:
        return "Near"
    return "Safe"
