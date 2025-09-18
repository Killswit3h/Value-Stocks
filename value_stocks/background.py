"""Generate contextual blurbs for candidates."""
from __future__ import annotations

from datetime import date
from typing import List

from .models import CandidateMetrics
from .utils import format_percent


def build_background_blurb(metrics: CandidateMetrics, run_date: date) -> str:
    """Create a concise multi-sentence background string."""

    sentences: List[str] = []
    sentences.append(
        f"{metrics.company_name} ({metrics.ticker}) operates in the {metrics.industry.lower()} space within {metrics.sector}."
    )
    if metrics.triggered_conditions:
        sentences.append(
            "The stock qualified due to "
            + ", ".join(metrics.triggered_conditions)
            + "."
        )
    if metrics.free_cash_flow_yield is not None and metrics.five_year_stats.median_pe:
        sentences.append(
            "Valuation screens attractive with FCF yield "
            f"{format_percent(metrics.free_cash_flow_yield)} and P/E {metrics.pe_ratio:.1f} "
            f"vs 5Y median {metrics.five_year_stats.median_pe:.1f}."
        )
    if metrics.revenue_cagr_3y is not None and metrics.roic is not None:
        sentences.append(
            "Fundamentals show {growth} revenue CAGR and ROIC {roic}.".format(
                growth=format_percent(metrics.revenue_cagr_3y),
                roic=format_percent(metrics.roic),
            )
        )
    risk_parts: List[str] = []
    if metrics.beta is not None:
        risk_parts.append(f"beta {metrics.beta:.2f}")
    if metrics.realized_vol_30d is not None:
        risk_parts.append(f"30d vol {format_percent(metrics.realized_vol_30d)}")
    if metrics.short_interest_percent is not None:
        risk_parts.append(f"short interest {format_percent(metrics.short_interest_percent)}")
    if risk_parts:
        sentences.append("Risk check: " + ", ".join(risk_parts) + ".")
    if metrics.headline_flags:
        sentences.append(
            "Watch headlines: " + ", ".join(metrics.headline_flags) + "."
        )
    sentences.append(
        "Key risks include execution on turn-around and maintaining liquidity buffers."
    )
    return " ".join(sentences)
