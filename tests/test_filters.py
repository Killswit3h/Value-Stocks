"""Unit tests for filters."""
from __future__ import annotations

from datetime import date, timedelta

from value_stocks.filters import determine_drop_trigger, passes_value_filters
from value_stocks.models import CandidateMetrics, FiveYearStats


def _baseline_metrics() -> CandidateMetrics:
    return CandidateMetrics(
        ticker="AAA",
        company_name="Test Co",
        sector="Industrials",
        industry="Tools",
        market_cap=1_000_000_000,
        price=25.0,
        open_price=26.0,
        high_price=27.0,
        low_price=24.0,
        change_1d=-0.45,
        change_5d=-0.12,
        change_1m=-0.2,
        change_6m=-0.3,
        change_1y=-0.5,
        change_5y=0.1,
        distance_from_52w_high=-0.5,
        distance_from_200dma=-0.2,
        percent_to_200dma=0.2,
        rsi_14=25.0,
        beta=1.0,
        realized_vol_30d=0.4,
        short_interest_percent=0.06,
        avg_dollar_volume=6_000_000,
        shares_outstanding_trend=0.0,
        free_cash_flow_yield=0.12,
        ev_to_ebitda=6.0,
        pe_ratio=11.0,
        pb_ratio=1.1,
        ps_ratio=2.2,
        gross_margin=0.4,
        gross_margin_stability=0.05,
        operating_margin=0.2,
        revenue_cagr_3y=0.1,
        roic=0.12,
        net_debt_to_ebitda=1.5,
        interest_coverage=6.0,
        operating_cash_flow_positive=True,
        insider_net_buy_percent=0.01,
        earnings_date=date.today() + timedelta(days=15),
        last_earnings_date=date.today() - timedelta(days=90),
        headline_flags=[],
        sec_delinquent=False,
        going_concern=False,
        trading_halted=False,
        price_history_5y=[],
        five_year_stats=FiveYearStats(),
    )


def test_determine_drop_trigger_single_day() -> None:
    metrics = _baseline_metrics()
    passed, trigger, reasons = determine_drop_trigger(metrics, expanded=False)
    assert passed
    assert trigger == "1-day −40%"
    assert reasons


def test_value_filter_rejects_high_leverage() -> None:
    metrics = _baseline_metrics()
    metrics.net_debt_to_ebitda = 5.0
    result = passes_value_filters(
        metrics,
        {
            "max_net_debt_ebitda": 4,
            "min_interest_coverage": 3,
            "min_gross_margin": 0.15,
            "allow_low_margin_sectors": [],
            "earnings_days_buffer": 2,
            "flagged_headline_window_days": 3,
        },
        date.today(),
    )
    assert not result.passed
    assert any("Net debt" in reason for reason in result.reasons)
