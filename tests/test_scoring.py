"""Unit tests for scoring."""
from __future__ import annotations

from datetime import date, timedelta

from value_stocks.models import CandidateMetrics, FiveYearStats
from value_stocks.scoring import calculate_scores


def _make_metrics(ticker: str, fcf: float, earnings_offset: int = 20) -> CandidateMetrics:
    return CandidateMetrics(
        ticker=ticker,
        company_name="Test Co",
        sector="Industrials",
        industry="Tools",
        market_cap=1_000_000_000,
        price=25.0,
        open_price=26.0,
        high_price=27.0,
        low_price=24.0,
        change_1d=-0.42,
        change_5d=-0.1,
        change_1m=-0.2,
        change_6m=-0.3,
        change_1y=-0.5,
        change_5y=0.2,
        distance_from_52w_high=-0.45,
        distance_from_200dma=-0.2,
        percent_to_200dma=0.2,
        rsi_14=25.0,
        beta=1.1,
        realized_vol_30d=0.45,
        short_interest_percent=0.05,
        avg_dollar_volume=10_000_000,
        shares_outstanding_trend=-0.01,
        free_cash_flow_yield=fcf,
        ev_to_ebitda=5.0,
        pe_ratio=10.0,
        pb_ratio=1.0,
        ps_ratio=2.0,
        gross_margin=0.35,
        gross_margin_stability=0.05,
        operating_margin=0.18,
        revenue_cagr_3y=0.12,
        roic=0.11,
        net_debt_to_ebitda=1.0,
        interest_coverage=5.0,
        operating_cash_flow_positive=True,
        insider_net_buy_percent=0.02,
        earnings_date=date.today() + timedelta(days=earnings_offset),
        last_earnings_date=date.today() - timedelta(days=90),
        headline_flags=[],
        sec_delinquent=False,
        going_concern=False,
        trading_halted=False,
        price_history_5y=[],
        five_year_stats=FiveYearStats(),
    )


def test_higher_fcf_yield_scores_better() -> None:
    metrics_a = _make_metrics("AAA", 0.15)
    metrics_b = _make_metrics("BBB", 0.05)
    config = {
        "valuation": {"fcf_yield_weight": 15, "ev_to_ebitda_weight": 10, "price_to_book_weight": 5},
        "quality": {"roic_weight": 10, "gross_margin_stability_weight": 5, "revenue_cagr_weight": 5},
        "momentum": {"rsi_weight": 10, "distance_to_200dma_weight": 5},
        "risk": {"beta_weight": 10, "beta_band_center": 1.0, "beta_band_halfwidth": 0.4, "volatility_weight": 5},
        "liquidity": {"dollar_volume_weight": 5, "short_interest_weight": 5},
        "penalties": {"earnings_weight": 10, "headline_weight": 25, "sec_delinquency_weight": 40, "going_concern_weight": 60},
    }
    scores = calculate_scores([metrics_a, metrics_b], config, date.today())
    assert scores["AAA"].total > scores["BBB"].total


def test_earnings_penalty_applied() -> None:
    metrics_safe = _make_metrics("SAFE", 0.12, earnings_offset=30)
    metrics_near = _make_metrics("NEAR", 0.12, earnings_offset=3)
    config = {
        "valuation": {"fcf_yield_weight": 15, "ev_to_ebitda_weight": 10, "price_to_book_weight": 5},
        "quality": {"roic_weight": 10, "gross_margin_stability_weight": 5, "revenue_cagr_weight": 5},
        "momentum": {"rsi_weight": 10, "distance_to_200dma_weight": 5},
        "risk": {"beta_weight": 10, "beta_band_center": 1.0, "beta_band_halfwidth": 0.4, "volatility_weight": 5},
        "liquidity": {"dollar_volume_weight": 5, "short_interest_weight": 5},
        "penalties": {"earnings_weight": 10, "headline_weight": 25, "sec_delinquency_weight": 40, "going_concern_weight": 60},
    }
    scores = calculate_scores([metrics_safe, metrics_near], config, date.today())
    assert scores["SAFE"].total > scores["NEAR"].total
