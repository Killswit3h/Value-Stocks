"""Dataclasses for the screening pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence


@dataclass
class PricePoint:
    """Simple price point used for charting."""

    date: date
    close: float
    volume: float | None = None


@dataclass
class FiveYearStats:
    """Contextual statistics for five year lookback."""

    median_pe: Optional[float] = None
    median_ev_to_ebitda: Optional[float] = None
    median_gross_margin: Optional[float] = None
    median_roic: Optional[float] = None
    rsi_under_30_frequency: Optional[float] = None
    median_forward_return_after_rsi: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    drawdown_from_200dma_pct: Optional[float] = None


@dataclass
class CandidateMetrics:
    """Metrics used for scoring and filtering."""

    ticker: str
    company_name: str
    sector: str
    industry: str
    market_cap: float
    price: float
    open_price: float
    high_price: float
    low_price: float
    change_1d: float
    change_5d: float
    change_1m: float
    change_6m: float
    change_1y: float
    change_5y: Optional[float]
    distance_from_52w_high: float
    distance_from_200dma: Optional[float]
    percent_to_200dma: Optional[float]
    rsi_14: Optional[float]
    beta: Optional[float]
    realized_vol_30d: Optional[float]
    short_interest_percent: Optional[float]
    avg_dollar_volume: float
    shares_outstanding_trend: Optional[float]
    free_cash_flow_yield: Optional[float]
    ev_to_ebitda: Optional[float]
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    ps_ratio: Optional[float]
    gross_margin: Optional[float]
    gross_margin_stability: Optional[float]
    operating_margin: Optional[float]
    revenue_cagr_3y: Optional[float]
    roic: Optional[float]
    net_debt_to_ebitda: Optional[float]
    interest_coverage: Optional[float]
    operating_cash_flow_positive: bool
    insider_net_buy_percent: Optional[float]
    earnings_date: Optional[date]
    last_earnings_date: Optional[date]
    headline_flags: List[str] = field(default_factory=list)
    sec_delinquent: bool = False
    going_concern: bool = False
    trading_halted: bool = False
    price_history_5y: Sequence[PricePoint] = field(default_factory=list)
    five_year_stats: FiveYearStats = field(default_factory=FiveYearStats)
    trigger: Optional[str] = None
    triggered_conditions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    chart_path: Optional[str] = None

    def add_flag(self, flag: str) -> None:
        if flag not in self.headline_flags:
            self.headline_flags.append(flag)


@dataclass
class FilterResult:
    """Outcome of running the filter checks."""

    passed: bool
    reasons: List[str]


@dataclass
class ScoreBreakdown:
    """Stores the component scores for a candidate."""

    total: float
    valuation: float
    quality: float
    momentum: float
    risk: float
    liquidity: float
    penalties: float
    normalized_inputs: Dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateResult:
    """Final aggregated result for a candidate."""

    metrics: CandidateMetrics
    filter_result: FilterResult
    score: ScoreBreakdown
    rank: Optional[int] = None


@dataclass
class RunMetadata:
    """Metadata about a pipeline execution."""

    run_datetime: datetime
    market_status: str
    universe_size: int
    considered: int
    passed_filters: int
    expanded_criteria_used: bool
    data_source: str
    api_warnings: List[str] = field(default_factory=list)
    run_log_path: Optional[str] = None


@dataclass
class RulesBlock:
    """Structured rules text for the report footer."""

    bullets: List[str]
