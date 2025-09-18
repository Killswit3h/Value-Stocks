"""Market data provider implementations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import random
import statistics
import time

from .models import CandidateMetrics, FiveYearStats, PricePoint
from .utils import in_memory_cache, retry_with_backoff, setup_logger


def _percent_change(points: Sequence[PricePoint], periods: int) -> Optional[float]:
    if periods <= 0 or len(points) <= periods:
        return None
    end = points[-1].close
    start = points[-(periods + 1)].close
    if start == 0:
        return None
    return (end - start) / start


def _distance_from_high(points: Sequence[PricePoint], periods: int) -> Optional[float]:
    if not points:
        return None
    window = points[-(periods + 1) :] if len(points) > periods else points
    high = max(point.close for point in window)
    if high == 0:
        return None
    return (points[-1].close - high) / high


def _moving_average(points: Sequence[PricePoint], window: int) -> Optional[float]:
    if len(points) < window or window <= 0:
        return None
    subset = points[-window:]
    total = sum(point.close for point in subset)
    return total / window


def _distance_from_moving_average(points: Sequence[PricePoint], window: int) -> Tuple[Optional[float], Optional[float]]:
    ma = _moving_average(points, window)
    if ma is None or ma == 0:
        return None, None
    last = points[-1].close
    return (last - ma) / ma, abs(last - ma) / ma


def _compute_rsi_from_window(window: Sequence[float]) -> Optional[float]:
    if len(window) <= 1:
        return None
    gains = 0.0
    losses = 0.0
    for idx in range(1, len(window)):
        change = window[idx] - window[idx - 1]
        if change >= 0:
            gains += change
        else:
            losses += -change
    period = len(window) - 1
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _calculate_rsi(points: Sequence[PricePoint], period: int = 14) -> Optional[float]:
    if len(points) <= period:
        return None
    window = [point.close for point in points[-(period + 1) :]]
    return _compute_rsi_from_window(window)


def _daily_returns(points: Sequence[PricePoint]) -> List[float]:
    returns: List[float] = []
    for idx in range(1, len(points)):
        prev = points[idx - 1].close
        curr = points[idx].close
        if prev <= 0:
            continue
        returns.append((curr - prev) / prev)
    return returns


def _calculate_realized_vol(points: Sequence[PricePoint], window: int = 30) -> Optional[float]:
    if len(points) <= window:
        return None
    returns = _daily_returns(points[-(window + 1) :])
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(252)


def _calculate_avg_dollar_volume(points: Sequence[PricePoint], window: int = 30) -> Optional[float]:
    if not points:
        return None
    subset = points[-window:] if len(points) >= window else points
    values = [point.close * point.volume for point in subset if point.volume]
    if not values:
        return None
    return sum(values) / len(values)


def _calculate_beta(
    asset_points: Sequence[PricePoint], market_points: Sequence[PricePoint]
) -> Optional[float]:
    asset_returns = _daily_returns(asset_points)
    market_returns = _daily_returns(market_points)
    n = min(len(asset_returns), len(market_returns))
    if n < 2:
        return None
    asset_slice = asset_returns[-n:]
    market_slice = market_returns[-n:]
    mean_asset = sum(asset_slice) / n
    mean_market = sum(market_slice) / n
    covariance = sum((a - mean_asset) * (m - mean_market) for a, m in zip(asset_slice, market_slice))
    covariance /= n - 1
    variance = sum((m - mean_market) ** 2 for m in market_slice) / (n - 1)
    if variance == 0:
        return None
    return covariance / variance


def _max_drawdown(points: Sequence[PricePoint]) -> Optional[float]:
    if not points:
        return None
    peak = points[0].close
    max_drawdown = 0.0
    for point in points:
        if point.close > peak:
            peak = point.close
        if peak == 0:
            continue
        drawdown = (point.close - peak) / peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def _drawdown_from_200dma(points: Sequence[PricePoint]) -> Optional[float]:
    distance, _ = _distance_from_moving_average(points, 200)
    return distance


def _rsi_signal_stats(
    points: Sequence[PricePoint], period: int = 14, forward: int = 30
) -> Tuple[Optional[float], Optional[float]]:
    if len(points) <= period + 1:
        return None, None
    closes = [point.close for point in points]
    rsi_values: List[Tuple[int, float]] = []
    for idx in range(period, len(closes)):
        window = closes[idx - period : idx + 1]
        rsi = _compute_rsi_from_window(window)
        if rsi is None:
            continue
        rsi_values.append((idx, rsi))
    if not rsi_values:
        return None, None
    signals = [item for item in rsi_values if item[1] < 30]
    frequency = len(signals) / len(rsi_values) if rsi_values else None
    forward_returns: List[float] = []
    for idx, _ in signals:
        start = closes[idx]
        target_idx = min(idx + forward, len(closes) - 1)
        end = closes[target_idx]
        if start > 0:
            forward_returns.append((end - start) / start)
    median_forward = statistics.median(forward_returns) if forward_returns else None
    return frequency, median_forward


@dataclass
class ProviderConfig:
    base_url: str
    api_key: str | None
    max_retries: int
    backoff_seconds: int
    cache_ttl_seconds: int


class MarketDataProvider:
    """Facade that chooses between live and sample backends."""

    def __init__(self, config: ProviderConfig, use_sample: bool | None = None) -> None:
        self.logger = setup_logger(__name__)
        if use_sample is None:
            use_sample = not bool(config.api_key)
        self.use_sample = use_sample
        if use_sample:
            self.backend: BaseBackend = SampleBackend()
            self.logger.warning(
                "Using sample data backend. Provide a DATA_API_KEY to enable live data."
            )
        else:
            self.backend = PolygonBackend(config)

    def load_candidates(self, run_date: date, expanded: bool = False) -> List[CandidateMetrics]:
        """Return potential candidates."""
        return self.backend.load_candidates(run_date, expanded)


class BaseBackend:
    """Base backend interface."""

    def load_candidates(self, run_date: date, expanded: bool = False) -> List[CandidateMetrics]:
        raise NotImplementedError


class PolygonBackend(BaseBackend):
    """Polygon.io backed data provider."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.logger = setup_logger(__name__)
        if not config.api_key:
            raise ValueError("PolygonBackend requires an API key")
        try:
            import requests  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("requests is required for Polygon backend") from exc
        self._requests = requests
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {config.api_key}"})
        self._news_keywords = [
            "guidance",
            "restatement",
            "sec",
            "delisting",
            "fraud",
            "fda",
            "trial",
            "crl",
            "pdufa",
            "going concern",
        ]
        self._market_proxy = "SPY"
        self._max_trigger_universe = 250

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, allow_404: bool = False) -> Dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        delay = max(self.config.backoff_seconds, 1)
        for attempt in range(self.config.max_retries + 1):
            response = self.session.get(url, params=params, timeout=30)
            status = response.status_code
            if status == 404 and allow_404:
                return {}
            if status in {429, 500, 502, 503, 504}:
                sleep_for = delay * (2**attempt)
                time.sleep(sleep_for + random.uniform(0, 0.5))
                continue
            if status >= 400:
                self.logger.error("Polygon API error %s: %s", status, response.text)
                response.raise_for_status()
            try:
                return response.json()
            except ValueError as exc:  # pragma: no cover - invalid response
                raise RuntimeError("Failed to decode Polygon response") from exc
        raise RuntimeError("Polygon API request failed after retries")

    def load_candidates(self, run_date: date, expanded: bool = False) -> List[CandidateMetrics]:  # type: ignore[override]
        self.logger.info("Fetching Polygon data for %s (expanded=%s)", run_date, expanded)
        todays_data = self._grouped_agg(run_date)
        if not todays_data:
            self.logger.warning("No grouped aggregates returned for %s", run_date)
            return []
        prev_day = self._previous_trading_day(run_date)
        if prev_day is None:
            self.logger.warning("Unable to determine previous trading day for %s", run_date)
            return []
        prev_data = self._grouped_agg(prev_day)
        triggered = self._screen_for_triggers(todays_data, prev_data, run_date, expanded)
        candidates: List[CandidateMetrics] = []
        market_history = self._get_price_history(self._market_proxy, run_date - timedelta(days=365 * 5), run_date)
        for ticker, bar, trigger, reasons in triggered:
            try:
                metrics = self._build_candidate_metrics(
                    ticker,
                    bar,
                    prev_data.get(ticker),
                    run_date,
                    trigger,
                    reasons,
                    market_history,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.warning("Skipping %s due to data error: %s", ticker, exc)
                continue
            candidates.append(metrics)
        return candidates

    @in_memory_cache()
    def _grouped_agg(self, target_date: date) -> Dict[str, Dict[str, Any]]:
        path = f"/v2/aggs/grouped/locale/us/market/stocks/{target_date.isoformat()}"
        payload = self._get(path, params={"adjusted": "true"}, allow_404=True)
        results = {}
        for item in payload.get("results", []):
            ticker = item.get("T")
            if not ticker:
                continue
            results[ticker] = item
        return results

    def _previous_trading_day(self, run_date: date) -> Optional[date]:
        for offset in range(1, 7):
            candidate = run_date - timedelta(days=offset)
            if self._grouped_agg(candidate):
                return candidate
        return None

    def _screen_for_triggers(
        self,
        todays: Dict[str, Dict[str, Any]],
        previous: Dict[str, Dict[str, Any]],
        run_date: date,
        expanded: bool,
    ) -> List[Tuple[str, Dict[str, Any], str, List[str]]]:
        entries: List[Tuple[str, Dict[str, Any], float]] = []
        for ticker, bar in todays.items():
            prev = previous.get(ticker)
            prev_close = prev.get("c") if prev else None
            close = bar.get("c")
            if not prev_close or not close or prev_close <= 0:
                continue
            change = (close - prev_close) / prev_close
            entries.append((ticker, bar, change))
        entries.sort(key=lambda item: item[2])
        # Limit the universe we examine deeply for performance reasons.
        entries = entries[: self._max_trigger_universe]
        triggered: List[Tuple[str, Dict[str, Any], str, List[str]]] = []
        for ticker, bar, change in entries:
            trigger_name: Optional[str] = None
            reasons: List[str] = []
            if change <= -0.40:
                trigger_name = "1-day −40%"
                reasons.append(f"1d move {change:.2%}")
            distance: Optional[float] = None
            if trigger_name is None:
                threshold = -0.04 if expanded else -0.08
                if change <= threshold:
                    history_1y = self._get_price_history(
                        ticker, run_date - timedelta(days=370), run_date
                    )
                    distance = _distance_from_high(history_1y, 252) or 0.0
                    if distance <= -0.40:
                        trigger_name = "52w −40% + down day"
                        reasons.append(
                            f"52w drawdown {distance:.2%}, 1d {change:.2%}"
                        )
            if trigger_name is None:
                continue
            bar_copy = dict(bar)
            bar_copy["change"] = change
            if distance is not None:
                bar_copy["distance_52w"] = distance
            triggered.append((ticker, bar_copy, trigger_name, reasons))
        return triggered

    def _build_candidate_metrics(
        self,
        ticker: str,
        bar: Dict[str, Any],
        prev_bar: Optional[Dict[str, Any]],
        run_date: date,
        trigger: str,
        reasons: List[str],
        market_history: Sequence[PricePoint],
    ) -> CandidateMetrics:
        details = self._get_ticker_details(ticker)
        profile = self._get_company_profile(ticker)
        company_name = details.get("name") or profile.get("name") or ticker
        sector = profile.get("sector") or profile.get("sic_description") or "Unknown"
        industry = profile.get("industry") or profile.get("sic_description") or sector
        market_cap = details.get("market_cap")
        share_count = details.get("share_class_shares_outstanding")
        price = bar.get("c", 0.0)
        if market_cap is None and share_count:
            market_cap = price * share_count
        market_cap = market_cap or 0.0

        history_5y = self._get_price_history(ticker, run_date - timedelta(days=365 * 5 + 30), run_date)
        if not history_5y:
            raise RuntimeError(f"No price history for {ticker}")
        history_sorted = sorted(history_5y, key=lambda point: point.date)
        change_1d = bar.get("change")
        if change_1d is None and prev_bar and prev_bar.get("c"):
            prev_close = prev_bar.get("c")
            if prev_close:
                change_1d = (price - prev_close) / prev_close
        change_5d = _percent_change(history_sorted, 5)
        change_1m = _percent_change(history_sorted, 21)
        change_6m = _percent_change(history_sorted, 126)
        change_1y = _percent_change(history_sorted, 252)
        change_5y = _percent_change(history_sorted, 252 * 5)
        distance_52w = bar.get("distance_52w")
        if distance_52w is None:
            distance_52w = _distance_from_high(history_sorted, 252)
        distance_200, percent_to_200 = _distance_from_moving_average(history_sorted, 200)
        rsi_14 = _calculate_rsi(history_sorted)
        realized_vol = _calculate_realized_vol(history_sorted)
        avg_dollar_volume = _calculate_avg_dollar_volume(history_sorted)
        beta = _calculate_beta(history_sorted, market_history)

        financials_q = self._get_financials(ticker, timeframe="quarterly", limit=16)
        financials_a = self._get_financials(ticker, timeframe="annual", limit=6)
        fundamentals = self._compute_fundamentals(
            financials_q,
            financials_a,
            market_cap,
            price,
        )
        short_interest = self._get_short_interest(ticker)
        insider_buy = self._get_insider_activity(ticker)
        earnings_date, last_earnings = self._get_earnings_window(ticker, run_date)
        headline_flags, sec_delinquent, going_concern = self._collect_headline_flags(ticker, run_date)

        five_year_stats = self._build_five_year_stats(
            history_sorted,
            financials_a,
        )

        metrics = CandidateMetrics(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            industry=industry,
            market_cap=market_cap,
            price=price,
            open_price=bar.get("o", 0.0),
            high_price=bar.get("h", 0.0),
            low_price=bar.get("l", 0.0),
            change_1d=change_1d or 0.0,
            change_5d=change_5d or 0.0,
            change_1m=change_1m or 0.0,
            change_6m=change_6m or 0.0,
            change_1y=change_1y or 0.0,
            change_5y=change_5y,
            distance_from_52w_high=distance_52w or 0.0,
            distance_from_200dma=distance_200,
            percent_to_200dma=percent_to_200,
            rsi_14=rsi_14,
            beta=beta,
            realized_vol_30d=realized_vol,
            short_interest_percent=short_interest,
            avg_dollar_volume=avg_dollar_volume or 0.0,
            shares_outstanding_trend=fundamentals.get("shares_trend"),
            free_cash_flow_yield=fundamentals.get("fcf_yield"),
            ev_to_ebitda=fundamentals.get("ev_to_ebitda"),
            pe_ratio=fundamentals.get("pe"),
            pb_ratio=fundamentals.get("pb"),
            ps_ratio=fundamentals.get("ps"),
            gross_margin=fundamentals.get("gross_margin"),
            gross_margin_stability=fundamentals.get("gross_margin_stability"),
            operating_margin=fundamentals.get("operating_margin"),
            revenue_cagr_3y=fundamentals.get("revenue_cagr"),
            roic=fundamentals.get("roic"),
            net_debt_to_ebitda=fundamentals.get("net_debt_to_ebitda"),
            interest_coverage=fundamentals.get("interest_coverage"),
            operating_cash_flow_positive=fundamentals.get("operating_cash_flow_positive", False),
            insider_net_buy_percent=insider_buy,
            earnings_date=earnings_date,
            last_earnings_date=last_earnings,
            headline_flags=headline_flags,
            sec_delinquent=sec_delinquent,
            going_concern=going_concern,
            trading_halted=False,
            price_history_5y=history_sorted,
            five_year_stats=five_year_stats,
        )
        metrics.trigger = trigger
        metrics.triggered_conditions = reasons
        return metrics

    @in_memory_cache()
    def _get_ticker_details(self, ticker: str) -> Dict[str, Any]:
        payload = self._get(f"/v3/reference/tickers/{ticker}")
        return payload.get("results", {}) if payload else {}

    @in_memory_cache()
    def _get_company_profile(self, ticker: str) -> Dict[str, Any]:
        payload = self._get(f"/v1/meta/symbols/{ticker}/company", allow_404=True)
        return payload or {}

    @in_memory_cache()
    def _get_financials(
        self, ticker: str, timeframe: str, limit: int
    ) -> List[Dict[str, Any]]:
        params = {"ticker": ticker, "timeframe": timeframe, "limit": limit, "order": "desc"}
        payload = self._get("/v3/reference/financials", params=params, allow_404=True)
        results = payload.get("results", []) if payload else []
        return sorted(
            results,
            key=lambda item: item.get("calendar_date") or item.get("start_date") or "",
            reverse=True,
        )

    @in_memory_cache()
    def _get_price_history(
        self, ticker: str, start: date, end: date
    ) -> List[PricePoint]:
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 5000,
        }
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        payload = self._get(path, params=params, allow_404=True)
        points: List[PricePoint] = []
        for item in payload.get("results", []) if payload else []:
            timestamp = item.get("t")
            close = item.get("c")
            volume = item.get("v")
            if timestamp is None or close is None:
                continue
            bar_date = datetime.utcfromtimestamp(timestamp / 1000).date()
            points.append(PricePoint(date=bar_date, close=float(close), volume=float(volume) if volume else None))
        return points

    @in_memory_cache()
    def _get_short_interest(self, ticker: str) -> Optional[float]:
        payload = self._get(
            "/v3/reference/short_interest",
            params={"ticker": ticker, "limit": 1, "order": "desc"},
            allow_404=True,
        )
        results = payload.get("results") if payload else None
        if not results:
            return None
        latest = results[0]
        float_shares = latest.get("float_shares") or latest.get("free_float")
        short_interest = latest.get("short_interest") or latest.get("short_position")
        if not float_shares or not short_interest:
            return None
        if float_shares == 0:
            return None
        return float(short_interest) / float(float_shares)

    @in_memory_cache()
    def _get_insider_activity(self, ticker: str) -> Optional[float]:
        payload = self._get(
            "/v3/reference/insiders",
            params={"ticker": ticker, "limit": 20, "order": "desc"},
            allow_404=True,
        )
        results = payload.get("results") if payload else None
        if not results:
            return None
        net_shares = 0.0
        total_shares = 0.0
        for item in results:
            shares = item.get("shares_transacted") or 0.0
            if not shares:
                continue
            total_shares += abs(shares)
            direction = item.get("transaction_type", "").lower()
            if "buy" in direction:
                net_shares += shares
            elif "sell" in direction:
                net_shares -= abs(shares)
        if total_shares == 0:
            return None
        return net_shares / total_shares

    def _get_earnings_window(
        self, ticker: str, run_date: date
    ) -> Tuple[Optional[date], Optional[date]]:
        payload = self._get(
            "/v3/reference/earnings",
            params={"ticker": ticker, "limit": 10, "order": "desc"},
            allow_404=True,
        )
        results = payload.get("results") if payload else []
        upcoming: Optional[date] = None
        last: Optional[date] = None
        for item in results:
            report_date_str = item.get("report_date") or item.get("fiscal_period_end")
            if not report_date_str:
                continue
            report_date = datetime.fromisoformat(report_date_str).date()
            if report_date >= run_date:
                upcoming = report_date
            elif last is None or report_date > last:
                last = report_date
        return upcoming, last

    def _collect_headline_flags(
        self, ticker: str, run_date: date
    ) -> Tuple[List[str], bool, bool]:
        since = (run_date - timedelta(days=5)).isoformat()
        payload = self._get(
            "/v2/reference/news",
            params={"ticker": ticker, "published_utc.gte": since, "limit": 50},
            allow_404=True,
        )
        results = payload.get("results") if payload else []
        flags: List[str] = []
        sec_delinquent = False
        going_concern = False
        for item in results:
            headline = (item.get("title") or "").lower()
            for keyword in self._news_keywords:
                if keyword in headline:
                    flags.append(item.get("title", ""))
                    if "going concern" in headline:
                        going_concern = True
                    if "sec" in headline and "delin" in headline:
                        sec_delinquent = True
                    break
        unique_flags = list(dict.fromkeys(flags))
        return unique_flags, sec_delinquent, going_concern

    def _compute_fundamentals(
        self,
        financials_q: Sequence[Dict[str, Any]],
        financials_a: Sequence[Dict[str, Any]],
        market_cap: float,
        price: float,
    ) -> Dict[str, Any]:
        def extract(entry: Dict[str, Any], section: str, *candidates: str) -> Optional[float]:
            financials = entry.get("financials", {}) if entry else {}
            data = financials.get(section, {}) if isinstance(financials, dict) else {}
            for key in candidates:
                value = data.get(key)
                if value is None:
                    continue
                if isinstance(value, dict):
                    raw = value.get("value")
                    if raw is not None:
                        return float(raw)
                elif isinstance(value, (int, float)):
                    return float(value)
            return None

        def ttm_sum(section: str, *fields: str) -> Optional[float]:
            selected = financials_q[:4]
            if len(selected) < 4:
                return None
            totals: List[float] = []
            for entry in selected:
                value = extract(entry, section, *fields)
                if value is None:
                    return None
                totals.append(value)
            return sum(totals)

        revenue_ttm = ttm_sum("income_statement", "revenues", "revenue", "sales")
        gross_profit_ttm = ttm_sum("income_statement", "gross_profit")
        operating_income_ttm = ttm_sum("income_statement", "operating_income", "income_from_operations")
        ebitda_ttm = ttm_sum("income_statement", "ebitda")
        net_income_ttm = ttm_sum("income_statement", "net_income")
        interest_expense_ttm = ttm_sum("income_statement", "interest_expense")
        operating_cash_flow_ttm = ttm_sum(
            "cash_flow_statement", "net_cash_flow_from_operating_activities", "operating_cash_flow"
        )
        capex_ttm = ttm_sum("cash_flow_statement", "capital_expenditure", "capital_expenditures")
        cash_latest = extract(financials_q[0], "balance_sheet", "cash_and_cash_equivalents", "cash") if financials_q else None
        total_debt_latest = extract(
            financials_q[0],
            "balance_sheet",
            "total_debt",
            "long_term_debt",
            "short_term_debt",
        ) if financials_q else None
        shareholders_equity_latest = extract(
            financials_q[0],
            "balance_sheet",
            "shareholders_equity",
            "total_shareholder_equity",
        ) if financials_q else None
        total_assets_latest = extract(
            financials_q[0],
            "balance_sheet",
            "total_assets",
            "assets",
        ) if financials_q else None
        current_liabilities_latest = extract(
            financials_q[0],
            "balance_sheet",
            "current_liabilities",
        ) if financials_q else None
        shares_outstanding_latest = extract(
            financials_q[0],
            "income_statement",
            "weighted_average_shares_outstanding_basic",
            "weighted_average_shares_outstanding_diluted",
        ) if financials_q else None

        free_cash_flow_ttm = None
        if operating_cash_flow_ttm is not None and capex_ttm is not None:
            free_cash_flow_ttm = operating_cash_flow_ttm - capex_ttm
        fcf_yield = None
        if free_cash_flow_ttm is not None and market_cap > 0:
            fcf_yield = free_cash_flow_ttm / market_cap

        ev = None
        if market_cap and total_debt_latest is not None:
            cash_value = cash_latest or 0.0
            ev = market_cap + total_debt_latest - cash_value
        ev_to_ebitda = None
        if ev is not None and ebitda_ttm and ebitda_ttm != 0:
            ev_to_ebitda = ev / ebitda_ttm

        pe = None
        if net_income_ttm and net_income_ttm != 0 and shares_outstanding_latest:
            eps = net_income_ttm / shares_outstanding_latest
            if eps != 0:
                pe = price / eps

        pb = None
        if shareholders_equity_latest and shareholders_equity_latest != 0 and shares_outstanding_latest:
            book_value_per_share = shareholders_equity_latest / shares_outstanding_latest
            if book_value_per_share != 0:
                pb = price / book_value_per_share

        ps = None
        if revenue_ttm and revenue_ttm != 0:
            ps = market_cap / revenue_ttm

        gross_margin = None
        if gross_profit_ttm is not None and revenue_ttm:
            if revenue_ttm != 0:
                gross_margin = gross_profit_ttm / revenue_ttm

        operating_margin = None
        if operating_income_ttm is not None and revenue_ttm:
            if revenue_ttm != 0:
                operating_margin = operating_income_ttm / revenue_ttm

        interest_coverage = None
        if operating_income_ttm is not None and interest_expense_ttm:
            if interest_expense_ttm != 0:
                interest_coverage = operating_income_ttm / interest_expense_ttm

        net_debt_to_ebitda = None
        if total_debt_latest is not None and cash_latest is not None and ebitda_ttm and ebitda_ttm != 0:
            net_debt_to_ebitda = (total_debt_latest - cash_latest) / ebitda_ttm

        operating_cash_flow_positive = bool(operating_cash_flow_ttm and operating_cash_flow_ttm > 0)

        revenue_cagr = None
        if len(financials_a) >= 4:
            recent_revenue = extract(
                financials_a[0], "income_statement", "revenues", "revenue", "sales"
            )
            old_revenue = extract(
                financials_a[3], "income_statement", "revenues", "revenue", "sales"
            )
            if recent_revenue and old_revenue and old_revenue > 0:
                years = 3
                revenue_cagr = (recent_revenue / old_revenue) ** (1 / years) - 1

        gross_margin_values: List[float] = []
        for entry in financials_q[:12]:
            revenue = extract(entry, "income_statement", "revenues", "revenue", "sales")
            gross_profit = extract(entry, "income_statement", "gross_profit")
            if revenue and revenue != 0 and gross_profit is not None:
                gross_margin_values.append(gross_profit / revenue)
        gross_margin_stability = (
            statistics.stdev(gross_margin_values) if len(gross_margin_values) >= 2 else None
        )

        roic = None
        invested_capital = None
        if total_assets_latest is not None and current_liabilities_latest is not None:
            invested_capital = total_assets_latest - current_liabilities_latest
            if cash_latest is not None:
                invested_capital -= cash_latest
        if invested_capital and invested_capital != 0 and operating_income_ttm is not None:
            roic = operating_income_ttm / invested_capital

        shares_trend = None
        share_values: List[float] = []
        for entry in financials_q[:12]:
            value = extract(
                entry,
                "income_statement",
                "weighted_average_shares_outstanding_basic",
                "weighted_average_shares_outstanding_diluted",
            )
            if value is not None:
                share_values.append(value)
        if len(share_values) >= 2:
            shares_trend = (share_values[0] - share_values[-1]) / share_values[-1] if share_values[-1] else None

        return {
            "fcf_yield": fcf_yield,
            "ev_to_ebitda": ev_to_ebitda,
            "pe": pe,
            "pb": pb,
            "ps": ps,
            "gross_margin": gross_margin,
            "gross_margin_stability": gross_margin_stability,
            "operating_margin": operating_margin,
            "revenue_cagr": revenue_cagr,
            "roic": roic,
            "net_debt_to_ebitda": net_debt_to_ebitda,
            "interest_coverage": interest_coverage,
            "operating_cash_flow_positive": operating_cash_flow_positive,
            "shares_trend": shares_trend,
        }

    def _build_five_year_stats(
        self,
        history: Sequence[PricePoint],
        financials_a: Sequence[Dict[str, Any]],
    ) -> FiveYearStats:
        closes = sorted(history, key=lambda point: point.date)
        max_drawdown = _max_drawdown(closes)
        drawdown_200dma = _drawdown_from_200dma(closes)
        rsi_freq, forward_return = _rsi_signal_stats(closes)
        medians = self._historical_medians(closes, financials_a)
        return FiveYearStats(
            median_pe=medians.get("pe"),
            median_ev_to_ebitda=medians.get("ev_to_ebitda"),
            median_gross_margin=medians.get("gross_margin"),
            median_roic=medians.get("roic"),
            rsi_under_30_frequency=rsi_freq,
            median_forward_return_after_rsi=forward_return,
            max_drawdown_pct=max_drawdown,
            drawdown_from_200dma_pct=drawdown_200dma,
        )

    def _historical_medians(
        self,
        history: Sequence[PricePoint],
        financials_a: Sequence[Dict[str, Any]],
    ) -> Dict[str, Optional[float]]:
        closes_map = {point.date: point.close for point in history}

        def extract(entry: Dict[str, Any], section: str, *candidates: str) -> Optional[float]:
            financials = entry.get("financials", {}) if entry else {}
            data = financials.get(section, {}) if isinstance(financials, dict) else {}
            for key in candidates:
                value = data.get(key)
                if value is None:
                    continue
                if isinstance(value, dict):
                    raw = value.get("value")
                    if raw is not None:
                        return float(raw)
                elif isinstance(value, (int, float)):
                    return float(value)
            return None

        pe_values: List[float] = []
        ev_ebitda_values: List[float] = []
        gross_margin_values: List[float] = []
        roic_values: List[float] = []

        for entry in financials_a:
            end_date_str = entry.get("calendar_date") or entry.get("start_date")
            if not end_date_str:
                continue
            end_date = datetime.fromisoformat(end_date_str).date()
            close = closes_map.get(end_date)
            if close is None and history:
                closest = min(history, key=lambda point: abs((point.date - end_date).days))
                close = closest.close
            if close is None:
                continue

            revenue = extract(entry, "income_statement", "revenues", "revenue", "sales")
            gross_profit = extract(entry, "income_statement", "gross_profit")
            operating_income = extract(entry, "income_statement", "operating_income", "income_from_operations")
            ebitda = extract(entry, "income_statement", "ebitda")
            net_income = extract(entry, "income_statement", "net_income")
            shares = extract(
                entry,
                "income_statement",
                "weighted_average_shares_outstanding_basic",
                "weighted_average_shares_outstanding_diluted",
            )
            cash = extract(entry, "balance_sheet", "cash_and_cash_equivalents", "cash")
            total_debt = extract(entry, "balance_sheet", "total_debt", "long_term_debt", "short_term_debt")
            assets = extract(entry, "balance_sheet", "total_assets", "assets")
            current_liabilities = extract(entry, "balance_sheet", "current_liabilities")

            if shares and net_income:
                eps = net_income / shares if shares else None
                if eps and eps != 0:
                    pe_values.append(close / eps)
            if shares and total_debt is not None and ebitda and ebitda != 0:
                market_cap = close * shares
                ev = market_cap + total_debt - (cash or 0.0)
                ev_ebitda_values.append(ev / ebitda)
            if revenue and revenue != 0 and gross_profit is not None:
                gross_margin_values.append(gross_profit / revenue)
            if (
                operating_income is not None
                and assets is not None
                and current_liabilities is not None
            ):
                invested_capital = assets - current_liabilities - (cash or 0.0)
                if invested_capital:
                    roic_values.append(operating_income / invested_capital)

        medians: Dict[str, Optional[float]] = {
            "pe": statistics.median(pe_values) if pe_values else None,
            "ev_to_ebitda": statistics.median(ev_ebitda_values) if ev_ebitda_values else None,
            "gross_margin": statistics.median(gross_margin_values) if gross_margin_values else None,
            "roic": statistics.median(roic_values) if roic_values else None,
        }
        return medians


class SampleBackend(BaseBackend):
    """Generate deterministic sample data suitable for tests and dry runs."""

    def __init__(self) -> None:
        self.logger = setup_logger(__name__)
        self._seed = 42

    def load_candidates(self, run_date: date, expanded: bool = False) -> List[CandidateMetrics]:  # type: ignore[override]
        rng = random.Random(self._seed)
        tickers = [
            ("ACME", "Acme Industrial", "Industrials"),
            ("BIOX", "Biovex Labs", "Health Care"),
            ("CTEC", "CyberTech Corp", "Information Technology"),
            ("DRIL", "Drillers United", "Energy"),
            ("EVGO", "Evergreen Goods", "Consumer Staples"),
            ("FINS", "FinServe Group", "Financials"),
            ("GLXY", "Galaxy Retail", "Consumer Discretionary"),
            ("HOMR", "Home Repair Co", "Industrials"),
            ("ICRX", "InnovaCare Rx", "Health Care"),
            ("JETT", "Jetsetter Airlines", "Industrials"),
            ("KART", "Kart Logistics", "Industrials"),
            ("LUMN", "Lumina Media", "Communication Services"),
        ]
        results: List[CandidateMetrics] = []
        base_price = 40.0
        for idx, (ticker, name, sector) in enumerate(tickers):
            price = base_price * (1 - 0.03 * idx)
            change_1d = -0.41 + 0.01 * (idx % 3)
            distance_52w = -0.42 - 0.02 * (idx % 4)
            if idx % 4 == 0:
                change_1d = -0.45
            elif idx % 4 == 1:
                change_1d = -0.12
            elif idx % 4 == 2:
                change_1d = -0.09
            else:
                change_1d = -0.5
            revenue_cagr = 0.1 + 0.01 * (idx % 5)
            rsi = 22 + (idx % 6)
            distance_200dma = -0.25 - 0.01 * (idx % 7)
            change_1m = -0.2 + 0.02 * (idx % 4)
            change_6m = -0.35 + 0.015 * (idx % 5)
            change_1y = -0.5 + 0.02 * (idx % 6)
            change_5d = -0.12 + 0.01 * (idx % 5)
            change_5y = 0.2 - 0.05 * (idx % 3)
            pe = 8 + idx * 0.5
            pb = 0.9 + 0.05 * (idx % 5)
            ev_ebitda = 5 + 0.4 * (idx % 6)
            roic = 0.12 - 0.005 * (idx % 4)
            gross_margin = 0.35 - 0.01 * (idx % 4)
            gm_stability = 0.04 + 0.005 * (idx % 3)
            net_debt = 1.5 + 0.2 * (idx % 5)
            interest_cov = 5 + 0.5 * (idx % 4)
            adv = 8_000_000 + 500_000 * (idx % 5)
            fcf_yield = 0.12 + 0.01 * (idx % 4)
            beta = 1.1 - 0.1 * (idx % 5)
            vol_30d = 0.55 + 0.02 * (idx % 4)
            short_interest = 0.05 + 0.005 * (idx % 5)
            insider_buy = 0.01 * (idx % 3)
            share_trend = -0.01 * (idx % 4)
            history = self._generate_price_history(price, rng)
            stats = FiveYearStats(
                median_pe=15.0,
                median_ev_to_ebitda=8.5,
                median_gross_margin=0.4,
                median_roic=0.13,
                rsi_under_30_frequency=0.18,
                median_forward_return_after_rsi=0.045,
                max_drawdown_pct=-0.62,
                drawdown_from_200dma_pct=-0.28,
            )
            earnings_date = run_date + timedelta(days=7 + idx)
            metrics = CandidateMetrics(
                ticker=ticker,
                company_name=name,
                sector=sector,
                industry=f"Industry {idx % 6}",
                market_cap=800_000_000 + 50_000_000 * idx,
                price=price,
                open_price=price * 1.05,
                high_price=price * 1.08,
                low_price=price * 0.92,
                change_1d=change_1d,
                change_5d=change_5d,
                change_1m=change_1m,
                change_6m=change_6m,
                change_1y=change_1y,
                change_5y=change_5y,
                distance_from_52w_high=distance_52w,
                distance_from_200dma=distance_200dma,
                percent_to_200dma=abs(distance_200dma),
                rsi_14=rsi,
                beta=beta,
                realized_vol_30d=vol_30d,
                short_interest_percent=short_interest,
                avg_dollar_volume=adv,
                shares_outstanding_trend=share_trend,
                free_cash_flow_yield=fcf_yield,
                ev_to_ebitda=ev_ebitda,
                pe_ratio=pe,
                pb_ratio=pb,
                ps_ratio=1.2 + 0.05 * (idx % 4),
                gross_margin=gross_margin,
                gross_margin_stability=gm_stability,
                operating_margin=0.18 - 0.01 * (idx % 4),
                revenue_cagr_3y=revenue_cagr,
                roic=roic,
                net_debt_to_ebitda=net_debt,
                interest_coverage=interest_cov,
                operating_cash_flow_positive=True,
                insider_net_buy_percent=insider_buy,
                earnings_date=earnings_date,
                last_earnings_date=run_date - timedelta(days=50),
                headline_flags=[] if idx % 5 else ["Guidance cut"],
                sec_delinquent=False,
                going_concern=False,
                trading_halted=False,
                price_history_5y=history,
                five_year_stats=stats,
            )
            if idx % 5 == 0:
                metrics.add_flag("Guidance")
            results.append(metrics)
        if not expanded:
            return results
        # When expanded criteria requested, adjust drawdown thresholds to include milder drops.
        for metrics in results:
            if metrics.change_1d > -0.04:
                metrics.change_1d = -0.05
            if metrics.distance_from_52w_high < -0.2:
                continue
            metrics.distance_from_52w_high -= 0.2
        return results

    def _generate_price_history(
        self, last_price: float, rng: random.Random, days: int = 252 * 5
    ) -> Sequence[PricePoint]:
        prices: List[PricePoint] = []
        current_price = last_price
        start_date = datetime.utcnow().date() - timedelta(days=days)
        for offset in range(days):
            day = start_date + timedelta(days=offset)
            if day.weekday() >= 5:
                continue
            drift = rng.gauss(-0.0002, 0.02)
            current_price *= max(0.1, 1 + drift)
            prices.append(PricePoint(date=day, close=round(current_price, 2)))
        return prices[-750:]
