from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests

from ..logging_utils import get_logger
from ..utils.time import to_datestr
from ..utils.cache import DiskCache


logger = get_logger(__name__)


def _retry_request(session: requests.Session, method: str, url: str, *, params=None, headers=None, timeout=30, max_retries=5, base_delay=1.0, max_delay=8.0) -> requests.Response:
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = session.request(method, url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code}")
            return resp
        except Exception as e:
            if attempt >= max_retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= (0.5 + np.random.rand())  # jitter
            logger.debug(f"Retry {attempt} for {url} after {delay:.2f}s: {e}")
            import time
            time.sleep(delay)


@dataclass
class GroupedBar:
    ticker: str
    open: float
    close: float
    high: float
    low: float
    volume: float


class PolygonClient:
    def __init__(self, api_key: str, cfg: dict):
        self.api_key = api_key
        self.cfg = cfg
        self.session = requests.Session()
        self.base = "https://api.polygon.io"
        cache_dir = cfg.get("run", {}).get("cache_dir", ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = DiskCache(cache_dir)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def market_status(self) -> dict:
        url = f"{self.base}/v1/marketstatus/now"
        key = f"GET:{url}"
        cached = self.cache.get(key, max_age_sec=60)
        if cached:
            return cached
        r = _retry_request(self.session, "GET", url, headers=self._headers(), timeout=self.cfg["run"]["http_timeout_sec"], max_retries=self.cfg["run"]["max_retries"])
        data = r.json()
        self.cache.set(key, data)
        return data

    def us_universe(self) -> set[str]:
        # Build & cache universe of tickers on NYSE/NASDAQ/AMEX, common stock (CS)
        key = "UNIVERSE_V1"
        cached = self.cache.get(key, max_age_sec=24 * 3600)
        if cached:
            return set(cached["tickers"])  # type: ignore

        exchanges = self.cfg["universe"]["exchanges"]
        include_types = set(self.cfg["universe"]["include_types"])
        tickers: set[str] = set()
        for exch in exchanges:
            url = f"{self.base}/v3/reference/tickers"
            params = {
                "market": "stocks",
                "exchange": exch,
                "active": "true",
                "limit": 1000,
            }
            while True:
                r = _retry_request(self.session, "GET", url, params=params, headers=self._headers(), timeout=self.cfg["run"]["http_timeout_sec"], max_retries=self.cfg["run"]["max_retries"])
                j = r.json()
                for it in j.get("results", []):
                    t = it.get("ticker")
                    typ = it.get("type")
                    if not t or not typ:
                        continue
                    if typ not in include_types:
                        continue
                    # exclude ADRs, ETFs, PFDs, SPAC-like by name heuristics
                    name = (it.get("name") or "").lower()
                    if any(x in name for x in ["acquisition", "spac", "preferred"]):
                        continue
                    tickers.add(t)
                cursor = j.get("next_url")
                if cursor:
                    # next_url already includes query; but API requires auth key
                    url = cursor
                    params = None
                else:
                    break
        self.cache.set(key, {"tickers": sorted(list(tickers))})
        return tickers

    def grouped_daily(self, day: datetime) -> list[GroupedBar]:
        d = to_datestr(day)
        url = f"{self.base}/v2/aggs/grouped/locale/us/market/stocks/{d}"
        key = f"GET:{url}"
        cached = self.cache.get(key, max_age_sec=3600)
        if cached:
            raw = cached
        else:
            r = _retry_request(self.session, "GET", url, params={"adjusted": "true"}, headers=self._headers(), timeout=self.cfg["run"]["http_timeout_sec"], max_retries=self.cfg["run"]["max_retries"])
            raw = r.json()
            self.cache.set(key, raw)
        results = []
        for it in raw.get("results", []) or []:
            try:
                t = it.get("T")
                results.append(GroupedBar(
                    ticker=t,
                    open=float(it.get("o")),
                    close=float(it.get("c")),
                    high=float(it.get("h")),
                    low=float(it.get("l")),
                    volume=float(it.get("v")),
                ))
            except Exception:
                continue
        return results

    def one_year_bars(self, ticker: str, end_day: datetime) -> list[dict]:
        start = end_day - timedelta(days=365 + 10)
        url = f"{self.base}/v2/aggs/ticker/{ticker}/range/1/day/{to_datestr(start)}/{to_datestr(end_day)}"
        key = f"GET:{url}"
        cached = self.cache.get(key, max_age_sec=24 * 3600)
        if cached:
            return cached.get("results", [])
        r = _retry_request(self.session, "GET", url, params={"adjusted": "true", "limit": 50000}, headers=self._headers(), timeout=self.cfg["run"]["http_timeout_sec"], max_retries=self.cfg["run"]["max_retries"])
        data = r.json()
        self.cache.set(key, data)
        return data.get("results", [])

    def n_year_bars(self, ticker: str, end_day: datetime, years: int = 5) -> list[dict]:
        start = end_day - timedelta(days=365 * years + 10)
        url = f"{self.base}/v2/aggs/ticker/{ticker}/range/1/day/{to_datestr(start)}/{to_datestr(end_day)}"
        key = f"GET:{url}"
        cached = self.cache.get(key, max_age_sec=24 * 3600)
        if cached:
            return cached.get("results", [])
        r = _retry_request(self.session, "GET", url, params={"adjusted": "true", "limit": 50000}, headers=self._headers(), timeout=self.cfg["run"]["http_timeout_sec"], max_retries=self.cfg["run"]["max_retries"])
        data = r.json()
        self.cache.set(key, data)
        return data.get("results", [])

    def ticker_details(self, ticker: str) -> dict:
        url = f"{self.base}/v3/reference/tickers/{ticker}"
        key = f"GET:{url}"
        cached = self.cache.get(key, max_age_sec=24 * 3600)
        if cached:
            return cached
        r = _retry_request(self.session, "GET", url, headers=self._headers(), timeout=self.cfg["run"]["http_timeout_sec"], max_retries=self.cfg["run"]["max_retries"])
        data = r.json().get("results", {})
        self.cache.set(key, data)
        return data

    def financials(self, ticker: str, timeframe: str = "TTM", limit: int = 12) -> list[dict]:
        # timeframe: TTM, quarterly, annual
        url = f"{self.base}/vX/reference/financials"
        params = {"ticker": ticker, "timeframe": timeframe, "limit": limit}
        key = f"GET:{url}:{json.dumps(params, sort_keys=True)}"
        cached = self.cache.get(key, max_age_sec=24 * 3600)
        if cached:
            return cached.get("results", [])
        r = _retry_request(self.session, "GET", url, params=params, headers=self._headers(), timeout=self.cfg["run"]["http_timeout_sec"], max_retries=self.cfg["run"]["max_retries"])
        data = r.json()
        self.cache.set(key, data)
        return data.get("results", [])

    def news(self, ticker: str, from_day: datetime, to_day: datetime) -> list[dict]:
        url = f"{self.base}/v2/reference/news"
        params = {
            "ticker": ticker,
            "published_utc.gte": f"{to_datestr(from_day)} 00:00:00",
            "published_utc.lte": f"{to_datestr(to_day)} 23:59:59",
            "limit": 50,
            "order": "desc",
        }
        key = f"GET:{url}:{json.dumps(params, sort_keys=True)}"
        cached = self.cache.get(key, max_age_sec=3600)
        if cached:
            return cached.get("results", [])
        r = _retry_request(self.session, "GET", url, params=params, headers=self._headers(), timeout=self.cfg["run"]["http_timeout_sec"], max_retries=self.cfg["run"]["max_retries"])
        data = r.json()
        self.cache.set(key, data)
        return data.get("results", [])

    # ---- Pipeline helpers ----

    def find_candidates(self, day: datetime, grouped: list[GroupedBar], universe: set[str]) -> tuple[list[dict], bool]:
        if not grouped:
            return [], False
        min_price = float(self.cfg["universe"]["min_price"])
        one_day_thr = float(self.cfg["triggers"]["one_day_drop_pct"])  # -40
        deep_dd = float(self.cfg["triggers"]["deep_dd_day_move"])      # -8
        dd_52w = float(self.cfg["triggers"]["deep_dd_52w"])            # 40

        losers: list[dict] = []
        for g in grouped:
            if g.ticker not in universe:
                continue
            if g.close < min_price:
                continue
            if g.open <= 0:
                continue
            pct = (g.close - g.open) / g.open * 100.0
            trig = None
            if pct <= one_day_thr:
                trig = "1-day −40%"
            elif pct <= deep_dd:
                trig = "52w −40% + down day"
            if trig:
                losers.append({
                    "ticker": g.ticker,
                    "pct": pct,
                    "trigger": trig,
                    "close": g.close,
                })

        # For those with the second trigger, validate 52w drawdown
        cands: list[dict] = []
        for it in losers:
            if it["trigger"] == "1-day −40%":
                cands.append(it)
            else:
                bars = self.one_year_bars(it["ticker"], day)
                if not bars:
                    continue
                max_close = max(b.get("c", 0) for b in bars)
                if max_close <= 0:
                    continue
                dd = (max_close - it["close"]) / max_close * 100.0
                if dd >= dd_52w:
                    cands.append(it)

        expanded = False
        if len(cands) < 5:
            # Expand daily threshold to -4%
            deep_dd = float(self.cfg["triggers"]["expand_deep_dd_day_move"])  # -4
            for g in grouped:
                if g.ticker not in universe or g.close < min_price or g.open <= 0:
                    continue
                pct = (g.close - g.open) / g.open * 100.0
                if pct > deep_dd:
                    continue
                bars = self.one_year_bars(g.ticker, day)
                if not bars:
                    continue
                max_close = max(b.get("c", 0) for b in bars)
                if max_close <= 0:
                    continue
                dd = (max_close - g.close) / max_close * 100.0
                if dd >= dd_52w:
                    cands.append({
                        "ticker": g.ticker,
                        "pct": pct,
                        "trigger": "52w −40% + down day",
                        "close": g.close,
                    })
            expanded = True

        # Deduplicate
        seen = set()
        out = []
        for it in cands:
            t = it["ticker"]
            if t in seen:
                continue
            seen.add(t)
            out.append(it)
        return out, expanded

    def _thirty_day_adv_usd(self, ticker: str, end_day: datetime) -> float:
        start = end_day - timedelta(days=45)
        url = f"{self.base}/v2/aggs/ticker/{ticker}/range/1/day/{to_datestr(start)}/{to_datestr(end_day)}"
        key = f"GET:{url}"
        cached = self.cache.get(key, max_age_sec=24 * 3600)
        if cached:
            results = cached.get("results", [])
        else:
            r = _retry_request(self.session, "GET", url, params={"adjusted": "true", "limit": 50000}, headers=self._headers(), timeout=self.cfg["run"]["http_timeout_sec"], max_retries=self.cfg["run"]["max_retries"])
            data = r.json()
            self.cache.set(key, data)
            results = data.get("results", [])
        closes = [it.get("c", 0.0) for it in results][-30:]
        vols = [it.get("v", 0.0) for it in results][-30:]
        if not closes or not vols:
            return 0.0
        dv = [c * v for c, v in zip(closes, vols)]
        return float(np.mean(dv))

    def enrich_candidates(self, day: datetime, candidates: list[dict]) -> list:
        out = []
        for it in candidates:
            t = it["ticker"]
            details = self.ticker_details(t)
            name = details.get("name")
            mcap = details.get("market_cap") or 0.0
            sector = details.get("sic_description") or details.get("share_class_figi") or None
            adv = self._thirty_day_adv_usd(t, day)
            # fundamentals
            ttm = self.financials(t, timeframe="TTM", limit=1)
            qtr = self.financials(t, timeframe="quarterly", limit=12)
            ann = self.financials(t, timeframe="annual", limit=5)
            f = self._extract_fundamentals(ttm, qtr, ann, mcap)

            # price series for momentum / beta later if needed lazily
            prices = {
                "today_pct": it.get("pct"),
            }

            # news flags (last 3 trading days)
            news = self.news(t, day - timedelta(days=7), day)
            flags = self._news_flags(news)

            out.append({
                "ticker": t,
                "company": name,
                "sector": sector,
                "trigger": it["trigger"],
                "prices": prices,
                "fundamentals": f,
                "risk": {
                    "adv30_usd": adv,
                },
                "metrics": {},
                "flags": flags,
            })
        return out

    def _extract_fundamentals(self, ttm: list[dict], qtr: list[dict], ann: list[dict], mcap: float) -> dict:
        f = {
            "market_cap": mcap,
        }
        ttm0 = ttm[0] if ttm else {}
        fin = (ttm0.get("financials") or {}) if isinstance(ttm0, dict) else {}
        isec = fin.get("income_statement") or {}
        cf = fin.get("cash_flow_statement") or {}
        bs = fin.get("balance_sheet") or {}
        ratios = fin.get("ratios") or {}

        # Core metrics
        ebitda = isec.get("ebitda") or ratios.get("ebitda")
        ev = ratios.get("enterprise_value") or None
        pe = ratios.get("price_to_earnings") or None
        pb = ratios.get("price_to_book_value") or None
        ps = ratios.get("price_to_sales") or None
        fcf_ttm = cf.get("free_cash_flow") or cf.get("net_cash_flow_from_operations")
        ocf_ttm = cf.get("net_cash_flow_from_operations")
        gross_profit = isec.get("gross_profit")
        revenue = isec.get("revenues") or isec.get("revenue")
        operating_income = isec.get("operating_income") or isec.get("operating_income_loss")
        interest_expense = isec.get("interest_expense") or 0.0
        net_debt = (bs.get("total_debt") or 0.0) - (bs.get("cash_and_cash_equivalents") or 0.0)

        f.update({
            "ebitda": _safe_float(ebitda),
            "enterprise_value": _safe_float(ev),
            "pe": _safe_float(pe),
            "pb": _safe_float(pb),
            "ps": _safe_float(ps),
            "fcf_ttm": _safe_float(fcf_ttm),
            "ocf_ttm": _safe_float(ocf_ttm),
            "gross_margin": _ratio(_safe_float(gross_profit), _safe_float(revenue)),
            "operating_margin": _ratio(_safe_float(operating_income), _safe_float(revenue)),
            "revenue_ttm": _safe_float(revenue),
            "operating_income_ttm": _safe_float(operating_income),
            "interest_expense_ttm": _safe_float(interest_expense),
            "net_debt": _safe_float(net_debt),
        })

        # ROA/ROIC approximation
        net_income = isec.get("net_income") or isec.get("net_income_loss")
        total_assets = bs.get("assets") or bs.get("total_assets")
        f["roa"] = _ratio(_safe_float(net_income), _safe_float(total_assets))

        # EBITDA-based ratios
        if f["ebitda"] and f["enterprise_value"]:
            f["ev_to_ebitda"] = f["enterprise_value"] / f["ebitda"] if f["ebitda"] else None
        else:
            f["ev_to_ebitda"] = None

        if mcap and f.get("fcf_ttm") is not None:
            f["fcf_yield"] = f["fcf_ttm"] / mcap
        else:
            f["fcf_yield"] = None

        # Gross margin stability (std of last 12 qtrs)
        gm = []
        for q in qtr:
            finq = (q.get("financials") or {}) if isinstance(q, dict) else {}
            isq = finq.get("income_statement") or {}
            gp = _safe_float(isq.get("gross_profit"))
            rev = _safe_float(isq.get("revenues") or isq.get("revenue"))
            gm.append(_ratio(gp, rev))
        gm = [x for x in gm if x is not None]
        f["gross_margin_std_12q"] = float(np.nanstd(gm)) if gm else None

        # Revenue 3y CAGR (annual)
        revs = []
        for a in ann:
            fina = (a.get("financials") or {}) if isinstance(a, dict) else {}
            isa = fina.get("income_statement") or {}
            revs.append(_safe_float(isa.get("revenues") or isa.get("revenue")))
        revs = [r for r in revs if r is not None]
        if len(revs) >= 4 and revs[-4] and revs[-1]:
            f["rev_cagr_3y"] = (revs[-1] / revs[-4]) ** (1/3) - 1
        else:
            f["rev_cagr_3y"] = None

        # Coverage and leverage
        ebit = _safe_float(operating_income)
        int_exp = abs(_safe_float(interest_expense))
        ebitda = f.get("ebitda") or 0.0
        if int_exp > 0 and ebit is not None:
            f["interest_coverage"] = ebit / int_exp
        else:
            f["interest_coverage"] = None
        if ebitda:
            f["net_debt_to_ebitda"] = (f.get("net_debt") or 0.0) / ebitda if ebitda else None
        else:
            f["net_debt_to_ebitda"] = None

        return f

    def _news_flags(self, news: list[dict]) -> list[str]:
        flags = []
        keywords = [k.lower() for k in self.cfg.get("risk_keywords", [])]
        for n in news:
            title = (n.get("title") or "").lower()
            desc = (n.get("description") or "").lower()
            if any(k in title or k in desc for k in keywords):
                flags.append("headline_flag")
                break
        return flags

    def apply_filters(self, enriched: list[dict]) -> tuple[list[dict], dict]:
        thr = self.cfg["universe"]
        flt = self.cfg["filters"]
        min_mcap = float(thr["min_market_cap"])
        min_adv = float(thr["min_adv_usd_30d"])
        gm_min = float(flt["gross_margin_min"])
        allow_sectors = set(flt.get("allow_low_gm_sectors", []))
        earnings_buffer = int(flt.get("earnings_buffer_days", 2))

        rejects = {"mcap": 0, "adv": 0, "ocf": 0, "leverage": 0, "coverage": 0, "margin": 0, "flags": 0}
        out = []
        for c in enriched:
            f = c["fundamentals"]
            # Market cap
            if (f.get("market_cap") or 0) < min_mcap:
                rejects["mcap"] += 1
                continue
            # Liquidity
            if (c.get("risk", {}).get("adv30_usd") or 0) < min_adv:
                rejects["adv"] += 1
                continue
            # Earnings window (not implemented → treat as Unknown, do not reject)
            # OCF positive
            if self.cfg["filters"].get("require_positive_ocf_ttm", True):
                ocf = f.get("ocf_ttm")
                if ocf is None or ocf <= 0:
                    rejects["ocf"] += 1
                    continue
            # Leverage
            nde = f.get("net_debt_to_ebitda")
            if nde is not None and nde > float(self.cfg["filters"]["net_debt_to_ebitda_max"]):
                rejects["leverage"] += 1
                continue
            # Coverage
            cov = f.get("interest_coverage")
            if cov is not None and cov < float(self.cfg["filters"]["interest_coverage_min"]):
                rejects["coverage"] += 1
                continue
            # Gross margin
            gm = f.get("gross_margin")
            sector = c.get("sector") or ""
            if gm is not None and gm < gm_min and not any(s in sector for s in allow_sectors):
                rejects["margin"] += 1
                continue
            # Headline flags within last 3 trading days
            if any(flag == "headline_flag" for flag in c.get("flags", [])):
                rejects["flags"] += 1
                continue

            out.append(c)
        return out, rejects

    def price_returns(self, ticker: str, end_day: datetime) -> dict:
        bars = self.n_year_bars(ticker, end_day, years=5)
        closes = [b.get("c", 0.0) for b in bars]
        if not closes:
            return {}
        arr = np.array(closes, dtype=float)
        out = {}
        def pct_return(days: int) -> Optional[float]:
            if len(arr) <= days:
                return None
            start = arr[-days-1]
            if start <= 0:
                return None
            return (arr[-1] - start) / start * 100.0
        out["pct_5d"] = pct_return(5)
        out["pct_1m"] = pct_return(21)
        out["pct_6m"] = pct_return(126)
        out["pct_1y"] = pct_return(252)

        # 200DMA distance
        if len(arr) >= 200:
            dma200 = float(np.nanmean(arr[-200:]))
            if dma200 > 0:
                out["dist_to_200dma"] = (arr[-1] - dma200) / dma200
        # RSI14
        out["rsi14"] = _rsi(arr)
        # Realized vol 30-day
        out["vol30"] = _realized_vol(arr, 30)
        # Chart series
        out["series_5y"] = closes
        return out

    def chart_url_5y(self, ticker: str) -> str:
        # Build a QuickChart URL for 5-year closes (downsample to reduce URL size)
        series = self.cache.get(f"SERIES5Y:{ticker}")
        if not series:
            # Series is stored in price_returns; fallback minimal
            return f"https://quickchart.io/chart?c={{type:'line',data:{'{'}labels:[],datasets:[{{label:'{ticker}',data:[]}}]{'}'}}&w=800&h=300"
        data = series.get("closes", [])
        if len(data) > 800:
            step = len(data) // 800
            data = data[::step]
        import json as _json
        config = {
            "type": "line",
            "data": {"labels": ["" for _ in data], "datasets": [{"label": ticker, "data": data}]},
            "options": {"legend": {"display": False}, "scales": {"yAxes": [{"ticks": {"callback": ""}}]},},
        }
        return f"https://quickchart.io/chart?c={requests.utils.quote(json.dumps(config))}&w=800&h=300"

    def background_blurb(self, c: dict) -> str:
        name = c.get("company") or c.get("ticker")
        trig = c.get("trigger")
        f = c.get("fundamentals", {})
        pe = f.get("pe")
        ev_ebitda = f.get("ev_to_ebitda")
        gm = f.get("gross_margin")
        flags = c.get("flags", [])
        notes = []
        notes.append(f"{name} triggered {trig}. ")
        if pe is not None:
            notes.append(f"TTM P/E ~ {pe:.1f}. ")
        if ev_ebitda is not None:
            notes.append(f"EV/EBITDA ~ {ev_ebitda:.1f}. ")
        if gm is not None:
            notes.append(f"Gross margin ~ {gm*100:.0f}%. ")
        if flags:
            notes.append("Recent headlines suggest elevated risk.")
        else:
            notes.append("No major headline flags detected.")
        notes.append("Consider liquidity, event risk, and thesis validation before sizing.")
        return " ".join(notes)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _rsi(arr: np.ndarray) -> Optional[float]:
    if arr.size < 15:
        return None
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    n = 14
    if gains.size < n:
        return None
    avg_gain = np.mean(gains[-n:])
    avg_loss = np.mean(losses[-n:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def _realized_vol(arr: np.ndarray, window: int) -> Optional[float]:
    if arr.size <= window:
        return None
    rets = np.diff(np.log(arr + 1e-9))
    w = rets[-window:]
    if w.size == 0:
        return None
    return float(np.std(w) * np.sqrt(252) * 100)

