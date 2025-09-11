from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .logging_utils import get_logger
from .providers.polygon_client import PolygonClient
from .scoring import ScoreEngine
from .stats import compute_rsi, realized_vol_pct, beta_vs_benchmark
from .notion.api import NotionAPI
from .notion.format import build_daily_page_blocks, ensure_database_schema, build_db_row_properties
from .utils.time import to_datestr, us_trading_day_for
from .utils.cache import RunLog


logger = get_logger(__name__)


@dataclass
class Candidate:
    ticker: str
    company: Optional[str]
    sector: Optional[str]
    trigger: str
    prices: Dict[str, Any]
    fundamentals: Dict[str, Any]
    risk: Dict[str, Any]
    metrics: Dict[str, Any]
    flags: List[str]


def run_daily(cfg: dict, secrets: dict, when: datetime, dry_run: bool) -> None:
    runs_dir = cfg["run"].get("runs_dir", "runs")
    os.makedirs(runs_dir, exist_ok=True)
    run_day = us_trading_day_for(when)
    day_str = to_datestr(run_day)
    logger.info(f"Run date: {day_str}")

    # Initialize clients
    data_client = PolygonClient(api_key=secrets["DATA_API_KEY"], cfg=cfg)
    notion = NotionAPI(api_key=secrets["NOTION_API_KEY"], db_id=secrets["NOTION_DATABASE_ID"], cfg=cfg)

    # Market status / holiday note
    market_info = data_client.market_status()
    market_closed = market_info.get("market", "closed") != "open"

    # Universe and grouped daily bars
    universe = data_client.us_universe()
    logger.info(f"Universe tickers: {len(universe)}")

    grouped = data_client.grouped_daily(run_day)
    if not grouped:
        logger.warning("No grouped data, likely market holiday/weekend.")

    # Apply triggers to get candidates
    candidates, expanded = data_client.find_candidates(run_day, grouped, universe)
    logger.info(f"Triggered candidates: {len(candidates)} (expanded={expanded})")

    # Early exit if nothing triggers: still create page with note
    if not candidates:
        logger.info("No candidates; will create Notion page with note and top-closest by score from losers.")

    # Enrich candidates with fundamentals, adv, dd, etc.
    enriched: List[Candidate] = data_client.enrich_candidates(run_day, candidates)

    # Apply survivability filters
    filtered, rejects = data_client.apply_filters(enriched)
    logger.info(f"Filtered candidates: {len(filtered)}; rejects: {json.dumps(rejects)[:300]}...")

    # Compute price-derived metrics and beta vs SPY
    spy_ret = data_client.price_returns("SPY", run_day)
    spy_series = spy_ret.get("series_5y") or []
    for c in filtered:
        pr = data_client.price_returns(c["ticker"], run_day)
        c.setdefault("metrics", {}).update({
            "pct_5d": pr.get("pct_5d"),
            "pct_1m": pr.get("pct_1m"),
            "pct_6m": pr.get("pct_6m"),
            "pct_1y": pr.get("pct_1y"),
            "dist_to_200dma": pr.get("dist_to_200dma"),
            "rsi14": pr.get("rsi14"),
            "vol30": pr.get("vol30"),
        })
        if spy_series and pr.get("series_5y"):
            from .stats import beta_vs_benchmark
            c["metrics"]["beta"] = beta_vs_benchmark(pr.get("series_5y"), spy_series)
        # cache closes for chart
        data_client.cache.set(f"SERIES5Y:{c['ticker']}", {"closes": pr.get("series_5y") or []})

    # Compute metrics for scoring + ranking
    engine = ScoreEngine(cfg)
    scored = engine.score_all(run_day, filtered)

    # Sector caps and final picks
    picks, by_sector = engine.apply_sector_caps(scored)

    # Fallback if zero picks: list closest by score from enriched set
    closest: list[dict] = []
    if not picks:
        if enriched:
            closest = engine.score_all(run_day, enriched)[:5]

    # Build charts URLs and background blurbs
    for c in picks:
        if "chart_url_5y" not in c.metrics:
            c.metrics["chart_url_5y"] = data_client.chart_url_5y(c.ticker)
        if "background" not in c.metrics:
            c.metrics["background"] = data_client.background_blurb(c)

    # Ensure Notion database schema
    ensure_database_schema(notion)

    # Notion page title
    title = f"Daily −40% Drop Value Screen – {day_str}"

    # Summary and warnings
    warnings: List[str] = []
    if expanded:
        warnings.append("Expanded criteria used")
    if market_closed and not grouped:
        warnings.append("Market holiday/closed; using last available data")

    # Prepare run log
    runlog = RunLog(path=os.path.join(runs_dir, f"{day_str}.json"))
    runlog.data.update({
        "date": day_str,
        "universe_size": len(universe),
        "triggered": len(candidates),
        "filtered": len(filtered),
        "picked": len(picks),
        "expanded_criteria": expanded,
        "rejects": rejects,
        "warnings": warnings,
        "data_source": cfg.get("provider"),
        "timestamp": datetime.utcnow().isoformat(),
    })
    if closest:
        runlog.data["closest_by_score"] = [c.get("ticker") for c in closest]
    runlog.save()

    # Compose Notion blocks for the daily page
    blocks, summary = build_daily_page_blocks(
        date_str=day_str,
        market_status="Closed" if market_closed else "Open",
        counts={
            "screened": len(universe),
            "passing": len(picks),
        },
        data_source=cfg.get("provider"),
        warnings=warnings,
        picks=picks if picks else closest,
        by_sector=by_sector,
    )

    if dry_run:
        print(json.dumps({
            "page_title": title,
            "summary": summary,
            "blocks": blocks[:2],
            "db_rows": [build_db_row_properties(notion, c, day_str) for c in picks][:3]
        }, indent=2)[:5000])
        logger.info("Dry run complete.")
        return

    # Create or update the daily page with blocks
    page_id = notion.upsert_daily_page(title, blocks)

    # Upsert top N rows in target database
    for c in picks:
        notion.upsert_db_row(c, day_str)

    logger.info(f"Completed Notion update for {day_str}. Page ID: {page_id}")
