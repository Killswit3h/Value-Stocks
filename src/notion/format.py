from __future__ import annotations

from typing import Any, Dict, List

from ..logging_utils import get_logger


logger = get_logger(__name__)


DB_PROPERTIES = {
    "Ticker": {"type": "title"},
    "Company": {"type": "rich_text"},
    "Report Date": {"type": "date"},
    "Score": {"type": "number"},
    "Trigger": {"type": "select", "options": ["1-day −40%", "52w −40% + down day"]},
    "% Today": {"type": "number"},
    "% 5D": {"type": "number"},
    "% 1M": {"type": "number"},
    "% 1Y": {"type": "number"},
    "FCF Yield %": {"type": "number"},
    "EV/EBITDA": {"type": "number"},
    "P/E": {"type": "number"},
    "P/B": {"type": "number"},
    "ROIC %": {"type": "number"},
    "Rev 3y CAGR %": {"type": "number"},
    "Net Debt/EBITDA": {"type": "number"},
    "Interest Coverage": {"type": "number"},
    "Beta": {"type": "number"},
    "30D Realized Vol %": {"type": "number"},
    "Short Interest %": {"type": "number"},
    "Avg $ Vol (30D)": {"type": "number"},
    "Earnings Window": {"type": "select", "options": ["Safe", "Near", "Inside 2d", "Unknown"]},
    "Flags": {"type": "multi_select"},
    "5-Year Chart": {"type": "url"},
    "Background": {"type": "rich_text"},
}


def ensure_database_schema(notion) -> None:
    try:
        db = notion.get_database()
        current = db.get("properties", {})
        patch = {}
        for name, spec in DB_PROPERTIES.items():
            if name in current:
                continue
            t = spec["type"]
            if t == "title":
                patch[name] = {"title": {}}
            elif t == "rich_text":
                patch[name] = {"rich_text": {}}
            elif t == "number":
                patch[name] = {"number": {}}
            elif t == "date":
                patch[name] = {"date": {}}
            elif t == "select":
                options = spec.get("options", [])
                patch[name] = {"select": {"options": [{"name": o} for o in options]}}
            elif t == "multi_select":
                patch[name] = {"multi_select": {}}
            elif t == "url":
                patch[name] = {"url": {}}
        if patch:
            notion.update_database(patch)
    except Exception as e:
        logger.warning(f"Unable to ensure DB schema: {e}")


def _num_prop(v: float | None) -> dict:
    return {"number": float(v)} if v is not None else {"number": None}


def _text_prop(txt: str | None) -> dict:
    return {"rich_text": ([{"text": {"content": txt}}] if txt else [])}


def build_db_row_properties(notion, c: dict, date_str: str) -> dict:
    f = c.get("fundamentals", {})
    m = c.get("metrics", {})
    r = c.get("risk", {})
    props = {
        "Ticker": {"title": [{"text": {"content": c.get("ticker")}}]},
        "Company": _text_prop(c.get("company")),
        "Report Date": {"date": {"start": date_str}},
        "Score": _num_prop(m.get("score")),
        "Trigger": {"select": {"name": c.get("trigger")}},
        "% Today": _num_prop(c.get("prices", {}).get("today_pct")),
        "% 5D": _num_prop(m.get("pct_5d")),
        "% 1M": _num_prop(m.get("pct_1m")),
        "% 1Y": _num_prop(m.get("pct_1y")),
        "FCF Yield %": _num_prop((f.get("fcf_yield") or 0) * 100 if f.get("fcf_yield") is not None else None),
        "EV/EBITDA": _num_prop(f.get("ev_to_ebitda")),
        "P/E": _num_prop(f.get("pe")),
        "P/B": _num_prop(f.get("pb")),
        "ROIC %": _num_prop((f.get("roa") or 0) * 100 if f.get("roa") is not None else None),
        "Rev 3y CAGR %": _num_prop((f.get("rev_cagr_3y") or 0) * 100 if f.get("rev_cagr_3y") is not None else None),
        "Net Debt/EBITDA": _num_prop(f.get("net_debt_to_ebitda")),
        "Interest Coverage": _num_prop(f.get("interest_coverage")),
        "Beta": _num_prop(m.get("beta")),
        "30D Realized Vol %": _num_prop(m.get("vol30")),
        "Short Interest %": _num_prop(r.get("short_interest_pct")),
        "Avg $ Vol (30D)": _num_prop(r.get("adv30_usd")),
        "Earnings Window": {"select": {"name": m.get("earnings_window") or "Unknown"}},
        "Flags": {"multi_select": [{"name": fl} for fl in (c.get("flags") or [])]},
        "5-Year Chart": {"url": c.get("metrics", {}).get("chart_url_5y") or None},
        "Background": _text_prop(m.get("background")),
    }
    return props


def build_daily_page_blocks(date_str: str, market_status: str, counts: dict, data_source: str, warnings: list[str], picks: list[dict], by_sector: dict) -> tuple[list[dict], dict]:
    summary = {
        "date": date_str,
        "market_status": market_status,
        "screened": counts.get("screened"),
        "passing": counts.get("passing"),
        "data_source": data_source,
        "warnings": warnings,
    }
    blocks: list[dict] = []
    # Header
    blocks.append({
        "heading_1": {"rich_text": [{"text": {"content": f"Daily −40% Drop Value Screen – {date_str}"}}]}
    })
    # Callout summary
    callout = f"Market: {market_status} | Screened: {counts.get('screened')} | Passing: {counts.get('passing')} | Data: {data_source}"
    if warnings:
        callout += " | " + ", ".join(warnings)
    blocks.append({
        "callout": {"rich_text": [{"text": {"content": callout}}], "icon": {"emoji": "📊"}}
    })
    # Ranked list (compact)
    for i, c in enumerate(picks, start=1):
        m = c.get("metrics", {})
        p = c.get("prices", {})
        line = f"{i}. {c.get('ticker','')} — {c.get('company','')} | Score {m.get('score',0):.1f} | {p.get('today_pct',0):.1f}% today | 5D {m.get('pct_5d') or 0:.1f}% | 1M {m.get('pct_1m') or 0:.1f}% | 1Y {m.get('pct_1y') or 0:.1f}% | {c.get('trigger','')}"
        blocks.append({"paragraph": {"rich_text": [{"text": {"content": line}}]}})

    # Per-ticker analysis sections
    for c in picks:
        blocks.extend(_ticker_section(c))

    # Rules section
    blocks.append({
        "heading_2": {"rich_text": [{"text": {"content": "Rules"}}]}
    })
    rules = [
        "Max position 2% per name.",
        "Max 20% sector exposure from this screen.",
        "Entry: scale in thirds over 10 trading days if price closes above prior day low.",
        "Exit: hard stop −15% from average entry; 90-day time stop if thesis not validated.",
        "Disable names with pending binary events (FDA dates, litigation, going-concern).",
        "Avoid if liquidity < $5M ADV or spread > 60 bps.",
    ]
    for r in rules:
        blocks.append({"bulleted_list_item": {"rich_text": [{"text": {"content": r}}]}})

    # Disclaimer
    blocks.append({
        "callout": {"rich_text": [{"text": {"content": "This is not financial advice."}}], "icon": {"emoji": "⚠️"}}
    })

    return blocks, summary


def _table_block(header: list[str], rows: list[list[str]]) -> dict:
    # Deprecated: Keeping for reference; Not used in v1 due to API nesting constraints.
    return {"paragraph": {"rich_text": [{"text": {"content": "Table view omitted in v1."}}]}}


def _ticker_section(c: dict) -> list[dict]:
    t = c.get("ticker")
    f = c.get("fundamentals", {})
    m = c.get("metrics", {})
    # Collapsible toggle with 5-year analysis summary
    bullets = []
    bullets.append(f"EV/EBITDA: {f.get('ev_to_ebitda')}")
    bullets.append(f"P/E: {f.get('pe')}, P/B: {f.get('pb')}")
    bullets.append(f"Gross Margin: {round((f.get('gross_margin') or 0)*100,1) if f.get('gross_margin') is not None else 'N/A'}% | ROA: {round((f.get('roa') or 0)*100,1) if f.get('roa') is not None else 'N/A'}%")
    bullets.append(f"Drawdown from 200-DMA: {round((m.get('dist_to_200dma') or 0)*100,1) if m.get('dist_to_200dma') is not None else 'N/A'}%")
    bullets.append("Mean reversion note: RSI<30 frequency and 30D fwd return median not computed in v1")
    bullets.append("Disclaimer: This is not financial advice.")

    children = [{"bulleted_list_item": {"rich_text": [{"text": {"content": b}}]}} for b in bullets]
    return [{"toggle": {"rich_text": [{"text": {"content": f"{t} – 5-year analysis"}}], "children": children}}]
