"""Minimal Notion integration layer."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

from .models import CandidateResult, RunMetadata
from .utils import setup_logger


class NotionService:
    """Prepare and optionally send payloads to Notion."""

    def __init__(self, token: Optional[str], database_id: Optional[str]) -> None:
        self.logger = setup_logger(__name__)
        self.token = token
        self.database_id = database_id
        self.enabled = bool(token and database_id)
        self.session = requests.Session() if requests is not None else None
        if self.enabled and self.session is not None:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {token}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type": "application/json",
                }
            )
        else:
            self.logger.warning("Notion credentials missing – falling back to console output")
            if self.enabled and self.session is None:
                self.logger.warning("requests not available; Notion push disabled")
                self.enabled = False

    def build_placeholder_page(self, run_date: date, metadata: RunMetadata) -> Dict[str, Any]:
        title = f"Daily −40% Drop Value Screen – {run_date.isoformat()}"
        return {
            "title": title,
            "metadata": asdict(metadata),
            "summary": {
                "message": "Market closed – no screening performed.",
                "expanded": False,
                "median_score": 0,
            },
            "candidates": [],
            "rules": [],
            "analysis": [],
            "closest": [],
        }

    def build_report_payload(
        self,
        run_date: date,
        candidates: List[CandidateResult],
        all_candidates: List[CandidateResult],
        metadata: RunMetadata,
        rules_block: Iterable[str],
        expanded: bool,
    ) -> Dict[str, Any]:
        title = f"Daily −40% Drop Value Screen – {run_date.isoformat()}"
        table_rows = [self._build_row(candidate, metadata, run_date) for candidate in candidates]
        all_scores = [candidate.score.total for candidate in candidates if candidate.score]
        median_score = median(all_scores) if all_scores else 0.0
        analysis_blocks = [self._build_analysis(candidate) for candidate in candidates]
        closest = []
        if not candidates:
            self.logger.info("No passing candidates; selecting closest by score")
            ranked = sorted(all_candidates, key=lambda item: item.score.total, reverse=True)
            closest = [self._build_row(item, metadata, run_date) for item in ranked[:5]]
        payload = {
            "title": title,
            "metadata": asdict(metadata),
            "summary": {
                "message": f"{len(candidates)} candidates selected.",
                "expanded": expanded,
                "median_score": median_score,
            },
            "candidates": table_rows,
            "rules": list(rules_block),
            "analysis": analysis_blocks,
            "closest": closest,
            "notes": [note for candidate in candidates for note in candidate.metrics.notes],
        }
        if not candidates:
            payload["summary"]["message"] = "No candidates met criteria"
        if expanded:
            payload.setdefault("warnings", []).append("Expanded criteria used")
        if metadata.api_warnings:
            payload.setdefault("warnings", []).extend(metadata.api_warnings)
        return payload

    def dispatch(self, payload: Dict[str, Any], dry_run: bool) -> None:
        if dry_run or not self.enabled:
            print(json.dumps(payload, indent=2, default=str))
            return
        try:
            self._push_to_notion(payload)
        except Exception as exc:  # pragma: no cover - integration error
            self.logger.error("Failed to push to Notion: %s", exc)
            print(json.dumps(payload, indent=2, default=str))

    def _push_to_notion(self, payload: Dict[str, Any]) -> None:
        if self.session is None:
            raise RuntimeError("HTTP client unavailable")
        page_id = self._find_existing_page(payload["title"])
        if page_id:
            self.logger.info("Updating existing Notion page %s", page_id)
            self._clear_children(page_id)
        else:
            page_id = self._create_page(payload)
        self._append_blocks(page_id, payload)

    def _build_row(self, candidate: CandidateResult, metadata: RunMetadata, run_date: date) -> Dict[str, Any]:
        metrics = candidate.metrics
        return {
            "rank": candidate.rank,
            "ticker": metrics.ticker,
            "company": metrics.company_name,
            "score": candidate.score.total,
            "trigger": metrics.trigger,
            "change_today": metrics.change_1d,
            "change_5d": metrics.change_5d,
            "change_1m": metrics.change_1m,
            "change_1y": metrics.change_1y,
            "fcf_yield": metrics.free_cash_flow_yield,
            "ev_ebitda": metrics.ev_to_ebitda,
            "pe": metrics.pe_ratio,
            "pb": metrics.pb_ratio,
            "roic": metrics.roic,
            "revenue_cagr": metrics.revenue_cagr_3y,
            "net_debt_ebitda": metrics.net_debt_to_ebitda,
            "interest_coverage": metrics.interest_coverage,
            "beta": metrics.beta,
            "vol_30d": metrics.realized_vol_30d,
            "short_interest": metrics.short_interest_percent,
            "dollar_volume": metrics.avg_dollar_volume,
            "earnings_window": metrics.earnings_date.isoformat() if metrics.earnings_date else None,
            "flags": metrics.headline_flags,
            "chart_path": metrics.chart_path,
            "background": metrics.notes[-1] if metrics.notes else None,
        }

    def _build_analysis(self, candidate: CandidateResult) -> Dict[str, Any]:
        metrics = candidate.metrics
        stats = metrics.five_year_stats
        lines = [
            f"P/E {self._format_number(metrics.pe_ratio)} vs 5y {self._format_number(stats.median_pe)}",
            f"EV/EBITDA {self._format_number(metrics.ev_to_ebitda)} vs 5y {self._format_number(stats.median_ev_to_ebitda)}",
            f"Gross margin {self._format_percent(metrics.gross_margin)} vs 5y {self._format_percent(stats.median_gross_margin)}",
            f"ROIC {self._format_percent(metrics.roic)} vs 5y {self._format_percent(stats.median_roic)}",
        ]
        if stats.max_drawdown_pct is not None:
            lines.append(f"Drawdown from ATH {self._format_percent(stats.max_drawdown_pct)}")
        if stats.drawdown_from_200dma_pct is not None:
            lines.append(f"vs 200DMA {self._format_percent(stats.drawdown_from_200dma_pct)}")
        if stats.rsi_under_30_frequency is not None and stats.median_forward_return_after_rsi is not None:
            lines.append(
                f"RSI<30 on {self._format_percent(stats.rsi_under_30_frequency)} of days; median {self._format_percent(stats.median_forward_return_after_rsi)} 30d forward"
            )
        lines.append("Not financial advice.")
        return {"ticker": metrics.ticker, "lines": lines}

    def _create_page(self, payload: Dict[str, Any]) -> str:
        properties = {
            "Ticker": {"title": [{"text": {"content": payload["title"]}}]},
        }
        body = {
            "parent": {"database_id": self.database_id},
            "properties": properties,
        }
        if self.session is None:
            raise RuntimeError("HTTP client unavailable")
        response = self.session.post("https://api.notion.com/v1/pages", json=body, timeout=30)
        response.raise_for_status()
        page_id = response.json()["id"]
        return page_id

    def _find_existing_page(self, title: str) -> Optional[str]:
        if not self.database_id:
            return None
        if self.session is None:
            return None
        body = {
            "filter": {
                "property": "Ticker",
                "title": {"equals": title},
            }
        }
        response = self.session.post(
            f"https://api.notion.com/v1/databases/{self.database_id}/query",
            json=body,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None
        return results[0]["id"]

    def _clear_children(self, page_id: str) -> None:
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        if self.session is None:
            return
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
        for block in results:
            block_id = block.get("id")
            if not block_id:
                continue
            self.session.patch(
                f"https://api.notion.com/v1/blocks/{block_id}",
                json={"archived": True},
                timeout=30,
            )

    def _append_blocks(self, page_id: str, payload: Dict[str, Any]) -> None:
        blocks: List[Dict[str, Any]] = []
        summary_text = payload["summary"]["message"]
        callout = {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": summary_text}}],
                "icon": {"emoji": "📉"},
            },
        }
        blocks.append(callout)
        if payload.get("warnings"):
            warning_text = " | ".join(payload["warnings"])
            blocks.append(
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{"type": "text", "text": {"content": warning_text}}],
                        "icon": {"emoji": "⚠️"},
                    },
                }
            )
        if payload["candidates"]:
            table_block = self._build_table_block(payload["candidates"])
            blocks.append(table_block)
        elif payload.get("closest"):
            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Top 5 by score despite failing filters:"},
                            }
                        ]
                    },
                }
            )
            for item in payload["closest"]:
                text = f"{item['ticker']} – score {self._format_number(item['score'])}"
                blocks.append(
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                {"type": "text", "text": {"content": text}}
                            ]
                        },
                    }
                )
        if payload.get("analysis"):
            for item in payload["analysis"]:
                toggle_block = {
                    "object": "block",
                    "type": "toggle",
                    "toggle": {
                        "rich_text": [
                            {"type": "text", "text": {"content": f"{item['ticker']} five-year context"}}
                        ],
                        "children": [
                            {
                                "object": "block",
                                "type": "bulleted_list_item",
                                "bulleted_list_item": {
                                    "rich_text": [
                                        {"type": "text", "text": {"content": line}}
                                    ]
                                },
                            }
                            for line in item["lines"]
                        ],
                    },
                }
                blocks.append(toggle_block)
        if payload.get("rules"):
            rules_children = [
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": rule}}]
                    },
                }
                for rule in payload["rules"]
            ]
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Rules"}}]},
                }
            )
            blocks.extend(rules_children)
        if self.session is None:
            raise RuntimeError("HTTP client unavailable")
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        for chunk_start in range(0, len(blocks), 50):
            chunk = blocks[chunk_start : chunk_start + 50]
            response = self.session.patch(url, json={"children": chunk}, timeout=30)
            response.raise_for_status()

    def _build_table_block(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        columns = [
            "Rank",
            "Ticker",
            "Company",
            "Score",
            "% Today",
            "% 5D",
            "% 1M",
            "% 1Y",
            "FCF Yield",
            "EV/EBITDA",
            "P/E",
            "P/B",
            "ROIC",
            "Rev CAGR",
            "Net Debt/EBITDA",
            "Interest Cov",
            "Beta",
            "30D Vol",
            "Short Interest",
            "ADV",
        ]
        header_row = {
            "object": "block",
            "type": "table_row",
            "table_row": {
                "cells": [[{"type": "text", "text": {"content": col}}] for col in columns]
            },
        }
        body_rows = []
        for row in rows:
            cells = [
                self._cell(row.get("rank")),
                self._cell(row.get("ticker")),
                self._cell(row.get("company")),
                self._cell(self._format_number(row.get("score"))),
                self._cell(self._format_percent(row.get("change_today"))),
                self._cell(self._format_percent(row.get("change_5d"))),
                self._cell(self._format_percent(row.get("change_1m"))),
                self._cell(self._format_percent(row.get("change_1y"))),
                self._cell(self._format_percent(row.get("fcf_yield"))),
                self._cell(self._format_number(row.get("ev_ebitda"))),
                self._cell(self._format_number(row.get("pe"))),
                self._cell(self._format_number(row.get("pb"))),
                self._cell(self._format_percent(row.get("roic"))),
                self._cell(self._format_percent(row.get("revenue_cagr"))),
                self._cell(self._format_number(row.get("net_debt_ebitda"))),
                self._cell(self._format_number(row.get("interest_coverage"))),
                self._cell(self._format_number(row.get("beta"))),
                self._cell(self._format_percent(row.get("vol_30d"))),
                self._cell(self._format_percent(row.get("short_interest"))),
                self._cell(self._format_dollar(row.get("dollar_volume"))),
            ]
            body_rows.append({"object": "block", "type": "table_row", "table_row": {"cells": cells}})
        return {
            "object": "block",
            "type": "table",
            "table": {
                "table_width": len(columns),
                "has_column_header": True,
                "has_row_header": False,
            },
            "children": [header_row] + body_rows,
        }

    def _cell(self, content: Any) -> List[Dict[str, Any]]:
        if content is None:
            content = "N/A"
        else:
            content = str(content)
        return [{"type": "text", "text": {"content": content}}]

    def _format_percent(self, value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value * 100:.1f}%"

    def _format_number(self, value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value:.2f}"

    def _format_dollar(self, value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        if value >= 1_000_000:
            return f"${value/1_000_000:.1f}M"
        if value >= 1_000:
            return f"${value/1_000:.1f}K"
        return f"${value:.0f}"
