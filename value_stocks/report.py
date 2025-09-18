"""Pipeline orchestration for the daily screen."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .background import build_background_blurb
from .charts import generate_price_chart
from .config_loader import AppConfig
from .data_provider import MarketDataProvider, ProviderConfig
from .filters import determine_drop_trigger, passes_value_filters
from .models import CandidateMetrics, CandidateResult, RunMetadata, ScoreBreakdown
from .notion import NotionService
from .scoring import calculate_scores
from .utils import ensure_dir, is_weekend, nyc_now, save_json, setup_logger


class DailyReportRunner:
    """Main orchestration class."""

    def __init__(
        self,
        config: AppConfig,
        notion_service: NotionService,
        provider: MarketDataProvider,
        run_root: str | Path = "runs",
    ) -> None:
        self.config = config
        self.notion_service = notion_service
        self.provider = provider
        self.logger = setup_logger(__name__)
        self.run_root = Path(run_root)

    def run(self, run_date: date | None = None, dry_run: bool = False) -> RunMetadata:
        if run_date is None:
            run_date = nyc_now().date()
        market_status, is_open = self._market_status(run_date)
        self.logger.info("Running screen for %s – Market status: %s", run_date, market_status)

        if not is_open:
            self.logger.info("Market closed – writing placeholder page")
            metadata = RunMetadata(
                run_datetime=nyc_now(),
                market_status=market_status,
                universe_size=0,
                considered=0,
                passed_filters=0,
                expanded_criteria_used=False,
                data_source=self.config.data_source.get("provider", "unknown"),
                api_warnings=["Market closed"],
            )
            payload = self.notion_service.build_placeholder_page(run_date, metadata)
            self.notion_service.dispatch(payload, dry_run=dry_run)
            return metadata

        primary_candidates = self.provider.load_candidates(run_date, expanded=False)
        self.logger.info("Loaded %s raw candidates", len(primary_candidates))

        processed_primary = self._process_candidates(primary_candidates, run_date, expanded=False)
        triggered_count = len(processed_primary["triggered"])
        self.logger.info("Candidates triggering drop condition: %s", triggered_count)

        passed_candidates = list(processed_primary["passed"])
        expanded_used = False
        if len(passed_candidates) < self.config.execution.get("min_candidates_before_expand", 5):
            self.logger.info("Expanding criteria due to insufficient candidates")
            expanded_candidates = self.provider.load_candidates(run_date, expanded=True)
            processed_expanded = self._process_candidates(expanded_candidates, run_date, expanded=True)
            expanded_used = True
            passed_candidates.extend(processed_expanded["passed"])
            self._merge_rejections(processed_primary["rejections"], processed_expanded["rejections"])
            processed_primary["triggered"].extend(processed_expanded["triggered"])

        scores = calculate_scores(
            [candidate.metrics for candidate in processed_primary["triggered"]],
            self.config.scores,
            run_date,
        )
        for result in processed_primary["triggered"]:
            result.score = scores.get(result.metrics.ticker, result.score)

        final_candidates = self._rank_and_cap(passed_candidates)
        self.logger.info("Final candidate count after sector caps: %s", len(final_candidates))

        charts_dir = ensure_dir(self.config.execution.get("chart_dir", "charts"))
        for candidate in final_candidates:
            chart_path = generate_price_chart(
                candidate.metrics.ticker,
                candidate.metrics.price_history_5y,
                charts_dir,
            )
            candidate.metrics.chart_path = str(chart_path)

        for candidate in final_candidates:
            candidate.metrics.notes.append(build_background_blurb(candidate.metrics, run_date))

        rules_block = self._build_rules_block()
        run_metadata = RunMetadata(
            run_datetime=nyc_now(),
            market_status=market_status,
            universe_size=len(primary_candidates),
            considered=len(processed_primary["triggered"]),
            passed_filters=len(final_candidates),
            expanded_criteria_used=expanded_used,
            data_source=self.config.data_source.get("provider", "unknown"),
            api_warnings=[],
        )

        run_log = self._build_run_log(
            run_date,
            processed_primary,
            final_candidates,
            run_metadata,
        )
        log_path = self._write_run_log(run_date, run_log)
        run_metadata.run_log_path = str(log_path)

        summary_payload = self.notion_service.build_report_payload(
            run_date=run_date,
            candidates=final_candidates,
            all_candidates=processed_primary["triggered"],
            metadata=run_metadata,
            rules_block=rules_block,
            expanded=expanded_used,
        )
        self.notion_service.dispatch(summary_payload, dry_run=dry_run)
        return run_metadata

    def _process_candidates(
        self,
        metrics_list: List[CandidateMetrics],
        run_date: date,
        expanded: bool,
    ) -> Dict[str, Any]:
        triggered: List[CandidateResult] = []
        passed: List[CandidateResult] = []
        rejections: Dict[str, Dict[str, int]] = {"universe": {}, "filters": {}}

        universe_limits = self.config.universe
        filter_config = self.config.filters

        for metrics in metrics_list:
            is_triggered, trigger_name, reasons = determine_drop_trigger(metrics, expanded)
            if not is_triggered:
                continue
            metrics.trigger = trigger_name
            metrics.triggered_conditions = reasons

            universe_ok, universe_reasons = self._passes_universe(metrics, universe_limits)
            if not universe_ok:
                for reason in universe_reasons:
                    rejections.setdefault("universe", {}).setdefault(reason, 0)
                    rejections["universe"][reason] += 1
                continue

            filter_result = passes_value_filters(metrics, filter_config, run_date)
            candidate = CandidateResult(
                metrics=metrics,
                filter_result=filter_result,
                score=ScoreBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
            triggered.append(candidate)

            if not filter_result.passed:
                for reason in filter_result.reasons:
                    rejections.setdefault("filters", {}).setdefault(reason, 0)
                    rejections["filters"][reason] += 1
                continue
            passed.append(candidate)
        return {"triggered": triggered, "passed": passed, "rejections": rejections}

    def _passes_universe(
        self, metrics: CandidateMetrics, config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if metrics.price < config.get("min_price", 3.0):
            reasons.append("Price below minimum")
        if metrics.market_cap < config.get("min_market_cap", 0):
            reasons.append("Market cap below minimum")
        if metrics.avg_dollar_volume < config.get("min_adv_usd", 0):
            reasons.append("Liquidity below threshold")
        if any(keyword.lower() in metrics.company_name.lower() for keyword in config.get("excluded_keywords", [])):
            reasons.append("Excluded keyword match")
        if metrics.trading_halted:
            reasons.append("Trading halted")
        return len(reasons) == 0, reasons

    def _rank_and_cap(self, candidates: List[CandidateResult]) -> List[CandidateResult]:
        max_total = self.config.universe.get("max_total_candidates", 25)
        max_sector = self.config.universe.get("max_per_sector", 5)
        unique: Dict[str, CandidateResult] = {}
        for candidate in candidates:
            unique[candidate.metrics.ticker] = candidate
        sorted_candidates = sorted(
            unique.values(),
            key=lambda result: result.score.total,
            reverse=True,
        )
        sector_counts: Dict[str, int] = {}
        final: List[CandidateResult] = []
        for result in sorted_candidates:
            if len(final) >= max_total:
                break
            sector = result.metrics.sector
            count = sector_counts.get(sector, 0)
            if count >= max_sector:
                continue
            sector_counts[sector] = count + 1
            final.append(result)
        for idx, result in enumerate(final[: self.config.execution.get("max_candidates", 12)], start=1):
            result.rank = idx
        return final[: self.config.execution.get("max_candidates", 12)]

    def _market_status(self, run_date: date) -> Tuple[str, bool]:
        if is_weekend(run_date):
            return "Closed (Weekend)", False
        try:
            import pandas_market_calendars as mcal  # type: ignore

            calendar = mcal.get_calendar("XNYS")
            schedule = calendar.schedule(run_date, run_date)
            if schedule.empty:
                return "Closed (Holiday)", False
        except Exception:  # pragma: no cover - fallback when library missing
            self.logger.debug("pandas_market_calendars unavailable; assuming open weekday")
        return "Open", True

    def _build_rules_block(self) -> List[str]:
        return [
            "Max position 2% of portfolio per name.",
            "Max 20% sector exposure from this screen.",
            "Entry plan: scale in thirds over 10 trading days if price closes above the prior day low.",
            "Exit plan: hard stop at −15% from average entry, time stop at 90 trading days if thesis not validated.",
            "Disable names with pending binary events (biotech FDA dates, litigation, going-concern).",
            "Avoid if liquidity < $5M ADV or spread > 60 bps.",
        ]

    def _build_run_log(
        self,
        run_date: date,
        processed: Dict[str, List[CandidateResult]],
        final_candidates: List[CandidateResult],
        metadata: RunMetadata,
    ) -> Dict[str, Any]:
        return {
            "run_date": run_date.isoformat(),
            "metadata": asdict(metadata),
            "triggered": [self._serialize_candidate(result) for result in processed["triggered"]],
            "final": [self._serialize_candidate(result) for result in final_candidates],
            "rejections": processed["rejections"],
        }

    def _serialize_candidate(self, result: CandidateResult) -> Dict[str, Any]:
        return {
            "ticker": result.metrics.ticker,
            "company": result.metrics.company_name,
            "sector": result.metrics.sector,
            "score": result.score.total,
            "trigger": result.metrics.trigger,
            "filter_passed": result.filter_result.passed,
            "filter_reasons": result.filter_result.reasons,
        }

    def _write_run_log(self, run_date: date, log_data: Dict[str, Any]) -> Path:
        ensure_dir(self.run_root)
        path = self.run_root / f"{run_date.isoformat()}.json"
        save_json(path, log_data)
        return path

    def _merge_rejections(
        self, base: Dict[str, Dict[str, int]], incoming: Dict[str, Dict[str, int]]
    ) -> None:
        for bucket, values in incoming.items():
            for reason, count in values.items():
                base.setdefault(bucket, {}).setdefault(reason, 0)
                base[bucket][reason] += count


def create_runner(config: AppConfig, env: Dict[str, str]) -> DailyReportRunner:
    provider_config = ProviderConfig(
        base_url=config.data_source.get("base_url", ""),
        api_key=env.get("DATA_API_KEY"),
        max_retries=int(config.data_source.get("max_retries", 5)),
        backoff_seconds=int(config.data_source.get("backoff_seconds", 1)),
        cache_ttl_seconds=int(config.data_source.get("cache_ttl_seconds", 900)),
    )
    provider = MarketDataProvider(provider_config)
    notion_service = NotionService(
        token=env.get("NOTION_API_KEY"),
        database_id=env.get("NOTION_DATABASE_ID"),
    )
    return DailyReportRunner(config=config, notion_service=notion_service, provider=provider)
