"""Composite scoring logic."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import math

from .models import CandidateMetrics, ScoreBreakdown


@dataclass
class MetricDefinition:
    key: str
    weight: float
    higher_is_better: bool = True
    cap: float | None = None
    floor: float | None = None
    clip_range: Tuple[float, float] | None = None
    transform: Optional[Callable[[CandidateMetrics], Optional[float]]] = None


def _extract(
    values: Iterable[CandidateMetrics],
    attribute: str,
    transform: Optional[Callable[[CandidateMetrics], Optional[float]]] = None,
) -> List[float]:
    extracted: List[float] = []
    for metrics in values:
        if transform is not None:
            value = transform(metrics)
        else:
            value = getattr(metrics, attribute)
        if value is None:
            continue
        extracted.append(float(value))
    return extracted


def _normalize(values: List[float], higher_is_better: bool = True) -> Dict[float, float]:
    if not values:
        return {}
    unique_values = sorted(set(values))
    if not higher_is_better:
        unique_values = list(reversed(unique_values))
    if len(unique_values) == 1:
        return {unique_values[0]: 1.0}
    max_rank = len(unique_values) - 1
    mapping: Dict[float, float] = {}
    for rank, value in enumerate(unique_values):
        mapping[value] = rank / max_rank
    # Duplicate values receive the same score
    for value in values:
        if value not in mapping:
            mapping[value] = 0.0
    return mapping


def _score_metric(
    metrics: CandidateMetrics,
    definition: MetricDefinition,
    normalization_map: Dict[float, float],
) -> float:
    if definition.transform is not None:
        raw_value = definition.transform(metrics)
    else:
        raw_value = getattr(metrics, definition.key)
    if raw_value is None:
        return 0.0
    value = float(raw_value)
    if definition.clip_range is not None:
        lo, hi = definition.clip_range
        if value < lo:
            value = lo
        if value > hi:
            value = hi
    if definition.cap is not None:
        value = min(value, definition.cap)
    if definition.floor is not None:
        value = max(value, definition.floor)
    if not normalization_map:
        return 0.0
    score = normalization_map.get(value)
    if score is None:
        closest = min(normalization_map.keys(), key=lambda x: abs(x - value))
        score = normalization_map[closest]
    return score * definition.weight


def calculate_scores(
    candidates: List[CandidateMetrics],
    config: Dict[str, Dict[str, float]],
    run_date: date,
) -> Dict[str, ScoreBreakdown]:
    """Compute composite scores."""

    valuation_defs = [
        MetricDefinition("free_cash_flow_yield", config["valuation"].get("fcf_yield_weight", 15)),
        MetricDefinition(
            "ev_to_ebitda",
            config["valuation"].get("ev_to_ebitda_weight", 10),
            higher_is_better=False,
        ),
        MetricDefinition(
            "pb_ratio",
            config["valuation"].get("price_to_book_weight", 5),
            higher_is_better=False,
        ),
    ]
    quality_defs = [
        MetricDefinition("roic", config["quality"].get("roic_weight", 10)),
        MetricDefinition(
            "gross_margin_stability",
            config["quality"].get("gross_margin_stability_weight", 5),
            higher_is_better=False,
        ),
        MetricDefinition(
            "revenue_cagr_3y",
            config["quality"].get("revenue_cagr_weight", 5),
            clip_range=(0, 0.30),
        ),
    ]
    momentum_defs = [
        MetricDefinition(
            "rsi_14",
            config["momentum"].get("rsi_weight", 10),
            higher_is_better=False,
        ),
        MetricDefinition(
            "percent_to_200dma",
            config["momentum"].get("distance_to_200dma_weight", 5),
            higher_is_better=False,
            cap=0.60,
        ),
    ]
    risk_defs = [
        MetricDefinition(
            "beta",
            config["risk"].get("beta_weight", 10),
        ),
        MetricDefinition(
            "realized_vol_30d",
            config["risk"].get("volatility_weight", 5),
            higher_is_better=False,
        ),
    ]
    liquidity_defs = [
        MetricDefinition("avg_dollar_volume", config["liquidity"].get("dollar_volume_weight", 5)),
        MetricDefinition(
            "short_interest_percent",
            config["liquidity"].get("short_interest_weight", 5),
            transform=lambda m: short_interest_adjustment(m.short_interest_percent)
            if m.short_interest_percent is not None
            else None,
        ),
    ]

    def build_norm_map(defs: List[MetricDefinition]) -> Dict[str, Dict[float, float]]:
        maps: Dict[str, Dict[float, float]] = {}
        for definition in defs:
            values = _extract(candidates, definition.key, definition.transform)
            if not values:
                maps[definition.key] = {}
                continue
            maps[definition.key] = _normalize(values, definition.higher_is_better)
        return maps

    valuation_map = build_norm_map(valuation_defs)
    quality_map = build_norm_map(quality_defs)
    momentum_map = build_norm_map(momentum_defs)
    risk_map = build_norm_map(risk_defs)
    liquidity_map = build_norm_map(liquidity_defs)

    scores: Dict[str, ScoreBreakdown] = {}
    penalties_cfg = config.get("penalties", {})
    valuation_max = sum(definition.weight for definition in valuation_defs)
    quality_max = sum(definition.weight for definition in quality_defs)
    momentum_max = sum(definition.weight for definition in momentum_defs)
    risk_max = config.get("risk", {}).get("beta_weight", 10.0)
    risk_max += sum(
        definition.weight for definition in risk_defs if definition.key != "beta"
    )
    liquidity_max = sum(definition.weight for definition in liquidity_defs)
    max_core = valuation_max + quality_max + momentum_max + risk_max + liquidity_max
    for metrics in candidates:
        valuation_score = sum(
            _score_metric(metrics, definition, valuation_map.get(definition.key, {}))
            for definition in valuation_defs
            if metrics.ev_to_ebitda is not None or definition.key != "ev_to_ebitda"
        )
        quality_score = sum(
            _score_metric(metrics, definition, quality_map.get(definition.key, {}))
            for definition in quality_defs
        )
        momentum_score = sum(
            _score_metric(metrics, definition, momentum_map.get(definition.key, {}))
            for definition in momentum_defs
        )
        risk_score = _beta_band_score(metrics, config.get("risk", {}))
        risk_score += sum(
            _score_metric(metrics, definition, risk_map.get(definition.key, {}))
            for definition in risk_defs
            if definition.key != "beta"
        )
        liquidity_score = sum(
            _score_metric(metrics, definition, liquidity_map.get(definition.key, {}))
            for definition in liquidity_defs
        )
        penalty_score = _event_penalties(metrics, penalties_cfg, run_date)
        core_score = (
            valuation_score + quality_score + momentum_score + risk_score + liquidity_score
        )
        normalized_core = (core_score / max_core) * 100 if max_core else 0.0
        total = max(0.0, min(100.0, normalized_core + penalty_score))
        normalized_inputs = {
            "fcf_yield": getattr(metrics, "free_cash_flow_yield"),
            "ev_to_ebitda": getattr(metrics, "ev_to_ebitda"),
            "pb_ratio": getattr(metrics, "pb_ratio"),
            "roic": getattr(metrics, "roic"),
            "gross_margin_stability": getattr(metrics, "gross_margin_stability"),
            "revenue_cagr_3y": getattr(metrics, "revenue_cagr_3y"),
            "rsi_14": getattr(metrics, "rsi_14"),
            "percent_to_200dma": getattr(metrics, "percent_to_200dma"),
            "beta": getattr(metrics, "beta"),
            "realized_vol_30d": getattr(metrics, "realized_vol_30d"),
            "avg_dollar_volume": getattr(metrics, "avg_dollar_volume"),
            "short_interest_percent": getattr(metrics, "short_interest_percent"),
            "short_interest_adj": short_interest_adjustment(
                metrics.short_interest_percent or 0.0
            ),
            "penalty": penalty_score,
        }
        scores[metrics.ticker] = ScoreBreakdown(
            total=round(total, 2),
            valuation=round((valuation_score / max_core) * 100, 2) if max_core else 0.0,
            quality=round((quality_score / max_core) * 100, 2) if max_core else 0.0,
            momentum=round((momentum_score / max_core) * 100, 2) if max_core else 0.0,
            risk=round((risk_score / max_core) * 100, 2) if max_core else 0.0,
            liquidity=round((liquidity_score / max_core) * 100, 2) if max_core else 0.0,
            penalties=round(penalty_score, 2),
            normalized_inputs=normalized_inputs,
        )
    return scores


def _beta_band_score(metrics: CandidateMetrics, config: Dict[str, float]) -> float:
    center = config.get("beta_band_center", 1.0)
    halfwidth = config.get("beta_band_halfwidth", 0.4)
    weight = config.get("beta_weight", 10.0)
    if metrics.beta is None:
        return 0.0
    distance = abs(metrics.beta - center)
    score = max(0.0, (halfwidth - distance) / halfwidth)
    return score * weight


def _event_penalties(metrics: CandidateMetrics, config: Dict[str, float], run_date: date) -> float:
    score = 0.0
    earnings_weight = config.get("earnings_weight", 10.0)
    if metrics.earnings_date is not None:
        delta = (metrics.earnings_date - run_date).days
        if abs(delta) <= 10:
            score -= earnings_weight
    if metrics.headline_flags:
        score -= config.get("headline_weight", 25.0)
    if metrics.sec_delinquent:
        score -= config.get("sec_delinquency_weight", 40.0)
    if metrics.going_concern:
        score -= config.get("going_concern_weight", 60.0)
    return score


def short_interest_adjustment(value: float) -> float:
    """Bell-curve style preference for 2-10% short interest."""
    if value <= 0:
        return 0.0
    mean = 0.06
    sigma = 0.03
    exponent = -((value - mean) ** 2) / (2 * sigma**2)
    return math.exp(exponent)
