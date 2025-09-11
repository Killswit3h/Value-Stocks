from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .logging_utils import get_logger


logger = get_logger(__name__)


def _norm01(values: list[float | None]) -> list[float | None]:
    xs = np.array([v for v in values if v is not None], dtype=float)
    if xs.size == 0:
        return [None for _ in values]
    lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
    if hi - lo < 1e-12:
        return [0.5 if v is not None else None for v in values]
    out = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            out.append((float(v) - lo) / (hi - lo))
    return out


def _bell_curve_score(val: float | None, sweet_lo: float, sweet_hi: float) -> float | None:
    if val is None:
        return None
    # Score highest in the middle (sweet spot), drop toward edges.
    mid = 0.5 * (sweet_lo + sweet_hi)
    width = (sweet_hi - sweet_lo) / 2.0
    if width <= 0:
        return 0.5
    x = (val - mid) / width
    # Gaussian-like bell: exp(-x^2)
    import math
    return float(math.exp(-x * x))


class ScoreEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.top_n = int(cfg.get("run", {}).get("top_n", 12))

    def score_all(self, day, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        # Ensure price-derived metrics per candidate
        for c in candidates:
            # We assume price returns & RSI are added later if needed. For scoring, allow None and reweight.
            pass

        # Collect metric arrays for normalization
        vals = {
            "fcf_yield": [c.get("fundamentals", {}).get("fcf_yield") for c in candidates],
            "ev_to_ebitda": [c.get("fundamentals", {}).get("ev_to_ebitda") for c in candidates],
            "pb": [c.get("fundamentals", {}).get("pb") for c in candidates],
            "roa": [c.get("fundamentals", {}).get("roa") for c in candidates],
            "gm_std": [c.get("fundamentals", {}).get("gross_margin_std_12q") for c in candidates],
            "rev_cagr": [c.get("fundamentals", {}).get("rev_cagr_3y") for c in candidates],
            "rsi14": [c.get("metrics", {}).get("rsi14") for c in candidates],
            "dist200": [c.get("metrics", {}).get("dist_to_200dma") for c in candidates],
            "beta": [c.get("metrics", {}).get("beta") for c in candidates],
            "vol30": [c.get("metrics", {}).get("vol30") for c in candidates],
            "dvol": [c.get("risk", {}).get("adv30_usd") for c in candidates],
            "short_pct": [c.get("risk", {}).get("short_interest_pct") for c in candidates],
        }

        # Normalize
        norm = {
            "fcf_yield": _norm01(vals["fcf_yield"]),
            "ev_to_ebitda_inv": [1 - x if x is not None else None for x in _norm01(vals["ev_to_ebitda"])],
            "pb_inv": [1 - x if x is not None else None for x in _norm01(vals["pb"])],
            "roa": _norm01(vals["roa"]),
            "gm_stability_inv": [1 - x if x is not None else None for x in _norm01(vals["gm_std"])],
            "rev_cagr": _norm01([min(max(v if v is not None else 0.0, 0.0), 0.30) for v in vals["rev_cagr"]]),
            "rsi14_inv": [1 - x if x is not None else None for x in _norm01(vals["rsi14"])],
            "dist_to_200dma_inv": None,  # custom cap below
            "beta_mid": None,  # custom bucket
            "vol30_inv": [1 - x if x is not None else None for x in _norm01(vals["vol30"])],
            "dvol_rank": _norm01(vals["dvol"]),
            "short_sweet": None,
        }

        # Distance to 200DMA inverse capped at 60%
        dist = vals["dist200"]
        capped = []
        for v in dist:
            if v is None:
                capped.append(None)
            else:
                x = min(abs(v), 0.60)
                capped.append(x)
        norm["dist_to_200dma_inv"] = [1 - x if x is not None else None for x in _norm01(capped)]

        # Beta mid-range [0.6, 1.4]
        beta_mid = []
        for b in vals["beta"]:
            if b is None:
                beta_mid.append(None)
            else:
                beta_mid.append(1.0 if 0.6 <= b <= 1.4 else 0.0)
        norm["beta_mid"] = beta_mid

        # Short interest sweet spot 2–10%
        short_scores = []
        for s in vals["short_pct"]:
            short_scores.append(_bell_curve_score(s, 2.0, 10.0))
        norm["short_sweet"] = short_scores

        weights = self.cfg["scoring"]["weights"]

        def bucket_score(i: int) -> tuple[float, dict]:
            parts = []
            details = {}
            # Valuation
            v_w = weights["valuation"]
            for k, w in v_w.items():
                key = k
                nkey = k
                if k.endswith("_inv"):
                    nkey = k
                if k == "ev_to_ebitda_inv":
                    nkey = "ev_to_ebitda_inv"
                if k == "pb_inv":
                    nkey = "pb_inv"
                val = norm.get(nkey, [None])[i]
                parts.append((val, w))
                details[nkey] = (val, w)
            # Quality
            q_w = weights["quality"]
            for k, w in q_w.items():
                nk = "gm_stability_inv" if k == "gm_stability_inv" else k
                val = norm.get(nk, [None])[i]
                parts.append((val, w))
                details[nk] = (val, w)
            # Momentum
            m_w = weights["momentum"]
            for k, w in m_w.items():
                nk = "dist_to_200dma_inv" if k == "dist_to_200dma_inv" else k
                val = norm.get(nk, [None])[i]
                parts.append((val, w))
                details[nk] = (val, w)
            # Risk controls
            r_w = weights["risk"]
            for k, w in r_w.items():
                nk = "beta_mid" if k == "beta_mid_range" else "vol30_inv"
                val = norm.get(nk, [None])[i]
                parts.append((val, w))
                details[nk] = (val, w)
            # Liquidity/ownership
            l_w = weights["liquidity"]
            for k, w in l_w.items():
                nk = "dvol_rank" if "dollar_volume" in k else "short_sweet"
                val = norm.get(nk, [None])[i]
                parts.append((val, w))
                details[nk] = (val, w)
            # Reweight if missing
            total_w = sum(w for v, w in parts if v is not None)
            if total_w == 0:
                return 0.0, details
            score01 = sum((v if v is not None else 0.0) * w for v, w in parts) / total_w
            score100 = max(0.0, min(100.0, score01 * 100.0))
            return score100, details

        # Add event penalties
        out = []
        for i, c in enumerate(candidates):
            score, details = bucket_score(i)
            penalties = 0
            pen_w = weights["penalties"]
            # Earnings window penalty: use metrics field
            ew = c.get("metrics", {}).get("earnings_window")  # "Safe" | "Near" | "Inside 2d" | "Unknown"
            if ew == "Inside 2d":
                penalties += pen_w["earnings_window"]
            elif ew == "Near":
                penalties += pen_w["earnings_window"] * 0.5
            # Headline flags
            if any(f == "headline_flag" for f in c.get("flags", [])):
                penalties += pen_w["headline_flags"]
            # SEC delinquency, going concern (if present in flags)
            if any(f == "sec_delinquency" for f in c.get("flags", [])):
                penalties += pen_w["sec_delinquency"]
            if any(f == "going_concern" for f in c.get("flags", [])):
                penalties += pen_w["going_concern"]

            final_score = max(0.0, score - penalties)
            c2 = dict(c)
            c2.setdefault("metrics", {})
            c2["metrics"]["score"] = final_score
            c2["metrics"]["score_breakdown"] = details
            out.append(c2)

        # Sort by score desc
        out.sort(key=lambda x: x.get("metrics", {}).get("score", 0.0), reverse=True)
        return out

    def apply_sector_caps(self, scored: list[dict]) -> tuple[list[dict], dict]:
        if not scored:
            return [], {}
        caps = self.cfg["universe"]
        max_total = int(caps.get("max_total_symbols", 25))
        max_per_sector = int(caps.get("max_per_sector", 5))
        top_n = int(self.top_n)

        by_sector: dict[str, int] = {}
        picks: list[dict] = []
        for c in scored:
            if len(picks) >= max_total:
                break
            sector = (c.get("sector") or "Unknown").split(" ")[0]
            cnt = by_sector.get(sector, 0)
            if cnt >= max_per_sector:
                continue
            by_sector[sector] = cnt + 1
            picks.append(c)
            if len(picks) >= top_n:
                break
        return picks, by_sector

