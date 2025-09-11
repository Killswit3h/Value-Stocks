import pytest

from src.scoring import ScoreEngine


def test_score_basic_normalization():
    cfg = {
        "run": {"top_n": 12},
        "scoring": {
            "weights": {
                "valuation": {"fcf_yield": 15, "ev_to_ebitda_inv": 10, "pb_inv": 5},
                "quality": {"roic_or_roa": 10, "gm_stability_inv": 5, "rev_cagr_3y": 5},
                "momentum": {"rsi14_inv": 10, "dist_to_200dma_inv": 5},
                "risk": {"beta_mid_range": 10, "vol30_inv": 5},
                "liquidity": {"dollar_volume_rank": 5, "short_interest_sweetspot": 5},
                "penalties": {"earnings_window": 10, "headline_flags": 25, "sec_delinquency": 40, "going_concern": 60},
            }
        }
    }
    engine = ScoreEngine(cfg)
    cands = [
        {"ticker": "AAA", "fundamentals": {"fcf_yield": 0.05, "ev_to_ebitda": 8, "pb": 1.2, "roa": 0.08, "gross_margin_std_12q": 0.05, "rev_cagr_3y": 0.10}, "metrics": {"rsi14": 30, "dist_to_200dma": -0.2, "beta": 1.0, "vol30": 40}, "risk": {"adv30_usd": 10000000, "short_interest_pct": 5}},
        {"ticker": "BBB", "fundamentals": {"fcf_yield": 0.02, "ev_to_ebitda": 12, "pb": 2.0, "roa": 0.03, "gross_margin_std_12q": 0.08, "rev_cagr_3y": 0.05}, "metrics": {"rsi14": 60, "dist_to_200dma": -0.1, "beta": 1.6, "vol30": 60}, "risk": {"adv30_usd": 6000000, "short_interest_pct": 1}},
    ]
    scored = engine.score_all(None, cands)
    assert len(scored) == 2
    assert scored[0]["ticker"] == "AAA"
    assert 0 <= scored[0]["metrics"]["score"] <= 100

