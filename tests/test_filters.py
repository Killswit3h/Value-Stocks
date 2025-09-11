import pytest

from src.providers.polygon_client import PolygonClient


def test_apply_filters_basic(monkeypatch):
    cfg = {
        "run": {"http_timeout_sec": 10, "max_retries": 1, "cache_dir": ".cache_test"},
        "universe": {"min_market_cap": 500000000, "min_adv_usd_30d": 5000000},
        "filters": {"require_positive_ocf_ttm": True, "net_debt_to_ebitda_max": 4.0, "interest_coverage_min": 3.0, "gross_margin_min": 0.15, "allow_low_gm_sectors": ["Energy", "Materials"]},
    }
    pc = PolygonClient(api_key="test", cfg=cfg)

    enriched = [
        {"ticker": "AAA", "sector": "Tech", "fundamentals": {"market_cap": 800000000, "ocf_ttm": 10, "net_debt_to_ebitda": 2.0, "interest_coverage": 5.0, "gross_margin": 0.4}, "risk": {"adv30_usd": 6000000}, "flags": []},
        {"ticker": "BBB", "sector": "Tech", "fundamentals": {"market_cap": 100000000, "ocf_ttm": 10, "net_debt_to_ebitda": 2.0, "interest_coverage": 5.0, "gross_margin": 0.4}, "risk": {"adv30_usd": 6000000}, "flags": []},
    ]
    filtered, rejects = pc.apply_filters(enriched)
    assert [c["ticker"] for c in filtered] == ["AAA"]
    assert rejects["mcap"] == 1

