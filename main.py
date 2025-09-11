import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from dateutil import tz

from dotenv import load_dotenv
import yaml

from src.logging_utils import get_logger
from src.pipeline import run_daily


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily −40% Drop Value Screen → Notion")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true", help="Execute and write to Notion")
    g.add_argument("--dry-run", action="store_true", help="Print Notion payloads only")
    return p.parse_args()


def main():
    load_dotenv()

    args = parse_args()
    cfg = load_config()
    logger = get_logger()

    notion_key = os.getenv("NOTION_API_KEY")
    notion_db = os.getenv("NOTION_DATABASE_ID")
    data_key = os.getenv("DATA_API_KEY")

    if not notion_key or not notion_db or not data_key:
        missing = [k for k, v in {
            "NOTION_API_KEY": notion_key,
            "NOTION_DATABASE_ID": notion_db,
            "DATA_API_KEY": data_key,
        }.items() if not v]
        logger.error(f"Missing required secrets: {', '.join(missing)}")
        raise SystemExit(2)

    tzname = os.getenv("TZ", cfg.get("timezone", "America/New_York"))
    run_tz = tz.gettz(tzname)
    now = datetime.now(run_tz)

    try:
        run_daily(
            cfg=cfg,
            secrets={
                "NOTION_API_KEY": notion_key,
                "NOTION_DATABASE_ID": notion_db,
                "DATA_API_KEY": data_key,
            },
            when=now,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.exception("Fatal error during run")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

