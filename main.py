"""Entry point for the Daily −40% Drop Value Screen."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from value_stocks.config_loader import load_config
from value_stocks.report import create_runner
from value_stocks.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily −40% Drop Value Screen")
    parser.add_argument("--run", action="store_true", help="Execute the pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Do not push to Notion")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Override run date in YYYY-MM-DD (for testing)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("value_stocks.main")
    if not args.run:
        logger.error("Use --run to execute the screen")
        return 1

    _load_env_file()
    env = {
        "NOTION_API_KEY": os.getenv("NOTION_API_KEY"),
        "NOTION_DATABASE_ID": os.getenv("NOTION_DATABASE_ID"),
        "DATA_API_KEY": os.getenv("DATA_API_KEY"),
    }
    config = load_config("config.yaml")
    runner = create_runner(config, env)
    run_date = None
    if args.date:
        try:
            run_date = datetime.fromisoformat(args.date).date()
        except ValueError as exc:
            logger.error("Invalid --date value: %s", exc)
            return 1
    try:
        metadata = runner.run(run_date=run_date, dry_run=args.dry_run)
    except Exception as exc:  # pragma: no cover - top-level guard
        logger.exception("Pipeline failed: %s", exc)
        return 1
    logger.info(
        "Completed run: %s candidates considered, %s passed",
        metadata.considered,
        metadata.passed_filters,
    )
    return 0


def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key and value:
                os.environ.setdefault(key.strip(), value.strip())


if __name__ == "__main__":
    sys.exit(main())
