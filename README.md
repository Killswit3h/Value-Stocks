# Daily −40% Drop Value Screen

This project generates a daily Notion report of U.S. value stocks that suffered a large price drawdown. The workflow:

1. Pulls candidates from the configured market data provider (Polygon by default, with a deterministic sample fallback).
2. Applies drop-trigger and survivability filters.
3. Scores candidates across valuation, quality, momentum, risk and liquidity components.
4. Renders charts/background blurbs and pushes the ranked table plus risk rules to Notion.
5. Stores a deterministic JSON run-log per day.

## Requirements

* Python 3.11+
* (Optional) `pip install -r requirements.txt` for third-party helpers such as `requests`
* Notion API credentials and a database to host the daily report
* API key for the chosen data provider (Polygon is assumed in the sample config)

## Configuration

Runtime settings live in `config.yaml`. You can adjust:

* Execution parameters – time zone, max candidates, directories.
* Universe filters – minimum price, market cap and sector caps.
* Hard filters – cash flow, leverage and earnings windows.
* Scoring weights – tweak each bucket or penalty without touching code.

## Environment variables

Create a `.env` file with:

```
NOTION_API_KEY=secret_token
NOTION_DATABASE_ID=database_uuid
DATA_API_KEY=polygon_or_other_key
```

If `DATA_API_KEY` is omitted, the pipeline uses a deterministic sample backend to make dry runs reproducible.

## Running locally

```
python main.py --run --dry-run       # Executes without pushing to Notion
python main.py --run                 # Full run (requires env vars)
python main.py --run --date 2024-05-01
```

Each run writes a summary to `runs/YYYY-MM-DD.json` and saves charts under `charts/`.

## Tests

```
pytest
```

Unit coverage focuses on the scoring engine and hard filters.

## GitHub Actions

The workflow `.github/workflows/daily_screen.yml` executes `python main.py --run` every weekday at 23:00 America/New_York and supports manual dispatch. The job publishes the run-log as an artifact for traceability.

## Notion setup notes

The script treats `NOTION_DATABASE_ID` as the database receiving the daily page. Ensure it has a **title** property named `Ticker`. The script creates/updates a page titled `Daily −40% Drop Value Screen – YYYY-MM-DD` and appends the summary blocks, table, five-year toggles and rules. Re-running the same day archives the prior blocks before appending fresh content for idempotency.

## Adjusting weights

Update `config.yaml` under the `scores` section. The total score rescales automatically to 0–100 regardless of bucket weights, so you can experiment with different emphasis without rewriting the code.
