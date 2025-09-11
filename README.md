# Daily −40% Drop Value Screen

This project builds a daily Notion report of U.S. value stocks that experienced a large drop, with risk-aware scoring and context.

## Features

- Universe: U.S. common stocks on NYSE/Nasdaq/AMEX with liquidity and survivability filters.
- Triggers:
  - 1-day crash: today ≤ −40% change
  - Deep drawdown: close ≥ 40% below 52w high AND today ≤ −8% (expandable to −4% if too few)
- Fundamentals, quality, risk, and momentum metrics with a composite Expected Value Score (0–100).
- Notion integration (official API via HTTPS) with idempotent upserts per Ticker + date.
- Daily page with summary, ranked table, per-ticker analysis sections, and rules.
- Five-year chart URLs (uses QuickChart external URLs by default).
- Caching, retry with backoff, deterministic run log `runs/YYYY-MM-DD.json`.
- Dry run mode to print Notion payloads instead of sending.
- GitHub Actions workflow scheduled at 23:00 America/New_York Mon–Fri.

## Requirements

- Python 3.11+
- Accounts/API keys:
  - Polygon.io (market data + fundamentals + news)
  - Notion internal integration key

## Setup

1. Create a Notion database. Copy its Database ID (from the URL).
2. Share the database with your integration (Notion → Share → Invite).
3. Create a `.env` file in the project root:

```
NOTION_API_KEY=secret_...
NOTION_DATABASE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DATA_API_KEY=polygon_api_key_here
TZ=America/New_York
```

4. Install dependencies:

```
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

5. Configure thresholds/weights in `config.yaml` as desired.

## Run

Dry run (no Notion writes):

```
python main.py --dry-run
```

Execute a full run:

```
python main.py --run
```

The script will:
- Determine today’s U.S. trading day
- Build the candidate set via Polygon grouped daily bars
- Fetch required fundamentals/series for candidates
- Apply filters and scoring
- Create or update the Notion daily page and the top 12 entries in the target database
- Write a run log at `runs/YYYY-MM-DD.json`

Re-running the same day updates in place.

## Notion Database Schema

On first run, the script attempts to add missing properties to the target database:

- Ticker (title), Company (rich_text), Report Date (date)
- Score (number)
- Trigger (select: "1-day −40%", "52w −40% + down day")
- % Today, % 5D, % 1M, % 1Y (number)
- FCF Yield %, EV/EBITDA, P/E, P/B, ROIC %, Rev 3y CAGR % (number)
- Net Debt/EBITDA, Interest Coverage (number)
- Beta, 30D Realized Vol %, Short Interest % (number)
- Avg $ Vol (30D) (number)
- Earnings Window (select: Safe / Near / Inside 2d / Unknown)
- Flags (multi_select)
- 5-Year Chart (url)
- Background (rich_text)

Each day also creates/updates a daily report page (as a page with content blocks) with:
- Header summary + callout
- Ranked table (as a Notion table block) of the top 12
- Per-ticker collapsible analysis blocks
- Rules section

Note: Page embeds a table block; ticker rows are also added to the database for tracking.

## Configuration

See `config.yaml`. You can tweak:

- Liquidity and survivability thresholds
- Trigger thresholds
- Scoring weights (Valuation/Quality/Momentum/Risk/Liquidity/Events)
- Sector caps and pick limits

## Testing

Run unit tests:

```
pytest -q
```

Tests cover scoring normalization and hard filters using fixtures.

## GitHub Actions

Workflow at `.github/workflows/daily.yml` runs at 23:00 America/New_York Mon–Fri and can be dispatched manually. It uses the following repository secrets:

- `NOTION_API_KEY`
- `NOTION_DATABASE_ID`
- `DATA_API_KEY`

## Notes & Limitations

- Trading halts detection and insider buys are best-effort; if unavailable, metrics are neutral or skipped with proportional reweighting.
- Short interest may be unavailable via Polygon; if missing, scoring reweights the Liquidity/Ownership bucket.
- Sector classification uses issuer data (SIC/industry) and may not perfectly map to GICS.
- Five-year charts use external URLs (QuickChart) by default.
- If fewer than 5 names pass, the script expands criteria automatically and flags the report.

This is not financial advice.

