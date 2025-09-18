"""Utility helpers."""
from __future__ import annotations

import functools
import json
import logging
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, TypeVar

from zoneinfo import ZoneInfo

T = TypeVar("T")


def setup_logger(name: str = "value_stocks") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def retry_with_backoff(
    retries: int = 5,
    base_delay: float = 1.0,
    jitter: float = 0.25,
    allowed_exceptions: Tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Simple retry decorator with exponential backoff and jitter."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            delay = base_delay
            while True:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions:
                    attempt += 1
                    if attempt > retries:
                        raise
                    sleep_for = delay + random.uniform(0, jitter)
                    time.sleep(sleep_for)
                    delay *= 2
        return wrapper

    return decorator


def in_memory_cache(ttl_seconds: int = 900) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Cache decorator for pure functions within a run."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[str, Tuple[float, T]] = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = json.dumps([args, kwargs], sort_keys=True, default=str)
            now = time.time()
            if key in cache:
                ts, value = cache[key]
                if now - ts <= ttl_seconds:
                    return value
            value = func(*args, **kwargs)
            cache[key] = (now, value)
            return value

        return wrapper

    return decorator


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def nyc_now(tz_name: str = "America/New_York") -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def format_percent(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def trading_day_for(date_input: datetime, tz_name: str = "America/New_York") -> date:
    local_date = date_input.astimezone(ZoneInfo(tz_name)).date()
    return local_date


def is_weekend(check_date: date) -> bool:
    return check_date.weekday() >= 5


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def date_range(start: date, end: date) -> Iterable[date]:
    delta = end - start
    for i in range(delta.days + 1):
        yield start + timedelta(days=i)
