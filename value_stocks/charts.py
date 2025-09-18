"""Chart generation utilities using only the standard library."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Sequence

from .models import PricePoint
from .utils import ensure_dir


def generate_price_chart(
    ticker: str, history: Sequence[PricePoint], output_dir: str | Path
) -> Path:
    """Render a minimalist PNG line chart for the last five years."""

    ensure_dir(output_dir)
    path = Path(output_dir) / f"{ticker}_5y.png"
    width, height = 640, 320
    pixels = [255, 255, 255] * width * height

    if history:
        closes = [point.close for point in history]
        min_price = min(closes)
        max_price = max(closes)
        price_range = max(max_price - min_price, 1e-6)
        step = max(1, len(closes) // width)
        sampled = closes[-width * step :]
        points = [sampled[i] for i in range(0, len(sampled), step)]
        if len(points) < width:
            pad = [points[-1]] * (width - len(points))
            points.extend(pad)
        for x, price in enumerate(points[:width]):
            norm = (price - min_price) / price_range
            y = height - 20 - int(norm * (height - 40))
            for dy in range(-1, 2):
                yy = max(0, min(height - 1, y + dy))
                idx = (yy * width + x) * 3
                pixels[idx : idx + 3] = [31, 119, 180]

    _write_png(path, width, height, pixels)
    return path


def _write_png(path: Path, width: int, height: int, pixels: Sequence[int]) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw_rows = []
    for y in range(height):
        start = y * width * 3
        end = start + width * 3
        raw_rows.append(b"\x00" + bytes(pixels[start:end]))
    compressed = zlib.compress(b"".join(raw_rows), level=9)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_bytes = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    with path.open("wb") as handle:
        handle.write(png_bytes)
