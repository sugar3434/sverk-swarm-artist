"""Deterministic Bitmap Generation from Prompt module for Sverk PikoClaw Swarm.

Provides deterministic CRC32-seeded raster color matrix generation from user text prompt,
and RGB color quantization to target palette.
"""
from __future__ import annotations

import binascii
import logging
import random
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger("vision.prompt_to_bitmap")

PALETTE: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0),       # Академист (чёрный контур)
    "red": (220, 30, 30),     # Экспрессионист (красный акцент)
    "blue": (30, 60, 220),    # Минималист (синий фон)
    "yellow": (245, 210, 30), # Детализатор (жёлтый блик)
}


def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    clean_hex = hex_str.lstrip("#").strip()
    if len(clean_hex) == 3:
        clean_hex = "".join([c * 2 for c in clean_hex])
    if len(clean_hex) != 6:
        raise ValueError(f"Invalid hex color format: {hex_str!r}")
    r = int(clean_hex[0:2], 16)
    g = int(clean_hex[2:4], 16)
    b = int(clean_hex[4:6], 16)
    return r, g, b


def quantize_to_palette(
    color: Union[str, Tuple[int, int, int], List[int]],
    palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> str:
    """Quantizes RGB or hex color to nearest official competition palette color."""
    target_palette = palette if palette is not None else PALETTE
    lower_palette_keys = {k.lower(): k for k in target_palette.keys()}

    if isinstance(color, str):
        val_str = color.strip().lower()
        if val_str in lower_palette_keys:
            return lower_palette_keys[val_str]
        if val_str.startswith("#") or all(c in "0123456789abcdef" for c in val_str):
            try:
                rgb = _hex_to_rgb(val_str)
            except ValueError:
                rgb = (0, 0, 0)
        else:
            logger.warning("Unknown color string %s, defaulting to 'black'.", color)
            return "black"
    elif isinstance(color, (tuple, list)) and len(color) >= 3:
        rgb = (int(color[0]), int(color[1]), int(color[2]))
    else:
        raise ValueError(f"Unsupported color format for quantization: {color!r}")

    r0, g0, b0 = rgb
    best_color_name: str = "black"
    min_dist_sq: float = float("inf")

    for col_name, (pr, pg, pb) in target_palette.items():
        dist_sq = float((r0 - pr) ** 2 + (g0 - pg) ** 2 + (b0 - pb) ** 2)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_color_name = col_name

    return best_color_name


def generate_bitmap(
    prompt: str,
    cols: int,
    rows: int,
    seed: Optional[int] = None,
) -> List[List[str]]:
    """Generates 2D bitmap matrix of palette colors deterministically from text prompt."""
    if cols <= 0 or rows <= 0:
        raise ValueError(f"Bitmap dimensions must be positive, got rows={rows}, cols={cols}")

    if seed is None:
        seed = binascii.crc32(prompt.encode("utf-8")) & 0xFFFFFFFF
    
    rng = random.Random(seed)

    text_lower = prompt.lower()
    weights: Dict[str, float] = {
        "black": 25.0,
        "red": 25.0,
        "blue": 25.0,
        "yellow": 25.0,
    }

    if any(k in text_lower for k in ("контур", "ночь", "тень", "черный", "чёрный", "black", "dark", "line", "академист")):
        weights["black"] += 40.0
    if any(k in text_lower for k in ("экспрессия", "огонь", "страсть", "красный", "red", "fire", "emotion", "экспрессионист")):
        weights["red"] += 40.0
    if any(k in text_lower for k in ("небо", "море", "вода", "синий", "холодной", "blue", "sky", "minimal", "минималист")):
        weights["blue"] += 40.0
    if any(k in text_lower for k in ("солнце", "свет", "звезды", "жёлтый", "желтый", "yellow", "sun", "light", "детализатор")):
        weights["yellow"] += 40.0

    color_choices = list(weights.keys())
    color_weights = list(weights.values())

    bitmap: List[List[str]] = []
    for r in range(rows):
        row_colors: List[str] = []
        for c in range(cols):
            selected_color = rng.choices(color_choices, weights=color_weights, k=1)[0]
            row_colors.append(selected_color)
        bitmap.append(row_colors)

    logger.info("Generated %dx%d bitmap for prompt %r (seed=%d).", cols, rows, prompt[:30], seed)
    return bitmap
