"""
Офлайн-генератор растра из текстового промпта («Рой дронов-художников»).

Задача модуля — превратить произвольную строку-промпт в детерминированный
растр cols x rows (RGB), не обращаясь к сети/LLM. Планирование целиком
укладывается в регламентный лимит 15 минут, поэтому генерация должна быть
быстрой (<1с) и стабильной: один и тот же промпт всегда даёт один и тот же
растр (важно для повторяемости на соревновании и для тестов).

Алгоритм:
1. По ключевым словам промпта (RU/EN) выбирается «сцена» — простой набор
   геометрических примитивов, которые рисуются PIL.ImageDraw на увеличенном
   холсте (по умолчанию 256x256) по белому фону: каждая фигура залита одним из 4 цветов
   палитры дронов (краска бывает только 4 цветов по регламенту) и обведена чёрным
   контуром сверху — это важно: без реальных цветовых заливок внутри фигур все
   залитые пиксели оказались бы на границе с фоном и после квантования все бы
   вынужденно стали «контуром» и принудительно красились в чёрный — такая «линия
   без заливки» оставила бы три из 4 дронов без работы, что недопустимо.
2. Если ни одно ключевое слово не найдено — рисуется абстрактная композиция,
   детерминированная по хэшу от промпта (разные промпты — разные картинки,
   один и тот же промпт — одна и та же картинка).
3. Холст уменьшается до размера сетки cols x rows (метод BOX/NEAREST) и
   яркость квантуется, получая финальный растр rows x cols троек RGB.

Дополнительно модуль умеет раскладывать растр по палитре дронов
(`quantize_to_palette`) и находить контурные ячейки (`detect_outline_cells`)
для дрона-Академиста (чёрный контур).
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw

# Размер «рабочего» холста в пикселях, на котором рисуются примитивы,
# перед финальным уменьшением до сетки полотна.
CANVAS_SIZE = 256

# Порог яркости (0-255): пиксели ярче этого значения считаются фоном
# ("не красить эту ячейку").
BACKGROUND_BRIGHTNESS_THRESHOLD = 235

Color = Tuple[int, int, int]
Bitmap = List[List[Color]]

# Палитра дронов (см. common.schema.COLORS) — конкретные эталонные RGB по
# регламенту. Определена на уровне модуля, чтобы её можно было импортировать
# из mission_runner.py и других модулей интеграции, а не только из demo-блока.
PALETTE: Dict[str, Color] = {
    "black": (20, 20, 20),
    "red": (210, 30, 30),
    "blue": (30, 60, 210),
    "yellow": (230, 200, 30),
}


# --------------------------------------------------------------------------
# Вспомогательные геометрические функции
# --------------------------------------------------------------------------

def _prompt_hash(prompt: str) -> int:
    """Стабильный (не зависящий от PYTHONHASHSEED) целочисленный хэш строки."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _draw_sun(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Солнце: жёлтый круг в центре верхней половины + красные лучи-отрезки,
    чёрный контур диска поверх заливки (чтобы @Академист обводил границу)."""
    cx, cy = size // 2, int(size * 0.32)
    # Диск и лучи должны оставаться одной связной областью после уменьшения
    # даже до сетки 4x4. Разрыв между ними раньше превращал каждый луч в
    # отдельный контур и на сетке 20x20 съедал весь лимит задач чёрным цветом.
    r = int(size * 0.19)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#e6c81e", outline="black", width=4)
    n_rays = 10
    for i in range(n_rays):
        angle = 2 * math.pi * i / n_rays
        x1 = cx + int((r - 2) * math.cos(angle))
        y1 = cy + int((r - 2) * math.sin(angle))
        x2 = cx + int((r + 24) * math.cos(angle))
        y2 = cy + int((r + 24) * math.sin(angle))
        draw.line([x1, y1, x2, y2], fill="#d21e1e", width=8)
    # Лучи рисуются после диска, поэтому восстанавливаем широкую цветную
    # сердцевину: на грубой сетке она гарантирует хотя бы одну ячейку заливки.
    inner_r = int(r * 0.78)
    draw.ellipse(
        [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
        fill="#e6c81e",
    )


def _draw_mountains(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Горы: несколько треугольников с вершинами разной высоты."""
    base_y = int(size * 0.82)
    # Один связный силуэт вместо трёх раздельных треугольников: раздельные
    # склоны давали слишком много фоновых границ и вытесняли цветную заливку.
    silhouette = [
        (0, base_y),
        (int(size * 0.18), int(size * 0.55)),
        (int(size * 0.38), int(size * 0.30)),
        (int(size * 0.58), int(size * 0.54)),
        (int(size * 0.73), int(size * 0.43)),
        (size, base_y),
    ]
    draw.polygon(silhouette, fill="#1e3cd2", outline="black")
    draw.line([(0, base_y), (size, base_y)], fill="black", width=3)


def _draw_heart(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Сердце: два круга сверху + треугольник (клин) снизу."""
    cx, cy = size // 2, int(size * 0.42)
    # Более широкая заливка сохраняет красную сердцевину на сетке 4x4.
    r = int(size * 0.23)
    draw.ellipse([cx - r * 2 + r // 2, cy - r, cx - r // 2 + r // 2, cy + r], fill="#d21e1e", outline="black", width=3)
    draw.ellipse([cx + r // 2 - r, cy - r, cx + r * 2 - r - r // 2, cy + r], fill="#d21e1e", outline="black", width=3)
    draw.polygon(
        [
            (cx - int(r * 1.8), cy + int(r * 0.3)),
            (cx + int(r * 1.8), cy + int(r * 0.3)),
            (cx, cy + int(r * 3.2)),
        ],
        fill="#d21e1e", outline="black",
    )


def _draw_star(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Пятиконечная звезда."""
    cx, cy = size // 2, size // 2
    r_out = int(size * 0.40)
    # Узкая звезда на 4x4 целиком классифицировалась как контур.
    r_in = int(size * 0.34)
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = r_out if i % 2 == 0 else r_in
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill="#e6c81e", outline="black", width=4)


def _draw_house(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Домик: квадрат-стены + треугольная крыша + дверь."""
    x0, x1 = int(size * 0.22), int(size * 0.78)
    roof_y = int(size * 0.30)
    wall_top = int(size * 0.48)
    wall_bottom = int(size * 0.80)
    draw.rectangle([x0, wall_top, x1, wall_bottom], fill="#e6c81e", outline="black", width=4)
    draw.polygon(
        [(x0 - 10, wall_top), ((x0 + x1) // 2, roof_y), (x1 + 10, wall_top)],
        fill="#d21e1e", outline="black",
    )
    door_w = (x1 - x0) // 5
    cx = (x0 + x1) // 2
    draw.rectangle([cx - door_w // 2, wall_bottom - int(size * 0.18), cx + door_w // 2, wall_bottom], fill="#141414", outline="black", width=3)


def _draw_tree(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Ёлка/дерево: стек треугольников + ствол."""
    cx = size // 2
    base_y = int(size * 0.82)
    trunk_h = int(size * 0.10)
    draw.rectangle([cx - 6, base_y, cx + 6, base_y + trunk_h], fill="#141414", outline="black", width=3)
    widths = [int(size * 0.30), int(size * 0.24), int(size * 0.18)]
    top = int(size * 0.18)
    step = int(size * 0.18)
    for i, w in enumerate(widths):
        y_bottom = base_y - i * step + 10
        y_top = top + i * (step * 0.4)
        draw.polygon(
            [(cx - w, y_bottom), (cx + w, y_bottom), (cx, y_top)],
            fill="#1e3cd2", outline="black",
        )


def _draw_flower(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Цветок: несколько лепестков-кругов вокруг центрального круга + стебель."""
    cx, cy = size // 2, int(size * 0.38)
    r_petal = int(size * 0.12)
    r_orbit = int(size * 0.16)
    n_petals = 6
    for i in range(n_petals):
        angle = 2 * math.pi * i / n_petals
        px = cx + int(r_orbit * math.cos(angle))
        py = cy + int(r_orbit * math.sin(angle))
        draw.ellipse([px - r_petal, py - r_petal, px + r_petal, py + r_petal], fill="#d21e1e", outline="black", width=2)
    r_center = int(size * 0.09)
    draw.ellipse([cx - r_center, cy - r_center, cx + r_center, cy + r_center], fill="#e6c81e", outline="black", width=2)
    draw.line([(cx, cy + r_orbit + r_petal), (cx, int(size * 0.90))], fill="#1e3cd2", width=5)


def _draw_animal_face(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Упрощённая мордочка животного (кот/собака): круг + треугольные уши + глаза."""
    cx, cy = size // 2, int(size * 0.52)
    # Крупная мордочка оставляет цветные внутренние ячейки даже на 4x4.
    r = int(size * 0.36)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#e6c81e", outline="black", width=4)
    ear = int(size * 0.14)
    draw.polygon(
        [(cx - r, cy - int(r * 0.6)), (cx - r + ear, cy - r - ear), (cx - int(r * 0.2), cy - r)],
        fill="#d21e1e", outline="black",
    )
    draw.polygon(
        [(cx + r, cy - int(r * 0.6)), (cx + r - ear, cy - r - ear), (cx + int(r * 0.2), cy - r)],
        fill="#d21e1e", outline="black",
    )
    eye_r = max(2, r // 8)
    draw.ellipse([cx - r // 2 - eye_r, cy - eye_r, cx - r // 2 + eye_r, cy + eye_r], fill="black")
    draw.ellipse([cx + r // 2 - eye_r, cy - eye_r, cx + r // 2 + eye_r, cy + eye_r], fill="black")
    draw.polygon(
        [(cx - 6, cy + int(r * 0.25)), (cx + 6, cy + int(r * 0.25)), (cx, cy + int(r * 0.40))],
        outline="black",
    )


def _draw_wave(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Волна/море: несколько горизонтальных синусоид."""
    n_lines = 3
    amplitude = int(size * 0.06)
    for line_idx in range(n_lines):
        base_y = int(size * (0.40 + line_idx * 0.18))
        pts = []
        for x in range(0, size + 1, 4):
            y = base_y + int(amplitude * math.sin((x / size) * 4 * math.pi + line_idx))
            pts.append((x, y))
        draw.line(pts, fill="#1e3cd2", width=6)


def _draw_abstract(draw: ImageDraw.ImageDraw, size: int, prompt: str) -> None:
    """Абстрактная композиция по умолчанию: детерминирована по хэшу промпта."""
    h = _prompt_hash(prompt)
    rng_state = h
    n_shapes = 4 + (h % 4)  # 4..7 фигур

    def next_val(modulo: int) -> int:
        nonlocal rng_state
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) % (2**64)
        return rng_state % modulo

    # Базовая связная цветная плоскость нужна не только эстетически: у набора
    # мелких разрозненных фигур на сетках 4x4/8x8 каждая ячейка оказывалась
    # границей с фоном и затем принудительно становилась чёрным контуром.
    margin = int(size * 0.15)
    base_colors = ("#d21e1e", "#1e3cd2", "#e6c81e")
    draw.rectangle(
        [margin, margin, size - margin, size - margin],
        fill=base_colors[h % len(base_colors)],
        outline="black",
        width=3,
    )

    for _ in range(n_shapes):
        shape_kind = next_val(3)
        span = size - 2 * margin
        x0 = margin + next_val(span)
        y0 = margin + next_val(span)
        w = size // 6 + next_val(size // 3)
        palette_cycle = ("#d21e1e", "#1e3cd2", "#e6c81e", "#141414")
        fill_color = palette_cycle[next_val(len(palette_cycle))]
        if shape_kind == 0:
            draw.ellipse(
                [x0, y0, min(size - margin, x0 + w), min(size - margin, y0 + w)],
                fill=fill_color,
                outline="black",
                width=3,
            )
        elif shape_kind == 1:
            x1 = min(size - margin, x0 + w)
            y1 = min(size - margin, y0 + w)
            draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline="black", width=3)
        else:
            x1 = margin + next_val(span)
            y1 = margin + next_val(span)
            draw.line([x0, y0, x1, y1], fill=fill_color, width=5)


# Ключевые слова (RU/EN) -> функция рисования. Порядок важен: первое
# совпадение в промпте определяет сцену.
_KEYWORD_SCENES = [
    (("солнце", "sun"), _draw_sun),
    (("сердце", "heart"), _draw_heart),
    (("звезда", "звезду", "star"), _draw_star),
    (("дом", "house"), _draw_house),
    (("ёлка", "елка", "ёлку", "елку", "дерево", "деревья", "tree"), _draw_tree),
    (("гора", "горы", "гор", "mountain"), _draw_mountains),
    (("цветок", "цветы", "flower"), _draw_flower),
    (("кот", "кошка", "собака", "пёс", "пес", "животное", "cat", "dog", "animal"), _draw_animal_face),
    (("волна", "волны", "море", "wave", "sea", "ocean"), _draw_wave),
]


def _select_scenes(prompt: str) -> List[str]:
    """Возвращает список имён сцен, совпавших по ключевым словам промпта."""
    lowered = prompt.lower()
    matched: List[str] = []
    for keywords, _fn in _KEYWORD_SCENES:
        if any(kw in lowered for kw in keywords):
            matched.append(keywords[0])
    return matched


def _render_canvas(prompt: str) -> Image.Image:
    """Рисует холст CANVAS_SIZE x CANVAS_SIZE по ключевым словам промпта."""
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), color="white")
    draw = ImageDraw.Draw(img)

    lowered = prompt.lower()
    drawn_any = False
    for keywords, fn in _KEYWORD_SCENES:
        if any(kw in lowered for kw in keywords):
            fn(draw, CANVAS_SIZE)
            drawn_any = True

    if not drawn_any:
        _draw_abstract(draw, CANVAS_SIZE, prompt)

    return img


# --------------------------------------------------------------------------
# Публичный API
# --------------------------------------------------------------------------

def generate_bitmap(prompt: str, cols: int, rows: int) -> Bitmap:
    """
    Детерминированно генерирует растр rows x cols из текстового промпта.

    Никаких обращений к сети или LLM: рисование ведётся простыми
    геометрическими примитивами PIL по ключевым словам промпта (RU/EN),
    либо абстрактной композицией, если ключевые слова не найдены.

    Параметры:
        prompt: текстовое описание картины (RU/EN).
        cols:   число столбцов сетки полотна.
        rows:   число строк сетки полотна.

    Возвращает:
        список списков (rows x cols) троек RGB (0-255).
    """
    if cols <= 0 or rows <= 0:
        raise ValueError("cols и rows должны быть положительными")

    canvas = _render_canvas(prompt)
    # Уменьшаем холст до размера сетки: BOX хорошо усредняет контуры контурных
    # линий в яркость ячейки (антиалиасинг), что удобно для последующего
    # порогового квантования по палитре.
    small = canvas.resize((cols, rows), resample=Image.BOX)

    pixels = small.load()
    bitmap: Bitmap = []
    for row in range(rows):
        row_colors: List[Color] = []
        for col in range(cols):
            r, g, b = pixels[col, row][:3]
            row_colors.append((int(r), int(g), int(b)))
        bitmap.append(row_colors)
    return bitmap


def _brightness(color: Color) -> float:
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b


def quantize_to_palette(
    bitmap: Bitmap,
    palette: Dict[str, Color],
) -> List[List[Optional[str]]]:
    """
    Приводит растр к палитре дронов.

    Для каждого пикселя ищется ближайший (по Евклидову расстоянию в RGB)
    цвет палитры. Если пиксель слишком светлый (яркость выше
    BACKGROUND_BRIGHTNESS_THRESHOLD) — считается фоном, и ячейка помечается
    None ("не красить").

    Параметры:
        bitmap:  растр rows x cols троек RGB (см. generate_bitmap).
        palette: словарь {имя_цвета: (r,g,b)}, напр. COLORS из common.schema.

    Возвращает:
        список списков rows x cols, элементы — имя цвета из palette либо None.
    """
    result: List[List[Optional[str]]] = []
    for row in bitmap:
        row_out: List[Optional[str]] = []
        for color in row:
            if _brightness(color) > BACKGROUND_BRIGHTNESS_THRESHOLD:
                row_out.append(None)
                continue

            best_name: Optional[str] = None
            best_dist = float("inf")
            for name, pal_color in palette.items():
                dr = color[0] - pal_color[0]
                dg = color[1] - pal_color[1]
                db = color[2] - pal_color[2]
                dist = dr * dr + dg * dg + db * db
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
            row_out.append(best_name)
        result.append(row_out)
    return result


def detect_outline_cells(quantized: List[List[Optional[str]]]) -> Set[Tuple[int, int]]:
    """
    Находит контурные ячейки растра (для дрона-Академиста, рисующего контур).

    Простая эвристика: ячейка считается контурной, если она сама закрашена
    (не None) и хотя бы один из 4 соседей (сверху/снизу/слева/справа) —
    фон (None) или выходит за границы растра (край картины тоже контур).

    Параметры:
        quantized: результат quantize_to_palette (rows x cols, имя цвета|None).

    Возвращает:
        множество пар (row, col) — координаты контурных ячеек.
    """
    outline: Set[Tuple[int, int]] = set()
    rows = len(quantized)
    if rows == 0:
        return outline
    cols = len(quantized[0])

    for r in range(rows):
        for c in range(cols):
            if quantized[r][c] is None:
                continue
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            is_outline = False
            for nr, nc in neighbors:
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    is_outline = True
                    break
                if quantized[nr][nc] is None:
                    is_outline = True
                    break
            if is_outline:
                outline.add((r, c))
    return outline


# --------------------------------------------------------------------------
# Демонстрация
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from common.schema import COLORS

    assert set(PALETTE.keys()) == set(COLORS)

    demo_prompt = "нарисуй солнце над горами"
    demo_cols, demo_rows = 24, 24

    bmp = generate_bitmap(demo_prompt, demo_cols, demo_rows)
    quant = quantize_to_palette(bmp, PALETTE)
    outline = detect_outline_cells(quant)

    print(f"Промпт: {demo_prompt!r}, сетка: {demo_cols}x{demo_rows}")
    print(f"Закрашенных ячеек: {sum(1 for row in quant for v in row if v is not None)}")
    print(f"Контурных ячеек: {len(outline)}")
    print()

    for r, row in enumerate(quant):
        line = []
        for c, val in enumerate(row):
            if val is None:
                line.append(".")
            else:
                symbol = val[0].upper()
                if (r, c) in outline:
                    symbol = symbol.lower()
                line.append(symbol)
        print("".join(line))
