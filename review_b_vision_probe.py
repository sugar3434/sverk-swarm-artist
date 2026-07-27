from collections import Counter
from dataclasses import asdict

from common.schema import PaintTask
from vision.prompt_to_bitmap import PALETTE, generate_bitmap, quantize_to_palette
from vision.bitmap_to_plan import bitmap_to_tasks, merge_adjacent_same_color

PROMPTS = [
    "нарисуй солнце над горами",
    "нарисуй сердце",
    "нарисуй звезду",
    "abc xyz непонятный промпт без ключевых слов",
    "",
    "нарисуй кота и собаку",
]
SIZES = [4, 8, 16, 20]

for size in SIZES:
    for prompt in PROMPTS:
        bmp = generate_bitmap(prompt, size, size)
        quant = quantize_to_palette(bmp, PALETTE)
        tasks = bitmap_to_tasks(quant)
        merged = merge_adjacent_same_color(tasks)
        for task in tasks + merged:
            PaintTask(**{k: v for k, v in asdict(task).items() if k in {"cell", "color", "duration_s", "passes", "priority", "x", "y", "z", "note"}})
        colors = Counter(t.color for t in tasks)
        print(size, repr(prompt), len(tasks), len(merged), dict(colors), "MULTI" if len(colors) > 1 else "SINGLE")
