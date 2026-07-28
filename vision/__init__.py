"""Vision and raster generation modules for Sverk PikoClaw Swarm."""
from vision.prompt_to_bitmap import PALETTE, quantize_to_palette, generate_bitmap
from vision.bitmap_to_plan import coords_to_cell_id, merge_adjacent_cells, bitmap_to_tasks

__all__ = [
    "PALETTE",
    "quantize_to_palette",
    "generate_bitmap",
    "coords_to_cell_id",
    "merge_adjacent_cells",
    "bitmap_to_tasks",
]
