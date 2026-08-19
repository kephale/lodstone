"""Renderer-neutral multiscale chunk planning and streaming."""

from .chunks import (
    ChunkGrid,
    chunk_boundaries,
    chunk_ids_in_region,
    chunk_shape_for,
    chunk_sizes_for,
    normalize_chunk_sizes,
    regular_chunk_sizes,
)
from .composition import fill_unloaded_chunks, nearest_resample_region
from .diagnostics import LevelDiagnosticArray, PlanComparison, PlanTrace
from .geometry import (
    anisotropic_extent_for_bytes,
    clamp_region_to_budget,
    isotropic_extent_for_bytes,
)
from .model import (
    ChunkEvent,
    ChunkState,
    Layout,
    Level,
    Plan,
    Pyramid,
    Region,
    Status,
    StreamDiagnostics,
    Tile,
    TileKey,
    Update,
    View,
    identity_transform,
)
from .planner import Planner
from .resident import (
    ResidentArrays,
    ResidentChange,
    ResidentTransition,
    ResidentWindow,
)
from .source import Source
from .stream import Stream
from .target import PassTarget, Target

__all__ = [
    "ChunkEvent",
    "ChunkGrid",
    "ChunkState",
    "Layout",
    "Level",
    "LevelDiagnosticArray",
    "PassTarget",
    "Plan",
    "PlanComparison",
    "PlanTrace",
    "Planner",
    "Pyramid",
    "Region",
    "ResidentArrays",
    "ResidentChange",
    "ResidentTransition",
    "ResidentWindow",
    "Source",
    "Status",
    "Stream",
    "StreamDiagnostics",
    "Target",
    "Tile",
    "TileKey",
    "Update",
    "View",
    "anisotropic_extent_for_bytes",
    "chunk_boundaries",
    "chunk_ids_in_region",
    "chunk_shape_for",
    "chunk_sizes_for",
    "clamp_region_to_budget",
    "fill_unloaded_chunks",
    "identity_transform",
    "isotropic_extent_for_bytes",
    "nearest_resample_region",
    "normalize_chunk_sizes",
    "regular_chunk_sizes",
]
