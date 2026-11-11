"""CPU composition augmentation with explicit coordinate transformations."""

from .transforms import (CompositionResult, apply_composition, clip_polygon,
                         describe_polygon, plan_transform, polygon_area,
                         transform_points, validate_preset)

__all__ = ["CompositionResult", "apply_composition", "clip_polygon", "describe_polygon",
           "plan_transform", "polygon_area", "transform_points", "validate_preset"]
