"""CPU foreground occlusion augmentation with explicit masks."""

from .core import OcclusionResult, VisibilityResult, compose_occlusion, compute_visibility

__all__ = ["OcclusionResult", "VisibilityResult", "compose_occlusion", "compute_visibility"]
