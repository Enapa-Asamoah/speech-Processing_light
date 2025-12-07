"""
Model architectures

Expose a `BaselineModel` symbol for backward compatibility. The baseline
module defines several architectures; use `CNN2D` as the default baseline
model and provide the `get_model` factory.
"""

from .baseline import MobileNetBaseline, EfficientNetBaseline, get_model

# Backwards-compatible name expected by some scripts
BaselineModel = MobileNetBaseline

__all__ = ["BaselineModel", "get_model", "MobileNetBaseline"]
