"""Backward-compatible pipeline module.

This module re-exports the shared pipeline builder from the model module for
existing imports and scripts.
"""

from __future__ import annotations

from src.model import build_pipeline

pipeline = build_pipeline()
