"""Vocabulary-only ``PointMesh`` operations: rename the tags, touch nothing else."""

from __future__ import annotations

from collections.abc import Mapping

from ..tags.tags import ElementTagSpec
from .pointmesh import PointMesh


def retag_element(mesh: PointMesh, tags: ElementTagSpec) -> PointMesh:
    """The same mesh with its ``element_tags`` changed, by what ``tags`` is: a mapping
    renames the existing names (a tag the map does not name is left alone, the map
    applies simultaneously, a key naming nothing raises, a rename to ``NO_TAG`` untags);
    a string names every element; a ``Tags`` or per-element sequence replaces the table.
    Geometry and the side tags are untouched."""
    return PointMesh(mesh.points,
                     mesh.element_tags.retag(tags))


__all__ = [
    "retag_element",
]
