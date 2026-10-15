"""Vocabulary-only ``LineMesh`` operations: rename the tags, touch nothing else.

Separate from :mod:`morph <nekmeshpy.linemesh.morph>`, which is the delta-0 rung for
operations on the *geometry*. These change neither coordinates nor connectivity nor
numbering -- only what the two tag tables call things -- so a caller reading the
sibling list can tell at a glance that a retag cannot have moved a node.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..pointmesh import tag as pointmesh
from ..tags.tags import ElementTagSpec
from .linemesh import LineMesh


def retag_element(mesh: LineMesh, tags: ElementTagSpec) -> LineMesh:
    """The same mesh with its ``element_tags`` changed, by what ``tags`` is: a mapping
    renames the existing names (a tag the map does not name is left alone, the map
    applies simultaneously, a key naming nothing raises, a rename to ``NO_TAG`` untags);
    a string names every element; a ``Tags`` or per-element sequence replaces the table.
    Geometry and the side tags are untouched."""
    return LineMesh(mesh.point_mesh, mesh.lines, mesh.interior,
                    mesh.element_tags.retag(tags))


def retag_point(mesh: LineMesh, mapping: Mapping[str, str]) -> LineMesh:
    """The same mesh with its ``point_tags`` renamed through ``mapping``, rows kept in
    stored order (see :func:`retag_element` for the map's own rules).

    Renaming a tag to ``""`` **drops** its rows rather than storing an empty name: a
    side-tag table is a named subset, so leaving it is leaving the table. That is how
    a boundary name that has stopped meaning anything is retired."""
    return LineMesh(pointmesh.retag_element(mesh.point_mesh, mapping),
                    mesh.lines, mesh.interior, mesh.element_tags)


__all__ = [
    "retag_element",
    "retag_point",
]
