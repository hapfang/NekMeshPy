"""Read-only ``QuadMesh`` queries -- the operations that leave the ladder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    Point,
    PointArray,
    StrArray,
    Vec3,
)
from ..core import conform, frames, measure
from ..core.interp import corner_indices
from ..core.quality import QualitySummary
from ..core.topology import TagReport, TopologyReport
from .quadmesh import QuadMesh


def _boundary_mask(quads: IntArray) -> tuple[IntArray, BoolArray]:
    """``(edges, is_boundary)``: every quad edge ``(4M,2)`` element-major
    (row ``4q+e``), and a mask of those borne by a single quad."""
    Q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    edges: IntArray = Q[:, QuadMesh.EDGE_POINTS].reshape(-1, 2)
    keys = np.sort(edges, axis=1)
    _, inverse, counts = conform.unique_rows(keys, return_counts=True)
    return edges, counts[inverse] == 1

def boundary_edges(mesh: QuadMesh) -> IntArray:
    """``(K,2)`` array of ``[quad id, local edge (1-4)]`` for every edge on
    the section boundary (borne by a single quad)."""
    _, mask = _boundary_mask(mesh.corners)
    rows = np.flatnonzero(mask)
    return np.column_stack([rows // 4, rows % 4 + 1]).astype(np.int64)

def boundary_elements(mesh: QuadMesh) -> IntArray:
    """Sorted unique quad ids with at least one edge on the section boundary."""
    return np.unique(boundary_edges(mesh)[:, 0])

def boundary_points(mesh: QuadMesh) -> IntArray:
    """Sorted unique point ids lying on the section boundary."""
    edges, mask = _boundary_mask(mesh.corners)
    be = edges[mask]
    return np.unique(be) if be.size else np.zeros(0, dtype=np.int64)

def scaled_jacobian(mesh: QuadMesh, *, order: int | None = None) -> FloatArray:
    """Per-quad minimum scaled Jacobian ``(n_quads,)``, read off the **curved**
    element the mesh actually stores.

    There is deliberately no corner-only reading. A corner scaled Jacobian cannot see
    where the high-order nodes went, so it reports a contented number for a mesh whose
    interior nodes are anywhere at all -- a node moved clean outside the element still
    scores the same. Anything that has to be trusted must read the curved block.

    ``order`` samples that block on a finer GLL lattice than the mesh's own -- what a
    solver running at ``lx1 = order`` does to it. The default reads the mesh's own
    order, where the value is exact at the nodes and silent between them: positive
    there is not a proof the element is not folded."""
    from . import quality
    return quality.scaled_jacobian(mesh, mesh.order if order is None else order)

def quality_summary(mesh: QuadMesh, *, order: int | None = None) -> QualitySummary:
    """Aggregate scaled-Jacobian statistics over the **curved** elements -- see
    :func:`scaled_jacobian <nekmeshpy.quadmesh.query.scaled_jacobian>`."""
    from . import quality
    return quality.summary(mesh, mesh.order if order is None else order)

def plane_normal(mesh: QuadMesh, *,
                 hint: Vec3 | Sequence[float] | None = None,
                 check: bool = True) -> Vec3:
    """The unit normal of the plane a **planar** section lies in: the smallest right
    singular vector of its centred points, i.e. the exact least-squares plane."""
    normal = None
    if not check:
        P: PointArray = np.asarray(mesh.points, dtype=float)
        c = P.mean(axis=0)
        normal = np.linalg.svd(P - c)[2][2]
    R, _ = frames.plane_frame(mesh.points, normal=normal, hint=hint)
    w: Vec3 = R[:, 2]
    return w


def element_blocks(mesh: QuadMesh) -> PointArray:
    """``(Q, (order+1)**2, 3)`` -- each quad's own node block, assembled natively from the
    B-rep: shared corners, then the shared edge-interior nodes in element traversal order,
    then the private per-quad interiors, each written at its lattice slot.  Nothing is
    resampled or deduplicated.

    The quad rung of :func:`linemesh.element_blocks
    <nekmeshpy.linemesh.query.element_blocks>`, one dimension up."""
    order = mesh.order
    row = order + 1
    out: PointArray = np.empty((mesh.n_quads, row * row, 3), dtype=float)
    out[:, corner_indices(order, 2), :] = mesh.points[mesh.corners]
    out[:, conform._edge_slots(2, order)[:, 1:-1], :] = conform.gather_edge_nodes(
        mesh.line_mesh.interior, mesh.quads, mesh.orient)
    out[:, conform._interior_slots(2, order), :] = mesh.interior
    return out


def _blocks(mesh: QuadMesh, high_order: bool) -> PointArray:
    """The node blocks a measure integrates: the curved ones the mesh stores, or the
    straight-sided corner blocks it reduces to."""
    if high_order:
        return element_blocks(mesh)
    return measure.corner_blocks(mesh.points, mesh.corners, 2)


def bounds(mesh: QuadMesh, *, high_order: bool = False) -> measure.Bounds:
    """The axis-aligned bounding box of the section's nodes -- corners only unless
    ``high_order=True`` asks for the stored interior nodes too.  See
    :func:`linemesh.bounds <nekmeshpy.linemesh.query.bounds>` for why neither reading
    bounds the polynomial itself."""
    return measure.bounds_of(_blocks(mesh, high_order) if high_order else mesh.points)


def element_areas(mesh: QuadMesh, *, high_order: bool = False) -> FloatArray:
    """``(Q,)`` surface area of each quad, integrated over the element as it sits in
    3-D (so a warped quad is measured as the ruled surface it is, not as a projection).

    Bilinear corners by default; ``high_order=True`` integrates the curved element the
    mesh stores.  A curved area is not a polynomial -- the integrand carries a square
    root -- so that reading is a quadrature approximation converging with the order,
    where :func:`hexmesh.element_volumes
    <nekmeshpy.hexmesh.query.element_volumes>` is exact."""
    return measure.integrate(_blocks(mesh, high_order), 2)[0]


def area(mesh: QuadMesh, *, high_order: bool = False) -> float:
    """Total surface area -- :func:`element_areas` summed.

    The wetted area of a block's wall is this over the wall's own mesh:
    ``quadmesh.area(hexmesh.boundary_mesh(block, "wall"))``."""
    return float(element_areas(mesh, high_order=high_order).sum())


def centroid(mesh: QuadMesh, *, high_order: bool = False) -> Point:
    """The **area-weighted** centroid ``integral x dA / integral dA`` -- the mass
    property, not the mean of the points."""
    return measure.centroid_of(_blocks(mesh, high_order), 2)


def tagged_edges(mesh: QuadMesh, tag: str) -> IntArray:
    """The **shared-edge ids** carrying ``tag``, ascending -- the quad rung's
    counterpart of :func:`hexmesh.tagged_faces
    <nekmeshpy.hexmesh.query.tagged_faces>`, and what
    :func:`quadmesh.attach <nekmeshpy.quadmesh.assemble.attach>` pairs.

    A tag that names nothing raises rather than returning an empty group."""
    t = mesh.edge_tags
    return t.flatnonzero(tag, strict=True)


def boundary_edge_ids(mesh: QuadMesh) -> BoolArray:
    """``(n_edges,)`` mask of the shared edges carried by exactly one quad.

    The edge-id form of :func:`boundary_edges`, and the cheaper one: the edges are
    already deduplicated in ``line_mesh``, so this is a bincount over the incidence.
    It is how a name is checked against the topology it was meant for --
    ``edge_tags.ids`` outside this mask are the tagged interior edges
    :func:`tag_report` counts."""
    return np.asarray(
        np.bincount(np.asarray(mesh.quads, dtype=np.int64).ravel(),
                    minlength=mesh.line_mesh.n_lines) == 1, dtype=bool)


def edge_rows(mesh: QuadMesh, edges: IntArray) -> tuple[IntArray, IntArray]:
    """``((K,2) [quad, local edge 1-4] rows, (len(edges),) occurrence counts)`` -- the
    inverse of ``quads``, for the given shared-edge ids; the quad rung's
    :func:`hexmesh.face_rows <nekmeshpy.hexmesh.query.face_rows>`.

    A **boundary** edge is carried by one quad and yields one row, an **interior** one
    by two and yields two, so ``counts`` is how a caller expands a per-edge value onto
    the rows it produced."""
    ids: IntArray = np.asarray(edges, dtype=np.int64).reshape(-1)
    if not ids.shape[0]:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    flat: IntArray = np.asarray(mesh.quads, dtype=np.int64).ravel()
    order = np.argsort(flat, kind="stable")
    lo = np.searchsorted(flat[order], ids, side="left")
    hi = np.searchsorted(flat[order], ids, side="right")
    counts: IntArray = (hi - lo).astype(np.int64)
    bounds: list[tuple[int, int]] = list(
        zip(np.asarray(lo, dtype=np.int64).tolist(),
            np.asarray(hi, dtype=np.int64).tolist()))
    picks: list[IntArray] = [np.arange(a, b, dtype=np.int64) for a, b in bounds]
    slots: IntArray = order[np.concatenate(picks) if picks
                            else np.zeros(0, dtype=np.int64)]
    return np.column_stack([slots // 4, slots % 4 + 1]).astype(np.int64), counts


def edge_tag_rows(mesh: QuadMesh) -> tuple[IntArray, StrArray]:
    """``((K,2) [quad, local edge 1-4] rows, their tags)`` for every named edge,
    lexsorted by ``(quad, edge)`` -- :func:`edge_rows` run over ``edge_tags.ids``; the
    quad rung's :func:`hexmesh.face_tag_rows <nekmeshpy.hexmesh.query.face_tag_rows>`."""
    named = mesh.line_mesh.element_tags
    if not len(named):
        return np.zeros((0, 2), dtype=np.int64), named.tags
    rows, counts = edge_rows(mesh, named.ids)
    tags: StrArray = np.repeat(named.tags, counts)
    p = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[p], tags[p]


def topology_report(mesh: QuadMesh) -> TopologyReport:
    """Watertightness / connectivity report of the mesh."""
    from ..core import topology
    return topology.quad_report(mesh.points, mesh.corners)


def is_watertight(mesh: QuadMesh) -> bool:
    """``True`` if the mesh boundary is a closed curve and the mesh is a single
    connected component. Does not imply conformity."""
    rep = topology_report(mesh)
    return rep.watertight and rep.n_components == 1


def is_conforming(mesh: QuadMesh) -> bool:
    """``True`` if the mesh has no hanging points (no T-junctions)."""
    return topology_report(mesh).conformal


def is_overlap_free(mesh: QuadMesh) -> bool:
    """``True`` if no two quads geometrically overlap. Independent of
    :func:`is_watertight` and :func:`is_conforming`, which are purely topological
    (facet-sharing) checks, and computed only when asked for, as at the hex rung."""
    from ..core import topology
    return topology.is_overlap_free_quads(mesh.points, mesh.corners)


def tag_report(mesh: QuadMesh) -> TagReport:
    """Cross-check ``edge_tags`` against the topological boundary (see
    :class:`TagReport <nekmeshpy.core.topology.TagReport>`)."""
    on_boundary = boundary_edge_ids(mesh)
    et = mesh.edge_tags
    named = et.is_tagged()
    return TagReport(len(et),
                     int(np.count_nonzero(on_boundary & ~named)),
                     int(np.count_nonzero(named & ~on_boundary)))


def report(mesh: QuadMesh) -> str:
    """Human-readable summary: element/point counts, scaled-Jacobian quality (curved,
    and on the solver-order lattice), per-name tagged-edge counts cross-checked against
    the boundary (:func:`tag_report`), and the topology report -- the quad rung's
    :func:`hexmesh.report <nekmeshpy.hexmesh.query.report>`.  There is no corner-only
    reading and no linear one: ``.re2`` is hex-only, so nothing here is exported flat."""
    import logging

    from ..core import topology
    from . import quality
    lines = ["%d quad elements, %d points" % (mesh.n_quads, mesh.n_points)]
    lines.append(quality.format_report(quality_summary(mesh)))
    scan = quality.order_scan(mesh)
    lines.append(quality.format_scan(scan, mesh.order))
    if scan.orders and not scan.clean:
        n, m = scan.worst
        logging.getLogger(__name__).warning(
            "mesh is clean at order %d but has %d inverted element(s) at sampling "
            "order %d (min scaled Jacobian %.4f)",
            mesh.order, scan.n_inverted[scan.orders.index(n)], n, m)
    names, counts = mesh.edge_tags.unique(return_counts=True)
    for name, n in zip(names, counts.tolist()):
        lines.append("  %-14s : %d edges" % (name, n))
    tags = tag_report(mesh)
    lines.append("  %-14s : %d edges" % ("untagged bdry", tags.n_untagged_boundary))
    lines.append("  %-14s : %d rows" % ("interior tags", tags.n_tagged_interior))
    lines.append(topology.format_report(topology.quad_report(mesh.points, mesh.corners),
                                        "quad"))
    return "\n".join(lines)


__all__ = [
    "TagReport",
    "area",
    "bounds",
    "boundary_edge_ids",
    "boundary_edges",
    "boundary_elements",
    "boundary_points",
    "centroid",
    "edge_rows",
    "edge_tag_rows",
    "element_areas",
    "element_blocks",
    "is_conforming",
    "is_overlap_free",
    "is_watertight",
    "plane_normal",
    "quality_summary",
    "report",
    "scaled_jacobian",
    "tag_report",
    "tagged_edges",
    "topology_report",
]
