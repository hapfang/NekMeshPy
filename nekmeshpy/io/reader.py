"""Readers for Nek5000's mesh formats: the ASCII ``.rea`` and the binary ``.re2``, each
at the hex rung (3-D) and the quad rung (2-D) -- ``read_rea`` / ``quad_read_rea``,
``read_re2`` / ``quad_read_re2``, the readers' side of ``writer.to_re2`` /
``writer.quad_to_re2``.

Only the geometry, the element regions and the boundary conditions are read -- the
surrounding parameters, thermal, restart, and drive-force sections either don't bear on
a mesh's shape or (curved sides) aren't supported yet and are refused rather than
silently dropped.  A mesh with ``nelv < nel`` is a conjugate one: its first ``nelv``
elements are the ``"fluid"`` region and the rest ``"solid"``.  A boundary row's raw Nek
code is its tag name, and where a file carries a second (thermal) block a side the
velocity block does not name takes the thermal block's code; a periodic row is read as
the name ``"P"`` with its partner dropped.  The row layout below (element/face packed into one field once
``NELGT`` reaches four digits, corners grouped ``X,Y,Z`` per half rather than per axis)
follows Nek5000's own ``core/reader_rea.f`` (``rdmesh``/``rdbdry``) exactly -- verified
against its upstream source, not reverse-engineered from one file.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Sequence

import numpy as np

from .._typing import IntArray, PointArray
from ..core import conform
from ..hexmesh import HexMesh
from ..hexmesh.tag import tag_faces
from ..quadmesh import QuadMesh
from ..quadmesh.tag import tag_edges
from ..tags import Tags

_log = logging.getLogger("nekmeshpy")

_FACE_POINTS = HexMesh.FACE_POINTS  # (6,4) local corner indices per Nek face
_EDGE_POINTS = QuadMesh.EDGE_POINTS  # (4,2) local corner indices per Nek 2-D face


class ReaFormatError(ValueError):
    """The ``.rea`` file doesn't match the section this reader expected next."""


class _UnionFind:
    """Plain array union-find over the ``nel*8`` raw (unwelded) corner slots."""

    def __init__(self, n: int) -> None:
        self.parent: IntArray = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = int(p[root])
        while p[x] != root:
            p[x], x = root, int(p[x])
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _find(lines: list[str], i: int, needle: str) -> int:
    """The index of the next line at or after ``i`` containing ``needle`` (case-
    insensitive), or a :class:`ReaFormatError` if the file runs out first."""
    upper = needle.upper()
    for j in range(i, len(lines)):
        if upper in lines[j].upper():
            return j
    raise ReaFormatError("rea: could not find a %r section" % needle)


def _floats(lines: list[str], i: int, n: int) -> tuple[list[float], int]:
    """``(values, next line index)``: ``n`` whitespace-separated floats, read from
    ``lines[i]`` onward -- Nek's own list-directed reads don't care about line breaks,
    so a short row is topped up from the next line rather than assumed short."""
    vals: list[float] = []
    while len(vals) < n:
        vals.extend(float(tok) for tok in lines[i].split())
        i += 1
    if len(vals) != n:
        raise ReaFormatError(
            "rea: expected %d values, found %d run together across lines" % (n, len(vals)))
    return vals, i


def _match_face(p1: PointArray, p2: PointArray) -> IntArray:
    """``(m,)``: for each of ``p1``'s ``m`` corners, the index into ``p2`` of the same
    physical corner -- found by nearest distance within just these two quartets, so it
    needs no global coincidence tolerance at all (a real hex face's 4 corners are never
    close enough to each other to confuse a nearest-corner match against its own
    partner face). Falls back to the optimal assignment only if that greedy match isn't
    already a bijection, which real geometry never triggers."""
    d = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
    match: IntArray = np.argmin(d, axis=1)
    if len(set(match.tolist())) == p1.shape[0]:
        return match
    from scipy.optimize import linear_sum_assignment
    _, cols = linear_sum_assignment(d)
    return cols.astype(np.int64)


# -- shared assembly ---------------------------------------------------------------
#: ``(element, 1-based side, code)`` rows a boundary block names, 0-based element.
_Named = list[tuple[int, int, str]]


def _regions(nel: int, nelv: int) -> Tags | None:
    """``"fluid"`` for the first ``nelv`` elements and ``"solid"`` for the rest, or
    ``None`` (untagged) when every element is velocity-mesh."""
    if nelv >= nel:
        return None
    return Tags.empty(nel).put(np.arange(nelv), "fluid").put(
        np.arange(nelv, nel), "solid")


def _weld_by_rows(corners: PointArray, side_points: IntArray,
                  internal: list[tuple[int, int, int, int]]
                  ) -> tuple[PointArray, IntArray]:
    """``(points, conn)`` from per-element ``corners`` ``(nel, k, 3)``, welded through
    the file's own ``"E"`` rows ``(e1, side1, e2, side2)`` (0-based).  Each connected
    side pair's corners are matched by nearest distance *within that pair alone*, which
    needs no tolerance because the search space is a handful of points, not the mesh."""
    nel, k, _ = corners.shape
    uf = _UnionFind(nel * k)
    for e1, f1, e2, f2 in internal:
        idx1, idx2 = side_points[f1], side_points[f2]
        match = _match_face(corners[e1, idx1, :], corners[e2, idx2, :])
        for a, b in zip(idx1, idx2[match]):
            uf.union(e1 * k + a, e2 * k + b)
    roots = np.array([uf.find(j) for j in range(nel * k)], dtype=np.int64)
    _, welded_id = np.unique(roots, return_inverse=True)
    n_points = int(welded_id.max()) + 1 if welded_id.size else 0
    sums = np.zeros((n_points, 3))
    counts = np.zeros(n_points)
    np.add.at(sums, welded_id, corners.reshape(nel * k, 3))
    np.add.at(counts, welded_id, 1.0)
    return sums / counts[:, None], welded_id.reshape(nel, k)


def _weld_by_coordinate(corners: PointArray) -> tuple[PointArray, IntArray]:
    """``(points, conn)`` welding corners that coincide: a binary ``.re2`` carries no
    element-to-element rows, but its corners are exact doubles written from shared
    points, so the radius is a rounding error of the model, not a tuning knob."""
    nel, k, _ = corners.shape
    flat = corners.reshape(nel * k, 3)
    tol = 1e-7 * conform.bbox_scale(flat)
    rep = conform.coincident_clusters(flat, tol) if tol > 0 else np.arange(nel * k)
    _, first, welded = np.unique(rep, return_index=True, return_inverse=True)
    return flat[first], welded.reshape(nel, k)


def _tag_rows(named: Sequence[_Named]) -> tuple[IntArray, list[str]]:
    """Fold per-block rows into one ``(K,2)`` ``[element, side]`` table and names, the
    **first** block winning: a later row on the same side is applied first so the
    ``put`` that follows (last wins) leaves the earlier block's name."""
    rows: list[tuple[int, int]] = []
    names: list[str] = []
    for block in reversed(list(named)):
        for e, f, code in block:
            rows.append((e, f))
            names.append(code)
    return np.array(rows, dtype=np.int64).reshape(-1, 2), names


def _hex_mesh(corners: PointArray, side_points: IntArray, nelv: int,
              internal: list[tuple[int, int, int, int]] | None,
              named: Sequence[_Named]) -> HexMesh:
    nel = corners.shape[0]
    points, conn = (_weld_by_rows(corners, side_points, internal)
                    if internal is not None else _weld_by_coordinate(corners))
    mesh = HexMesh.from_corners(points, conn, _regions(nel, nelv))
    rows, names = _tag_rows(named)
    if names:
        mesh = tag_faces(mesh, mesh.hexes[rows[:, 0], rows[:, 1] - 1], names)
    return mesh


def _quad_mesh(corners: PointArray, side_points: IntArray, nelv: int,
               internal: list[tuple[int, int, int, int]] | None,
               named: Sequence[_Named]) -> QuadMesh:
    nel = corners.shape[0]
    points, conn = (_weld_by_rows(corners, side_points, internal)
                    if internal is not None else _weld_by_coordinate(corners))
    mesh = QuadMesh.from_corners(points, conn, _regions(nel, nelv))
    rows, names = _tag_rows(named)
    if names:
        mesh = tag_edges(mesh, rows, names)
    return mesh


# -- .rea --------------------------------------------------------------------------
def _read_rea_raw(path: str, ndim: int, kind: str) -> tuple[
        PointArray, int, list[tuple[int, int, int, int]], list[_Named]]:
    """``(corners (nel, 2**ndim, 3), nelv, E rows, [fluid rows, thermal rows])`` of an
    ``ndim``-D ``.rea``.  Raises :class:`ReaFormatError` for the wrong dimension or any
    curved side."""
    with open(path, "r") as f:
        raw = f.readlines()
    nc, ns = 2 ** ndim, 2 * ndim

    i = _find(raw, 0, "**MESH DATA**") + 1
    nel, file_ndim, nelv = (int(float(tok)) for tok in raw[i].split()[:3])
    i += 1
    if file_ndim != ndim:
        raise ReaFormatError(
            "rea: %s reads a %d-D run, got NDIM=%d" % (kind, ndim, file_ndim))

    corners: PointArray = np.zeros((nel, nc, 3), dtype=float)
    half = nc // 2 if ndim == 3 else nc
    for e in range(nel):
        i += 1                                     # "ELEMENT n [tag] GROUP g" header
        if ndim == 2:
            corners[e, :, 0], i = _floats(raw, i, 4)
            corners[e, :, 1], i = _floats(raw, i, 4)
        else:
            for lo in (0, half):
                for axis in range(3):
                    corners[e, lo:lo + half, axis], i = _floats(raw, i, half)

    i = _find(raw, i, "CURVED SIDE DATA") + 1
    n_curved = int(float(raw[i].split()[0]))
    if n_curved:
        raise ReaFormatError(
            "rea: %d curved side(s) present; the reader reconstructs straight-sided "
            "elements only, and dropping curvature would silently hand back the wrong "
            "geometry rather than a linear approximation of it" % n_curved)
    i += 1

    internal: list[tuple[int, int, int, int]] = []       # (e1, f1, e2, f2), 0-based
    blocks: list[_Named] = []

    def block(i: int, n_elem: int) -> int:
        """Read ``n_elem * ns`` rows of one field, return the next line index.
        Fixed columns (``1x,a3`` then ``i3,i3`` below 1000 elements, ``i5,i1`` above);
        the values start at column 10 either way."""
        named: _Named = []
        for _ in range(ns * n_elem):
            ln = raw[i]
            i += 1
            code = ln[1:4].strip()
            if nel < 1000:
                iel, iface = int(ln[4:7]), int(ln[7:10])
            else:
                iel, iface = int(ln[4:9]), int(ln[9:10])
            vals = ln[10:].split()
            if code == "E":
                internal.append((iel - 1, iface - 1,
                                 int(float(vals[0])) - 1, int(float(vals[1])) - 1))
            elif code:
                named.append((iel - 1, iface, code))
        blocks.append(named)
        return i

    i = _find(raw, i, "BOUNDARY CONDITIONS") + 1
    if "NO " not in raw[i].upper():
        i = block(i + 1, nelv)                     # past "***** FLUID BOUNDARY ..."
    else:
        i += 1
        blocks.append([])
    # the thermal block covers every element, solid ones included
    if i < len(raw) and "THERMAL BOUNDARY CONDITIONS" in raw[i].upper() \
            and "NO " not in raw[i].upper():
        block(i + 1, nel)
    return corners, nelv, internal, blocks


def read_rea(path: str) -> HexMesh:
    """Read a 3-D Nek5000 ``.rea`` mesh into a linear (order-1) :class:`HexMesh
    <nekmeshpy.hexmesh.HexMesh>`.

    The ASCII format stores each element's 8 corners independently -- there is no
    shared point numbering to read directly, and floating-point round-trip through
    7-significant-digit text makes a coincidence-tolerance weld genuinely fragile (on
    the reference file this reader was built against, no single tolerance across four
    orders of magnitude was both watertight *and* conformal). Instead this reader uses
    the file's own ``"E"`` (element-to-element) boundary rows -- which name the exact
    connected element and face, not a inferred one -- and matches each connected face
    pair's 4 corners by nearest distance *within that pair alone*, which needs no
    tolerance because the search space is 4 points against 4, not the whole mesh.

    Named rows (``"W"``, ``"v"``, ``"O"``, ...) are carried over as face tags via
    :func:`hexmesh.tag_faces <nekmeshpy.hexmesh.tag.tag_faces>`, keyed by the raw Nek
    code -- this reader does not rename them. ``nelv < nel`` makes the first ``nelv``
    hexes ``"fluid"`` and the rest ``"solid"``.

    Raises :class:`ReaFormatError` for anything outside that scope: a 2-D mesh (see
    :func:`quad_read_rea`), or a nonzero curved-side count (this reader reconstructs
    straight-sided hexes only, and silently dropping curvature would hand back the
    wrong geometry rather than an approximate one)."""
    corners, nelv, internal, blocks = _read_rea_raw(path, 3, "read_rea")
    mesh = _hex_mesh(corners, _FACE_POINTS, nelv, internal, blocks)
    _log.info("rea: read %d hexes, %d welded points (%d raw) from %s",
              corners.shape[0], mesh.n_points, corners.shape[0] * 8, path)
    return mesh


def quad_read_rea(path: str) -> QuadMesh:
    """Read a 2-D Nek5000 ``.rea`` mesh into a linear (order-1) :class:`QuadMesh
    <nekmeshpy.quadmesh.QuadMesh>` in the ``z = 0`` plane -- :func:`read_rea` one rung
    down, welded through the file's own ``"E"`` rows the same way, with named rows
    carried over as edge tags (:func:`quadmesh.tag_edges
    <nekmeshpy.quadmesh.tag.tag_edges>`).  Raises :class:`ReaFormatError` for a 3-D
    run or a nonzero curved-side count."""
    corners, nelv, internal, blocks = _read_rea_raw(path, 2, "quad_read_rea")
    mesh = _quad_mesh(corners, _EDGE_POINTS, nelv, internal, blocks)
    _log.info("rea: read %d quads, %d welded points (%d raw) from %s",
              corners.shape[0], mesh.n_points, corners.shape[0] * 4, path)
    return mesh


# -- .re2 --------------------------------------------------------------------------
def _read_re2_raw(path: str, ndim: int, kind: str) -> tuple[
        PointArray, int, list[_Named]]:
    """``(corners (nel, 2**ndim, 3), nelv, [one row list per boundary block])``."""
    with open(path, "rb") as f:
        data = f.read()
    tok = data[:80].decode("ascii", "replace").split()
    if not tok or not tok[0].startswith("#v"):
        raise ReaFormatError("re2: %s: no '#v...' header in %s" % (kind, path))
    nel, file_ndim, nelv = int(tok[1]), int(tok[2]), int(tok[3])
    if file_ndim != ndim:
        raise ReaFormatError(
            "re2: %s reads a %d-D mesh, got ndim=%d" % (kind, ndim, file_ndim))
    try:
        nbc = int(tok[4])
    except (IndexError, ValueError):
        nbc = 2 if nel > nelv else 1     # Nek's own rule when the header states none
    if abs(float(np.frombuffer(data[80:84], "<f4")[0]) - 6.54321) > 1e-5:
        raise ReaFormatError("re2: %s: not a little-endian file (endian tag)" % kind)
    nc = 2 ** ndim
    o = 84
    rec = 1 + ndim * nc
    body: PointArray = np.frombuffer(data, "<f8", count=nel * rec, offset=o).reshape(
        nel, rec)
    o += nel * rec * 8
    corners: PointArray = np.zeros((nel, nc, 3), dtype=float)
    for axis in range(ndim):
        corners[:, :, axis] = body[:, 1 + axis * nc:1 + (axis + 1) * nc]
    n_curved = int(np.frombuffer(data, "<f8", count=1, offset=o)[0])
    o += 8
    if n_curved:
        raise ReaFormatError(
            "re2: %d curved side(s) present; the reader reconstructs straight-sided "
            "elements only" % n_curved)
    blocks: list[_Named] = []
    for _ in range(nbc):
        n = int(np.frombuffer(data, "<f8", count=1, offset=o)[0])
        o += 8
        rows: PointArray = np.frombuffer(data, "<f8", count=8 * n, offset=o).reshape(n, 8)
        o += 64 * n
        named: _Named = []
        for r in rows:
            code = struct.pack("<d", r[7]).rstrip(b"\0").decode("ascii", "replace").strip()
            if code and code != "E":
                named.append((int(r[0]) - 1, int(r[1]), code))
        blocks.append(named)
    if o != len(data):
        raise ReaFormatError(
            "re2: %s: %d trailing byte(s) after the boundary blocks" % (kind,
                                                                         len(data) - o))
    if any(code == "P" for blk in blocks for *_, code in blk):
        _log.warning("re2: periodic rows are read as the name 'P' with no partner")
    return corners, nelv, blocks


def read_re2(path: str) -> HexMesh:
    """Read a binary 3-D Nek ``.re2`` into a linear :class:`HexMesh
    <nekmeshpy.hexmesh.HexMesh>` -- the reader's side of :func:`writer.to_re2
    <nekmeshpy.io.writer.to_re2>`.

    A ``.re2`` stores each element's corners independently and, unlike a ``.rea``,
    carries no element-to-element rows, so coincident corners are welded by coordinate
    at ``1e-7`` of the model size (exact doubles, so this is a rounding radius, not a
    tolerance to tune).  Names, regions and the second block follow the module rules."""
    corners, nelv, blocks = _read_re2_raw(path, 3, "read_re2")
    return _hex_mesh(corners, _FACE_POINTS, nelv, None, blocks)


def quad_read_re2(path: str) -> QuadMesh:
    """Read a binary 2-D Nek ``.re2`` into a linear :class:`QuadMesh
    <nekmeshpy.quadmesh.QuadMesh>` -- :func:`read_re2` one rung down, and the reader's
    side of :func:`writer.quad_to_re2 <nekmeshpy.io.writer.quad_to_re2>`."""
    corners, nelv, blocks = _read_re2_raw(path, 2, "quad_read_re2")
    return _quad_mesh(corners, _EDGE_POINTS, nelv, None, blocks)


__all__ = ["ReaFormatError", "quad_read_re2", "quad_read_rea", "read_re2", "read_rea"]
