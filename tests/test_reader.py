"""``read_re2`` / ``quad_read_re2`` / ``read_rea`` / ``quad_read_rea`` -- the readers'
side of ``to_re2`` / ``quad_to_re2``, one rung apart and with the same contract."""

import numpy as np
import pytest

from nekmeshpy import BoundaryConditions, hexmesh, quadmesh, reader, writer
from nekmeshpy.io.reader import ReaFormatError


def _slab(q):
    """The section swept one unit in z, regions carried and the caps named."""
    return hexmesh.extrude(q, axis=(0, 0, 1), length=1.0, layers=np.linspace(0, 1, 3),
                           element_tags=q.element_tags, first_tag="front",
                           last_tag="back")


def _cht_quad():
    solid = quadmesh.rectangle([(-1, 0, 0), (0, 0, 0), (0, 1, 0), (-1, 1, 0)], 2, 2,
                               side_tags={"right": "interface", "left": "outer"},
                               element_tags="solid")
    fluid = quadmesh.rectangle([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], 3, 2,
                               side_tags={"left": "interface", "right": "outlet"},
                               element_tags="fluid")
    return quadmesh.merge([solid, fluid])


BC = BoundaryConditions(velocity={"outlet": "O  ", "interface": "W  "},
                        temperature={"outer": "I  ", "outlet": "O  "})


def _same_geometry(a, b):
    ka = np.round(np.sort(a.points[a.corners].reshape(-1, 3), axis=0), 9)
    kb = np.round(np.sort(b.points[b.corners].reshape(-1, 3), axis=0), 9)
    assert np.allclose(ka, kb)


def test_quad_re2_roundtrip(tmp_path):
    m = _cht_quad()
    f = str(tmp_path / "a.re2")
    writer.quad_to_re2(m, f, bc=BC, fluid="fluid")
    r = reader.quad_read_re2(f)
    assert r.n_quads == m.n_quads and r.n_points == m.n_points
    assert quadmesh.is_conforming(r) and quadmesh.is_watertight(r)
    assert r.element_tags.unique() == ["fluid", "solid"]
    # fluid first, as written
    assert (r.element_tags.to_dense()[:6] == "fluid").all()
    # the velocity block's name wins; the thermal-only side keeps its own
    assert sorted(r.edge_tags.unique()) == ["I", "O", "W"]
    assert quadmesh.area(r) == pytest.approx(quadmesh.area(m))


def test_hex_re2_roundtrip(tmp_path):
    m = _slab(_cht_quad())
    bc = BoundaryConditions(
        velocity={"outlet": "O  ", "interface": "W  ", "front": "SYM", "back": "SYM"},
        temperature={"outer": "I  ", "outlet": "O  ", "front": "I  ", "back": "I  "})
    f = str(tmp_path / "a.re2")
    writer.to_re2(m, f, bc=bc, fluid="fluid")
    r = reader.read_re2(f)
    assert r.n_hexes == m.n_hexes and r.n_points == m.n_points
    assert hexmesh.is_conforming(r) and hexmesh.is_watertight(r)
    assert r.element_tags.unique() == ["fluid", "solid"]
    assert hexmesh.volume(r) == pytest.approx(hexmesh.volume(m))


def test_dimension_mismatch_is_refused(tmp_path):
    f = str(tmp_path / "a.re2")
    writer.quad_to_re2(_cht_quad(), f, bc=BC, fluid="fluid")
    with pytest.raises(ReaFormatError, match="2"):
        reader.read_re2(f)


def _rea_text(mesh, ndim, names, nelv):
    """A minimal ``.rea`` mesh + BC section for ``mesh``, written the way Nek does (E rows
    name the neighbour) -- an independent producer for the reader to parse."""
    quad = ndim == 2
    nc, ns = (4, 4) if quad else (8, 6)
    elems = mesh.quads if quad else mesh.hexes
    nel = elems.shape[0]
    X = mesh.points[mesh.corners]
    out = ["**MESH DATA**", "%d %d %d NEL,NDIM,NELV" % (nel, ndim, nelv)]
    for e in range(nel):
        out.append("      ELEMENT %d [    1a]    GROUP     0" % (e + 1))
        for lo, hi in ((0, 4),) if quad else ((0, 4), (4, 8)):
            for axis in range(ndim if quad else 3):
                out.append("  ".join("%.9f" % v for v in X[e, lo:hi, axis]))
    out += ["  ***** CURVED SIDE DATA *****", "0 Curved sides follow",
            "  ***** BOUNDARY CONDITIONS *****", "  ***** FLUID   BOUNDARY CONDITIONS *****"]
    owner = {}
    for e in range(nel):
        for s in range(ns):
            owner.setdefault(int(elems[e, s]), []).append((e, s))
    tag = dict(zip(mesh.edge_tags.ids.tolist() if quad else mesh.face_tags.ids.tolist(),
                   (mesh.edge_tags if quad else mesh.face_tags).tags.tolist()))

    def rows(n_elem, velocity):
        for e in range(n_elem):
            for s in range(ns):
                sides = owner[int(elems[e, s])]
                other = [x for x in sides if x != (e, s)]
                # a conjugate wall is a fluid-side velocity condition and an ``E`` row
                # in the thermal block: the interface is named in only one of them
                if other and not (velocity and int(elems[e, s]) in tag):
                    yield " E  %3d%3d%14.5f%14.5f%14.5f%14.5f%14.5f" % (
                        e + 1, s + 1, other[0][0] + 1, other[0][1] + 1, 0, 0, 0)
                elif int(elems[e, s]) in tag:
                    yield " %-3s%3d%3d%14.5f%14.5f%14.5f%14.5f%14.5f" % (
                        names[tag[int(elems[e, s])]], e + 1, s + 1, 0, 0, 0, 0, 0)
                else:
                    yield "    %3d%3d%14.5f%14.5f%14.5f%14.5f%14.5f" % (
                        e + 1, s + 1, 0, 0, 0, 0, 0)
    out += list(rows(nelv, True))
    out += ["  ***** THERMAL BOUNDARY CONDITIONS *****"] + list(rows(nel, False))
    return "\n".join(out) + "\n"


def test_quad_rea_reads_a_conjugate_mesh(tmp_path):
    m = _cht_quad()
    f = tmp_path / "a.rea"
    names = {"interface": "W", "outer": "I", "outlet": "O"}
    nelv = int((m.element_tags.to_dense() == "fluid").sum())
    f.write_text(_rea_text(m, 2, names, nelv))
    r = reader.quad_read_rea(str(f))
    assert r.n_quads == m.n_quads and r.n_points == m.n_points
    assert quadmesh.is_conforming(r) and quadmesh.is_watertight(r)
    _same_geometry(r, m)
    assert r.element_tags.to_dense().tolist().count("fluid") == nelv
    assert sorted(r.edge_tags.unique()) == ["I", "O", "W"]


def test_hex_rea_reads_a_conjugate_mesh(tmp_path):
    m = _slab(_cht_quad())
    f = tmp_path / "a.rea"
    names = {"interface": "W", "outer": "I", "outlet": "O", "front": "I", "back": "I"}
    nelv = int((m.element_tags.to_dense() == "fluid").sum())
    f.write_text(_rea_text(m, 3, names, nelv))
    r = reader.read_rea(str(f))
    assert r.n_hexes == m.n_hexes and r.n_points == m.n_points
    assert hexmesh.is_conforming(r) and hexmesh.is_watertight(r)
    assert sorted(r.face_tags.unique()) == ["I", "O", "W"]
    assert r.element_tags.to_dense().tolist().count("fluid") == nelv


def test_rea_dimension_mismatch_is_refused(tmp_path):
    f = tmp_path / "a.rea"
    f.write_text(_rea_text(_cht_quad(), 2, {"interface": "W", "outer": "I",
                                           "outlet": "O"}, 6))
    with pytest.raises(ReaFormatError, match="3"):
        reader.read_rea(str(f))
