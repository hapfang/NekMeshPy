"""``quad_to_re2`` -- the 2-D ``.re2`` of a section mesh, with ``to_re2``'s contract."""

import struct

import numpy as np
import pytest

from nekmeshpy import BoundaryConditions, quadmesh, writer


def _rect(x0, x1, tag, sides):
    return quadmesh.rectangle([(x0, 0, 0), (x1, 0, 0), (x1, 1, 0), (x0, 1, 0)], 2, 2,
                              side_tags=sides, element_tags=tag)


def _cht():
    """Solid block stored FIRST, fluid second, sharing the named ``interface``."""
    solid = _rect(-1, 0, "solid", {"right": "interface", "left": "outer"})
    fluid = _rect(0, 1, "fluid", {"left": "interface", "right": "outlet"})
    return quadmesh.merge([solid, fluid])


def _read(path):
    b = open(path, "rb").read()
    nel, ndim, nelv, nbc = (int(b[5:21]), int(b[21:24]), int(b[24:40]), int(b[40:44]))
    o = 84 + nel * (1 + 4 * ndim) * 8 + 8
    elems = np.frombuffer(b[84:84 + nel * (1 + 4 * ndim) * 8], "<f8").reshape(nel, -1)
    blocks = []
    for _ in range(nbc):
        n = int(struct.unpack("<d", b[o:o + 8])[0])
        o += 8
        blocks.append(np.frombuffer(b[o:o + 64 * n], "<f8").reshape(n, 8))
        o += 64 * n
    assert o == len(b)
    code = lambda r: struct.pack("<d", r[7]).rstrip(b"\0").decode()   # noqa: E731
    return (nel, ndim, nelv), elems, [[(int(r[0]), int(r[1]), code(r), int(r[6]))
                                       for r in blk] for blk in blocks]


BC = BoundaryConditions(
    velocity={"outlet": "O  ", "interface": "W  "},
    temperature={"outer": "I  ", "outlet": "O  "})


def test_header_geometry_and_fluid_first(tmp_path):
    m = _cht()
    f = str(tmp_path / "a.re2")
    writer.quad_to_re2(m, f, bc=BC, fluid="fluid")
    (nel, ndim, nelv), elems, _ = _read(f)
    assert (nel, ndim, nelv) == (8, 2, 4)
    # fluid (x in [0,1]) is written first: element 1..4, then the solid
    assert (elems[:4, 1:5] >= 0).all() and (elems[4:, 1:5] <= 0).all()
    assert (elems[:, 0] == 0).all()


def test_blocks_are_per_field_and_scoped(tmp_path):
    m = _cht()
    f = str(tmp_path / "a.re2")
    writer.quad_to_re2(m, f, bc=BC, fluid="fluid")
    _, _, (vel, tmp) = _read(f)
    assert all(e <= 4 for e, *_ in vel)                       # fluid elements only
    assert sorted(c for *_, c, _ in [(r[0], r[1], r[2], r[3]) for r in vel]) == \
        ["O  ", "O  ", "W  ", "W  "]
    # temperature spans every element; the interface (absent) writes no row
    assert sorted(c for _, _, c, _ in tmp) == ["I  ", "I  ", "O  ", "O  "]
    assert any(e > 4 for e, *_ in tmp)


def test_face_is_the_quads_local_edge(tmp_path):
    m = _rect(0, 1, "fluid", {"bottom": "a", "right": "b", "top": "c", "left": "d"})
    f = str(tmp_path / "a.re2")
    writer.quad_to_re2(m, f, bc={"a": "W  ", "b": "O  ", "c": "W  ", "d": "v  "})
    _, _, (blk,) = _read(f)
    by = {c: sorted({fc for _, fc, cc, _ in blk if cc == c}) for c in ("O  ", "v  ")}
    assert by == {"O  ": [2], "v  ": [4]}


def test_missing_temperature_for_solid_raises(tmp_path):
    with pytest.raises(ValueError, match="temperature"):
        writer.quad_to_re2(_cht(), str(tmp_path / "a.re2"),
                           bc={"outlet": "O  "}, fluid="fluid")


def test_periodic_code_is_refused(tmp_path):
    with pytest.raises(ValueError, match="periodic"):
        writer.quad_to_re2(_cht(), str(tmp_path / "a.re2"), bc={"outlet": "P  "})


def test_non_planar_mesh_is_refused(tmp_path):
    m = quadmesh.translate(_cht(), [0, 0, 0.5])
    f = str(tmp_path / "a.re2")
    writer.quad_to_re2(m, f, bc={"outlet": "O  "})            # z = const is fine
    tilted = quadmesh.rotate(_cht(), 0.3, [1, 0, 0])
    with pytest.raises(ValueError, match="z = const"):
        writer.quad_to_re2(tilted, f, bc={"outlet": "O  "})


def test_bc_is_required(tmp_path):
    with pytest.raises(ValueError, match="bc="):
        writer.quad_to_re2(_cht(), str(tmp_path / "a.re2"), bc=None)
