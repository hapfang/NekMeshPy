"""The quad rung's topology / tag checks -- ``hexmesh``'s, one rung down."""

import numpy as np

from nekmeshpy import hexmesh, quadmesh


def _rect(x0, x1, nx, ny=2, **kw):
    return quadmesh.rectangle([(x0, 0, 0), (x1, 0, 0), (x1, 1, 0), (x0, 1, 0)],
                              nx, ny, **kw)


def test_welded_grid_is_clean():
    m = quadmesh.merge([_rect(0, 1, 3), _rect(1, 2, 3)])
    rep = quadmesh.topology_report(m)
    assert rep.n_elements == 12 and rep.n_components == 1
    assert rep.n_hanging_points == 0 and rep.n_open_edges == 0
    assert quadmesh.is_watertight(m) and quadmesh.is_conforming(m)
    assert quadmesh.is_overlap_free(m)


def test_unequal_seam_is_hanging():
    # 3 cells meet 2 on the shared side: the finer side's points hang on the other
    m = quadmesh.merge([_rect(0, 1, 3, ny=3), _rect(1, 2, 3, ny=2)])
    assert quadmesh.topology_report(m).n_hanging_points > 0
    assert not quadmesh.is_conforming(m)


def test_graded_rectangle_conforms():
    # the top and left sides run against the bottom and right; a graded fraction
    # must land on the same positions on both
    f = np.array([0.0, 0.1, 0.4, 1.0])
    m = _rect(0, 1, 3, x_frac=f)
    c = m.points[m.corners]
    assert np.allclose(c[:, 0, 0], c[:, 3, 0]) and np.allclose(c[:, 1, 0], c[:, 2, 0])
    left = quadmesh.merge([_rect(0, 1, 3, x_frac=f), _rect(1, 2, 3, x_frac=f)])
    assert quadmesh.is_conforming(left)


def test_disconnected_is_not_watertight():
    a, b = _rect(0, 1, 2), _rect(5, 6, 2)
    m = quadmesh.merge([a, b])
    assert quadmesh.topology_report(m).n_components == 2
    assert not quadmesh.is_watertight(m)


def test_overlap_is_found():
    a = _rect(0, 1, 2)
    shifted = quadmesh.translate(_rect(0, 1, 2), [0.5, 0.0, 0.0])
    # two stacked, offset sections welded by nothing: build the container directly
    m = quadmesh.merge([a, shifted], tol=1e-12)
    assert not quadmesh.is_overlap_free(m)


def test_overlap_in_a_tilted_plane():
    a = quadmesh.rotate(_rect(0, 1, 2), 0.7, [1, 0, 0])
    assert quadmesh.is_overlap_free(a)


def test_tag_report_matches_hex_semantics():
    m = _rect(0, 1, 2, side_tags={"left": "inlet", "right": "outlet"})
    rep = quadmesh.tag_report(m)
    assert rep.n_rows == 4                       # 2 inlet + 2 outlet edges
    assert rep.n_untagged_boundary == 4          # top and bottom
    assert rep.n_tagged_interior == 0
    mask = quadmesh.boundary_edge_ids(m)
    assert mask.sum() == len(quadmesh.boundary_edges(m))


def test_edge_tag_rows_roundtrip():
    m = _rect(0, 1, 2, side_tags={"left": "inlet"})
    rows, tags = quadmesh.edge_tag_rows(m)
    assert set(tags.tolist()) == {"inlet"} and rows.shape == (2, 2)
    assert (m.points[m.corners[rows[:, 0]][:, [0, 3]]][..., 0] == 0).all()
    # each named row's local edge really is a left-side edge
    assert set(rows[:, 1].tolist()) <= {4}


def test_report_runs_and_matches_hex_shape():
    m = _rect(0, 1, 2, side_tags={"left": "inlet"})
    text = quadmesh.report(m)
    assert "conformal      : True" in text and "inlet" in text
    h = hexmesh.extrude(m, axis=(0, 0, 1), length=1.0, layers=np.linspace(0, 1, 3))
    assert "conformal      : True" in hexmesh.report(h)
