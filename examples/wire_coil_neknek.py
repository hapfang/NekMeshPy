"""A helically coiled wire in a round pipe, meshed for overset (neknek) conjugate heat
transfer as two meshes: a screw-swept ring around the coil and a plain core cylinder.

The first mesh is one r-z slice, one pitch tall and ``R_CORE .. R_OUT`` wide, screw-swept
``TURNS`` times about the axis (slice coordinates are ``(x=z, y=r)``, so the sweep
direction ``+theta`` is the right-handed normal).  The slice is assembled bottom-up from
the pieces it is made of: the wire disc, a fluid ring around it that is squircled on the
side away from the wall, a single layer on the wall-facing side, two channels running
along the wall, Coons blocks around the box, and the solid pipe wall.  Whole pitches of
the helix make the two ends, and the two helicoid sides, periodic pairs under a pure axial
translation; the sweep's own seam between consecutive turns is welded and cleared.

The second mesh, ``core``, is a separate cylinder of radius ``R_CORE + R_OVERLAP`` and the
same length, offset by ``Z_OFFSET`` and rotated by ``THETA_OFFSET`` so it overlaps the
sweep's inner face rather than meeting it node for node.  Both faces are ``int``.

    PYTHONPATH=. python examples/wire_coil_neknek.py

Produces ``wire_coil_neknek.re2`` / ``.vtu`` and ``wire_coil_neknek_core.re2`` / ``.vtu``.
"""
import numpy as np

from nekmeshpy import BoundaryConditions, hexmesh, linemesh, quadmesh, writer
from nekmeshpy.core import affine
from nekmeshpy.tags import Tags

R_HELIX, RW = 0.375, 0.125 - 0.004
PITCH = (1.0 / 6.0) / 0.375
R_TUBE, WALL = 0.5, 0.10
R_CORE = R_HELIX - RW - 0.05
R_OUT = R_TUBE + WALL
RISE = PITCH / (2 * np.pi)

ORDER = 2
N_SIDE = 4
N_THETA = 40
TURNS = 2
SQ_N = 4.0
FAR_PUSH = 1.05
Z_PUSH = 1.05

H = R_TUBE - R_HELIX
CZ = PITCH / 2
Z_BOT, Z_TOP = CZ - H * Z_PUSH, CZ + H * Z_PUSH
N_ROWS = 3

R_OVERLAP = 0.01
Z_OFFSET = -0.02
THETA_OFFSET = np.deg2rad(3.0)
N_R = 10

cut = quadmesh.shape._slice_chain


def squircle(a, b):
    if b >= 0:
        return a, b
    f = (abs(a / H)**SQ_N + abs(b / H)**SQ_N)**(-1.0 / SQ_N)
    f *= 1.0 + (FAR_PUSH - 1.0) * min(-b / H, 1.0)
    return a * f, b * f


corners = [(+H, +H), (-H, +H), (-H, -H), (+H, -H), (+H, +H)]
loop = []
for (a0, b0), (a1, b1) in zip(corners[:-1], corners[1:]):
    for k in range(N_SIDE):
        f = k / N_SIDE
        a, b = squircle(a0 + f * (a1 - a0), b0 + f * (b1 - b0))
        loop.append((CZ + a * Z_PUSH, R_HELIX + b, 0.0))
loop = np.array(loop)
ring_pts = np.vstack([loop, loop[:1]])
R_A = loop[3 * N_SIDE + N_SIDE - 1][1]

circle = linemesh.circle(RW, 4 * N_SIDE, center=(CZ, R_HELIX, 0), start_theta=np.pi / 4,
                         order=ORDER)
outer = linemesh.loft(loop, loop=True, order=ORDER)
middle = linemesh.blend(outer, circle, [0.0, 0.5, 1.0])[1]

disc = quadmesh.ogrid(circle, N_SIDE, 1, wall_tag="coil",
                      center_scale=0.8, quadrant_scale=0.8, element_tags="solid")

ring = quadmesh.retag_element(quadmesh.merge([
    quadmesh.loft([cut(outer, N_SIDE + 1, 4 * N_SIDE - 1), cut(middle, N_SIDE + 1, 4 * N_SIDE - 1),
                   cut(circle, N_SIDE + 1, 4 * N_SIDE - 1)]),
    quadmesh.loft([cut(middle, 4 * N_SIDE - 1, 4 * N_SIDE), cut(circle, 4 * N_SIDE - 1, 4 * N_SIDE)]),
    quadmesh.loft([cut(middle, N_SIDE, N_SIDE + 1), cut(circle, N_SIDE, N_SIDE + 1)]),
], tol=1e-9), "fluid")

spoke_hi = 0.5 * (loop[0] + circle.points[0])
spoke_lo = 0.5 * (loop[N_SIDE] + circle.points[N_SIDE])
wall_layer = quadmesh.loft([
    linemesh.loft(np.array([spoke_hi, loop[1], loop[2], loop[3], spoke_lo]), order=ORDER),
    cut(circle, 0, N_SIDE)], element_tags="fluid")


def hline(z, tag=None):
    pts = np.array([(z, r, 0.0) for r in (R_A, spoke_hi[1], R_TUBE)])
    return linemesh.loft(pts, order=ORDER, element_tags=tag)


def channel(chain, z0, z1):
    zs = [z0 + (z1 - z0) * k / N_ROWS for k in range(N_ROWS + 1)]
    cap = "zhi" if z1 > z0 else "zlo"
    stack = [chain] + [hline(z) for z in zs[:-1]] + [hline(zs[-1], cap)]
    if z1 > z0:
        return quadmesh.loft(stack[::-1], first_tag=cap, element_tags="fluid")
    return quadmesh.loft(stack, element_tags="fluid")


bc_top = linemesh.loft(np.array([middle.points[4 * N_SIDE - 1], middle.points[0], loop[1]]),
                       interior=np.stack([middle.interior[4 * N_SIDE - 1],
                                          linemesh.loft(np.array([spoke_hi, loop[1]]),
                                                        order=ORDER).interior[0]]),
                       order=ORDER)
bc_bot = linemesh.loft(np.array([middle.points[N_SIDE + 1], middle.points[N_SIDE],
                                 loop[N_SIDE - 1]]),
                       interior=np.stack([middle.interior[N_SIDE][::-1],
                                          linemesh.loft(np.array([loop[N_SIDE - 1], spoke_lo]),
                                                        order=ORDER).interior[0][::-1]]),
                       order=ORDER)
channels = [channel(bc_top, Z_TOP, PITCH), channel(bc_bot, Z_BOT, 0.0)]

rs = [R_CORE, R_HELIX - H, R_A, R_TUBE, R_OUT]
zs = [0.0, Z_BOT, Z_TOP, PITCH]
V = {(i, j): np.array([zs[j], rs[i], 0.0]) for i in range(5) for j in range(4)}
V[(1, 1)], V[(1, 2)] = ring_pts[2 * N_SIDE], ring_pts[3 * N_SIDE]
sides = {
    ((3, 2), (3, 1)): ring_pts[0:N_SIDE + 1],
    ((1, 1), (1, 2)): ring_pts[2 * N_SIDE:3 * N_SIDE + 1],
    ((1, 2), (2, 2)): ring_pts[3 * N_SIDE:4 * N_SIDE],
    ((2, 1), (1, 1)): ring_pts[N_SIDE + 1:2 * N_SIDE + 1],
}
sides |= {(b, a): p[::-1] for (a, b), p in list(sides.items())}


def edge(va, vb, n, tag=""):
    pts = sides.get((va, vb))
    if pts is None:
        pts = V[va] + np.linspace(0.0, 1.0, n + 1)[:, None] * (V[vb] - V[va])
    return linemesh.loft(pts, order=ORDER, element_tags=tag or None)


def block(i, j, nr, nz, name, bottom="", top=""):
    c0, c1, c2, c3 = (i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)
    sfx = "_solid" if name == "solid" else ""
    return quadmesh.structured([
        edge(c0, c1, nz, bottom), edge(c1, c2, nr, "zhi" + sfx if j == 2 else ""),
        edge(c2, c3, nz, top), edge(c3, c0, nr, "zlo" + sfx if j == 0 else "")],
        element_tags=name)


rows = [N_ROWS, N_SIDE, N_ROWS]
blocks = [block(0, j, 2, rows[j], "fluid", bottom="core") for j in range(3)]
blocks += [block(1, j, N_SIDE - 1, N_ROWS, "fluid") for j in (0, 2)]
blocks += [block(3, j, 1, rows[j], "solid", bottom="wall", top="outer") for j in range(3)]

section = quadmesh.merge(blocks + channels + [disc, ring, wall_layer], tol=1e-9)


def screw(t):
    th = 2 * np.pi * TURNS * t

    def fn(P):
        P = np.asarray(P, float).reshape(-1, 3)
        z, r = P[:, 0], P[:, 1]
        return np.column_stack([r * np.cos(th), r * np.sin(th), z + RISE * th])
    return fn


is_solid = np.asarray(section.element_tags.to_dense()) == "solid"
mesh = hexmesh.loft_fn(lambda t: quadmesh.transform_fn(section, screw(t)),
                       np.linspace(0, 1, N_THETA * TURNS + 1),
                       element_tags=section.element_tags,
                       first_tag=Tags.from_dense(np.where(is_solid, "cut_lo", "inlet")),
                       last_tag=Tags.from_dense(np.where(is_solid, "cut_hi", "outlet")))
mesh = hexmesh.merge([mesh], tol=1e-9,
                     clear_seam_tags=["zlo", "zhi", "zlo_solid", "zhi_solid"])

core = hexmesh.extrude(
    quadmesh.ogrid(linemesh.circle(R_CORE + R_OVERLAP, N_THETA, order=ORDER), N_THETA // 4, 1,
                   wall_tag="overlap", center_scale=0.9, quadrant_scale=0.9),
    TURNS * PITCH, N_R * TURNS, element_tags="fluid", first_tag="inlet", last_tag="outlet")
core = hexmesh.rotate(core, THETA_OFFSET)
core = hexmesh.translate(core, (0.0, 0.0, Z_OFFSET))

axial = affine.translation([0.0, 0.0, TURNS * PITCH])

mesh_periodic = [
    hexmesh.Periodic("inlet", "outlet", axial),
    hexmesh.Periodic("cut_lo", "cut_hi", axial),
    hexmesh.Periodic("zlo", "zhi", axial),
    hexmesh.Periodic("zlo_solid", "zhi_solid", axial),
]
mesh_bc = BoundaryConditions(
    velocity={"coil": ("v  ", 1), "wall": ("W  ", 2), "core": ("int", 3),
              "inlet": ("P  ", 4), "outlet": ("P  ", 5), "zlo": ("P  ", 6), "zhi": ("P  ", 7)},
    temperature={"core": ("int", 1), "outer": ("f  ", 2),
                 "inlet": ("P  ", 3), "outlet": ("P  ", 4),
                 "cut_lo": ("P  ", 5), "cut_hi": ("P  ", 6),
                 "zlo": ("P  ", 7), "zhi": ("P  ", 8),
                 "zlo_solid": ("P  ", 9), "zhi_solid": ("P  ", 10)})

core_periodic = [hexmesh.Periodic("inlet", "outlet", axial)]
core_bc = BoundaryConditions(
    velocity={"overlap": ("int", 1), "inlet": ("P  ", 2), "outlet": ("P  ", 3)},
    temperature={"overlap": ("int", 1), "inlet": ("P  ", 2), "outlet": ("P  ", 3)})

print(hexmesh.report(mesh))
print(hexmesh.report(core))
for m, p in ((mesh, mesh_periodic), (core, core_periodic)):
    print("periodic pairs:", hexmesh.periodic_pairs(m, p).rows.shape[0], "rows, worst residual",
          hexmesh.periodic_pairs(m, p).worst)
writer.to_re2(mesh, "wire_coil_neknek.re2", bc=mesh_bc, periodic=mesh_periodic, fluid="fluid")
writer.to_re2(core, "wire_coil_neknek_core.re2", bc=core_bc, periodic=core_periodic)
writer.to_vtu(mesh, "wire_coil_neknek.vtu", bc=mesh_bc)
writer.to_vtu(core, "wire_coil_neknek_core.vtu", bc=core_bc)

