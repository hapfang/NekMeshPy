"""2-D conjugate channel with a free-standing rectangular obstacle.

Fluid channel ``[-5,11] x [1,3]`` with a rectangular obstacle ``[-0.05,0.05] x
[1.25,1.55]`` floating inside it, and a solid wall of thickness ``T_WALL`` on the
bottom (``y in [1-T,1]``) and top (``y in [3,3+T]``).

The domain is a 3x5 grid of structured rectangles cut at the x/y breaks below; the
obstacle's cell is the one left out. Only the true outer sides are tagged, so
``merge`` welds every shared edge into an interior one. Fluid blocks carry
``element_tags="fluid"``, wall blocks ``"solid"``; the fluid/solid interface is just
an interior edge between the two regions.

Boundary names: ``inlet`` / ``outlet`` (fluid ends), ``obstacle`` (its four sides),
``solid_end`` (wall ends), ``solid_outer`` (the wall's free faces).

    PYTHONPATH=. python examples/channel_obstacle_cht.py

Produces ``channel_obstacle_cht.vtu``.
"""

import logging


from nekmeshpy import quadmesh, writer
from nekmeshpy.core.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
X0, X1 = -5.0, 11.0          # channel extent
Y0, Y1 = 1.0, 3.0
OBS_X = (-0.05, 0.05)        # obstacle
OBS_Y = (1.25, 1.55)
T_WALL = 0.2                 # solid thickness, top and bottom
N_X = (60, 4, 120)           # cells: upstream / over the obstacle / downstream
X_RATIO = (1.04, 1.0, 1.025)  # geometric size ratio, clustered toward the obstacle
N_Y = (6, 6, 14)             # fluid cells: below / beside / above the obstacle
N_WALL = 4                   # cells through each solid wall
ORDER = 2
OUT_NAME = "channel_obstacle_cht"

xb = [X0, OBS_X[0], OBS_X[1], X1]
yb = [Y0 - T_WALL, Y0, OBS_Y[0], OBS_Y[1], Y1, Y1 + T_WALL]
ny = [N_WALL, *N_Y, N_WALL]
region = ["solid", "fluid", "fluid", "fluid", "solid"]

xfrac = [1.0 - geometric_spacing(N_X[0], X_RATIO[0])[::-1],
         geometric_spacing(N_X[1], X_RATIO[1]),
         geometric_spacing(N_X[2], X_RATIO[2])]

blocks = []
for j in range(5):
    for i in range(3):
        if (i, j) == (1, 2):                     # the obstacle itself
            continue
        sides = {}
        if i == 0:
            sides["left"] = ("inlet" if region[j] == "fluid" else "solid_end")
        if i == 2:
            sides["right"] = ("outlet" if region[j] == "fluid" else "solid_end")
        if j == 0:
            sides["bottom"] = "solid_outer"
        if j == 4:
            sides["top"] = "solid_outer"
        if (i, j) == (1, 1):
            sides["top"] = "obstacle"
        if (i, j) == (1, 3):
            sides["bottom"] = "obstacle"
        if (i, j) == (0, 2):
            sides["right"] = "obstacle"
        if (i, j) == (2, 2):
            sides["left"] = "obstacle"
        x0, x1, y0, y1 = xb[i], xb[i + 1], yb[j], yb[j + 1]
        blocks.append(quadmesh.rectangle(
            [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)],
            N_X[i], ny[j], x_frac=xfrac[i], side_tags=sides,
            element_tags=region[j], order=ORDER))

mesh = quadmesh.merge(blocks)

print("channel_obstacle_cht: %d quads (%d fluid), %d points, order %d"
      % (mesh.n_quads, int((mesh.element_tags.to_dense() == "fluid").sum()),
         mesh.n_points, mesh.order))
print(quadmesh.quality_summary(mesh))
writer.quad_to_vtu(mesh, OUT_NAME + ".vtu")
