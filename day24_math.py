#!/usr/bin/env python3

# Key insight from https://www.reddit.com/r/adventofcode/comments/18pnycy/2023_day_24_solutions/kepu26z/
# Geometric interpretation:
# - (Rock-)relative position of hailstone must be parallel to the relative velocity vector
#   => will result in a collision (from the stone's perspective)
# - Cross product of those vectors must therefore be the zero vector
# - This hold true for any two hailstones so we get 3 linear equations
# - So for the 6 unknowns we just need two hailstone pairs to get a solvable linear system of equations

# This script just helps with getting the correct matrix values for the actual solution.

from sympy import *

px, py, pz, vx, vy, vz = symbols("px, py, pz, vx, vy, vz")
px0, py0, pz0, vx0, vy0, vz0 = symbols("px0, py0, pz0, vx0, vy0, vz0")
px1, py1, pz1, vx1, vy1, vz1 = symbols("px1, py1, pz1, vx1, vy1, vz1")
p = Matrix([px, py, pz])
v = Matrix([vx, vy, vz])
p0 = Matrix([px0, py0, pz0])
v0 = Matrix([vx0, vy0, vz0])
p1 = Matrix([px1, py1, pz1])
v1 = Matrix([vx1, vy1, vz1])
# (p - p0) x (v - v0) = 0 = (p - p1) x (v - v1)
# (p - p0) x (v - v0) - (p - p1) x (v - v1) = 0
result = (p - p0).cross(v - v0) - (p - p1).cross(v - v1)
print(collect(expand(result[0]), [px, py, pz, vx, vy, vz]))
print(collect(expand(result[1]), [px, py, pz, vx, vy, vz]))
print(collect(expand(result[2]), [px, py, pz, vx, vy, vz]))

# Manually rearranged:
# px                py                pz                vx                vy                vz                    const
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# px*0            + py*(-vz0 + vz1) + pz*(vy0 - vy1)  + vx*0            + vy*(pz0 - pz1)  + vz*(-py0 + py1)     + py0*vz0 - pz0*vy0 - py1*vz1 + pz1*vy1
# px*(vz0 - vz1)  + py*0            + pz*(-vx0 + vx1) + vx*(-pz0 + pz1) + vy*0            + vz*(px0 - px1)      + pz0*vx0 - px0*vz0 - pz1*vx1 + px1*vz1
# px*(-vy0 + vy1) + py*(vx0 - vx1)    pz*0            + vx*(py0 - py1)  + vy*(-px0 + px1) + vz*0                + px0*vy0 - py0*vx0 - px1*vy1 + py1*vx1
# -----------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                                                          short:      p0 x v0      -      p1 x v1
