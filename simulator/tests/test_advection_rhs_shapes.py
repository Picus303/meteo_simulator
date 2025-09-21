import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.advection.term import make_advection_term


def test_advection_term_shapes():
    g = make_latlon_grid(20, 10, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    term = make_advection_term(g, tracer_names=("T","qv"))
    k = term(st, t=0.0)

    assert k.dM.shape == (g.ny, g.nx)
    assert k.dT.shape == (g.ny, g.nx)
    assert k.dqv.shape == (g.ny, g.nx)
    assert k.dMU.shape == (g.ny, g.nx + 1)
    assert k.dMV.shape == (g.ny + 1, g.nx)