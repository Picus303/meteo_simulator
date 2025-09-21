import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import cell_areas
from simulator.grid.staggering import to_u_centered
from simulator.core.state import State
from simulator.advection.fluxform import advect_mass_and_tracers, mass_fluxes


def test_rhs_conserves_mass_and_tracer_integrals():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=85.0)
    A = cell_areas(g)

    st = State.zeros(g.ny, g.nx)
    # Build a non-trivial tracer and mass field
    Y, X = np.meshgrid(g.lat_c, g.lon_c, indexing="ij")
    st.M = 1.0 + 0.2*np.sin(Y)  # nonuniform mass
    phi = 0.5 + 0.5*np.cos(X)   # smooth tracer (e.g., qv)
    st.qv[:] = phi

    # Impose a constant zonal velocity u0 by setting MU = M_u * u0
    u0 = 20.0  # m/s
    M_u = to_u_centered(st.M)
    st.MU = M_u * u0
    st.MV[:] = 0.0

    tend = advect_mass_and_tracers(st, g, {"qv": st.qv})

    # Global mass tendency integral should be ~0 (periodic zonal, v=0, and Fu cancels)
    dMdt_int = float(np.sum(tend.dM * A))
    assert abs(dMdt_int) < 1e-10

    # For tracer, integral of M*q should be conserved: ∫(M dq + q dM) dA = 0
    integrand = st.M * tend.dqv + st.qv * tend.dM
    dMqdt_int = float(np.sum(integrand * A))
    assert abs(dMqdt_int) < 1e-8

    # Check that mass tendency is small compared to typical mass flux scale
    scale = np.sum(np.abs(tend.dM) * A)
    tol = max(1e-12 * max(1.0, scale), 1e-12)
    assert abs(dMdt_int) <= tol