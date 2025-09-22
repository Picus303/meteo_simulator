from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.core.steppers import rk3_ssp
from simulator.core.tendencies import Tend
from simulator.operators.drag import drag_tendencies, DragConfig
from simulator.diagnostics.energy_budget import enthalpy
from simulator.diagnostics.energy_exchange import kinetic_energy
from simulator.grid.staggering import to_u_centered


def _rhs_drag(state, grid, r, heat, cp):
    def f(st: State, t: float) -> Tend:
        return drag_tendencies(st, grid, DragConfig(r_u=r, r_v=r, heat=heat, cp=cp))
    return f

def test_drag_with_heating_closes_energy_centers():
    g = make_latlon_grid(73, 37, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    
    M_u = to_u_centered(st.M)
    st.MU[:] = M_u * 15.0

    cp = 1004.0
    E0 = kinetic_energy(st.MU, st.MV, st.M, g)
    H0 = enthalpy(st.M, st.T, g, cp)

    f = _rhs_drag(st, g, r=1e-5, heat=True, cp=cp)
    s = rk3_ssp(st, 0.0, 120.0, f, post=None, enforce_at="final")

    E1 = kinetic_energy(s.MU, s.MV, s.M, g)
    H1 = enthalpy(s.M, s.T, g, cp)
    rel = abs((E1 + H1) - (E0 + H0)) / max(1e-12, abs(E0 + H0))
    assert rel < 1e-8