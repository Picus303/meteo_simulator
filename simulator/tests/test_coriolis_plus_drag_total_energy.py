from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.core.steppers import rk3_ssp
from simulator.core.tendencies import Tend
from simulator.operators.coriolis import make_coriolis_rotation_hook, CoriolisRotateConfig
from simulator.operators.drag import drag_tendencies, DragConfig
from simulator.diagnostics.energy_budget import total_energy
from simulator.grid.staggering import to_u_centered


def _rhs_drag_only(grid, cp):
    def f(st: State, t: float) -> Tend:
        return drag_tendencies(st, grid, DragConfig(r_u=1e-5, r_v=1e-5, heat=True, cp=cp))
    return f


def test_coriolis_rotation_plus_drag_maintains_total_energy():
    g = make_latlon_grid(73, 37, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    
    M_u = to_u_centered(st.M)
    st.MU[:] = M_u * 12.0

    cp = 1004.0; gacc = 9.81; rho_ref = 1.2
    hook = make_coriolis_rotation_hook(g, CoriolisRotateConfig(cap_zero_u=False, scheme="centers"))

    E0 = total_energy(st.MU, st.MV, st.M, st.T, g, cp, gacc, rho_ref)

    s = st.copy(); t = 0.0
    for _ in range(20):
        s = rk3_ssp(s, t, 120.0, _rhs_drag_only(g, cp), post=[hook], enforce_at="final")
        t += 120.0

    E1 = total_energy(s.MU, s.MV, s.M, s.T, g, cp, gacc, rho_ref)
    rel = abs(E1 - E0) / max(1e-12, abs(E0))
    assert rel < 1e-8