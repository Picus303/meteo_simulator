from __future__ import annotations

from typing import Callable, Iterable, Literal, Optional

from .state import State
from .tendencies import Tend, advance_state, lincomb_states

# RHS signature: f(state, t) -> Tend
RHSFunc = Callable[[State, float], Tend]
PostHook = Callable[[State, float, float, Literal["stage", "final"]], State]


def heun2(state: State, t: float, dt: float, f: RHSFunc, post: Optional[Iterable[PostHook]] = None,
          enforce_at: Literal["final", "stage"] = "final") -> State:
    """Heun's method (RK2). Optionally apply post hooks at each stage or only final."""
    # k1
    k1 = f(state, t)
    s1 = advance_state(state, k1, dt)
    if enforce_at == "stage":
        s1 = _apply_hooks(s1, post, t + dt, dt, when="stage")
    # k2
    k2 = f(s1, t + dt)
    s2 = lincomb_states(0.5, state, 0.5, advance_state(s1, k2, 0.0))  # average of states with Euler predictor
    out = s2
    if enforce_at == "final":
        out = _apply_hooks(out, post, t + dt, dt, when="final")
    return out


def rk4(state: State, t: float, dt: float, f: RHSFunc, post: Optional[Iterable[PostHook]] = None,
        enforce_at: Literal["final", "stage"] = "final") -> State:
    """Classic RK4 (4th order)."""
    k1 = f(state, t)
    s2 = advance_state(state, k1, dt/2)
    if enforce_at == "stage":
        s2 = _apply_hooks(s2, post, t + dt/2, dt/2, when="stage")

    k2 = f(s2, t + dt/2)
    s3 = advance_state(state, k2, dt/2)
    if enforce_at == "stage":
        s3 = _apply_hooks(s3, post, t + dt/2, dt/2, when="stage")

    k3 = f(s3, t + dt/2)
    s4 = advance_state(state, k3, dt)
    if enforce_at == "stage":
        s4 = _apply_hooks(s4, post, t + dt, dt, when="stage")

    k4 = f(s4, t + dt)

    out = State(
        M=state.M + (dt/6)*(k1.dM + 2*k2.dM + 2*k3.dM + k4.dM),
        T=state.T + (dt/6)*(k1.dT + 2*k2.dT + 2*k3.dT + k4.dT),
        qv=state.qv + (dt/6)*(k1.dqv + 2*k2.dqv + 2*k3.dqv + k4.dqv),
        qc=state.qc + (dt/6)*(k1.dqc + 2*k2.dqc + 2*k3.dqc + k4.dqc),
        qr=state.qr + (dt/6)*(k1.dqr + 2*k2.dqr + 2*k3.dqr + k4.dqr),
        MU=state.MU + (dt/6)*(k1.dMU + 2*k2.dMU + 2*k3.dMU + k4.dMU),
        MV=state.MV + (dt/6)*(k1.dMV + 2*k2.dMV + 2*k3.dMV + k4.dMV),
        Esfc=None if state.Esfc is None else state.Esfc.copy(),
    )
    if enforce_at == "final":
        out = _apply_hooks(out, post, t + dt, dt, when="final")
    return out


def rk3_ssp(state: State, t: float, dt: float, f: RHSFunc, post: Optional[Iterable[PostHook]] = None,
            enforce_at: Literal["final", "stage"] = "final") -> State:
    """
    Shu-Osher SSP RK3 (strong-stability preserving), stage-optional hooks.

        u1 = u^n + dt f(u^n)
        u2 = 3/4 u^n + 1/4 (u1 + dt f(u1))
        u3 = 1/3 u^n + 2/3 (u2 + dt f(u2))
    """
    u0 = state
    k1 = f(u0, t)
    u1 = advance_state(u0, k1, dt)
    if enforce_at == "stage":
        u1 = _apply_hooks(u1, post, t + dt, dt, when="stage")

    k2 = f(u1, t + dt)
    u2 = lincomb_states(3/4, u0, 1/4, advance_state(u1, k2, dt))
    if enforce_at == "stage":
        u2 = _apply_hooks(u2, post, t + dt, dt, when="stage")

    k3 = f(u2, t + dt/2)  # time label not essential here; half-step is common choice
    u3 = lincomb_states(1/3, u0, 2/3, advance_state(u2, k3, dt))

    out = u3
    if enforce_at == "final":
        out = _apply_hooks(out, post, t + dt, dt, when="final")
    return out


def _apply_hooks(s: State, post: Optional[Iterable[PostHook]], t: float, dt: float, when: Literal["stage", "final"]) -> State:
    if post is None:
        return s
    out = s
    for h in post:
        out = h(out, t, dt, when)
    return out