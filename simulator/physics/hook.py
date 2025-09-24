from __future__ import annotations
from typing import Callable, Iterable, Literal

from ..core.state import State
from ..core.tendencies import Tend
from ..core.steppers import rk3_ssp, PostHook
from ..grid.sphere import Grid
from ..physics.block import physics_step_strang, PhysicsConfig, PhysicsInputs


# ── Hook maker: apply a **half** physics step when called
def make_physics_half_hook(grid: Grid,
                           orb_spin: tuple,
                           pcfg: PhysicsConfig,
                           pin: PhysicsInputs) -> Callable[[State, float, float, str], State]:
    """
    Return a HOOK(state, t, dt, when)->state that applies **½-physics** over dt/2.

    The hook ignores `when` unless you want to conditionally call it (e.g., only at PRE/FINAL).
    It runs the same sub-sequence as physics_step_strang but at half dt.
    """
    def hook(state: State, t: float, dt: float, when: str = "stage") -> State:
        # We simply call the block with dt_half and return the updated state
        dt_half = 0.5 * dt
        s, _ = physics_step_strang(state, grid, t, dt_half, orb_spin, pcfg, pin)
        return s
    return hook


# ── Convenience: advance one step with Strang physics around RK3 dynamics
def advance_with_physics(state: State, t: float, dt: float, grid: Grid,
                         rhs: Callable[[State, float], Tend],
                         orb_spin: tuple,
                         pcfg: PhysicsConfig,
                         pin: PhysicsInputs,
                         post_stage_hooks: Iterable[PostHook] = ()) -> State:
    """
    Perform:  ½-physics  → RK3(dynamics) →  ½-physics.

    `post_stage_hooks` are applied at each RK3 stage/final (e.g., cap projection, positivity clip).
    This wrapper doesn't change your existing RK3; it just surrounds it with two half-physics steps.
    """

    pre = make_physics_half_hook(grid, orb_spin, pcfg, pin)
    post = make_physics_half_hook(grid, orb_spin, pcfg, pin)

    # 1) PRE: half physics
    s = pre(state, t, dt, "stage")

    # 2) Dynamics via RK3
    s = rk3_ssp(s, t, dt, rhs, post=list(post_stage_hooks), enforce_at="stage")

    # 3) POST: half physics
    s = post(s, t + dt, dt, "final")
    return s