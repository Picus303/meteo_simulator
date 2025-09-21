from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..core.state import State
from ..core.tendencies import Tend
from ..grid.sphere import Grid
from ..grid.metrics import _dual_areas
from ..grid.staggering import to_u_centered, to_v_centered
from .fluxform import mass_fluxes


@dataclass
class MomAdvConfig:
	eps_mass_rel: float = 1e-12


def _velocities_on_faces(state: State, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Return (u, v, M_u, M_v)."""
	M_u = to_u_centered(state.M)
	M_v = to_v_centered(state.M)
	u = state.MU / np.maximum(M_u, eps)
	v = state.MV / np.maximum(M_v, eps)
	return u, v, M_u, M_v


def _u_on_V(u: np.ndarray) -> np.ndarray:
	"""Interpolate U-face velocity u (ny, nx+1) onto V faces (ny+1, nx).
	- i: average neighbors u[:, i] and u[:, i+1] (periodic in i for the +1 at nx)
	- j: clamp at boundaries (replicate 1st/last row)
	"""
	ny, nxp1 = u.shape
	nx = nxp1 - 1
	out = np.empty((ny + 1, nx), dtype=u.dtype)
	# horizontal average
	uh = 0.5 * (u[:, :nx] + u[:, 1:])  # (ny, nx)
	# vertical clamp
	out[1:-1, :] = 0.5 * (uh[:-1, :] + uh[1:, :])
	out[0, :] = uh[0, :]
	out[-1, :] = uh[-1, :]
	return out


def _v_on_U(v: np.ndarray) -> np.ndarray:
	"""Interpolate V-face velocity v (ny+1, nx) onto U faces (ny, nx+1).
	- i: average v[:, i-1] and v[:, i] with periodic wrap for i-1 at 0
	- j: average v[j] and v[j+1] (clamp top with last row)
	"""
	nyp1, nx = v.shape
	ny = nyp1 - 1
	out = np.empty((ny, nx + 1), dtype=v.dtype)
	# vertical average to U rows
	vv = 0.5 * (v[:ny, :] + v[1:, :])  # (ny, nx)
	# horizontal wrap-average to U columns
	left = np.concatenate([vv[:, -1:], vv], axis=1)   # (ny, nx+1)
	right = np.concatenate([vv, vv[:, :1]], axis=1)   # (ny, nx+1)
	out[:] = 0.5 * (left + right)
	return out


def advect_momentum(state: State, grid: Grid, cfg: MomAdvConfig = MomAdvConfig()) -> Tend:
	"""Return tendencies for MU,MV due to advection by mass fluxes on U/V control volumes.

	Discrete form:
	  d(Mu)/dt = -div_U( [u*Fu]_{east-west}, [u|V * Fv]_{north-south} )
	  d(Mv)/dt = -div_V( [v|U * Fu]_{east-west}, [v*Fv]_{north-south} )

	where Fu,Fv are mass fluxes at U/V faces from Step 6 and u|V, v|U are
	co-located interpolations.
	"""
	eps = cfg.eps_mass_rel * float(np.mean(state.M))

	# Mass fluxes at faces
	Fu, Fv = mass_fluxes(state, grid)

	# Face velocities and dual areas
	u, v, _, _ = _velocities_on_faces(state, eps)
	Au, Av = _dual_areas(grid)

	# ── U component (on U control volumes)
	# East-West flux difference on U columns (periodic in i)
	Gx_u = u * Fu  # (ny, nx+1)
	Gx_u_east = np.concatenate([Gx_u[:, 1:], Gx_u[:, :1]], axis=1)
	dGx_u = Gx_u_east - Gx_u  # (ny, nx+1) east - west

	# North-South flux difference using u interpolated on V rows, lifted back to U columns
	u_on_V = _u_on_V(u)           # (ny+1, nx)
	Gy_u_base = u_on_V * Fv       # (ny+1, nx)
	# average Gy to U columns (between V columns i-1 and i), then take north - south
	Gy_left = np.concatenate([Gy_u_base[:, -1:], Gy_u_base], axis=1)   # (ny+1, nx+1)
	Gy_right = np.concatenate([Gy_u_base, Gy_u_base[:, :1]], axis=1)   # (ny+1, nx+1)
	Gy_u = 0.5 * (Gy_left + Gy_right)
	dGy_u = Gy_u[1:, :] - Gy_u[:-1, :]                                  # (ny, nx+1)

	dMU = -(dGx_u + dGy_u) / np.maximum(Au, 1e-30)

	# ── V component (on V control volumes)
	# 1) Lift Fu and v_on_U onto V rows (clamp vertically), compute x-flux on the EDGES of V
	v_on_U = _v_on_U(v)           				# (ny, nx+1)
	Fu_V = np.empty((grid.ny + 1, grid.nx + 1), dtype=Fu.dtype)
	Fu_V[1:-1, :] = 0.5 * (Fu[:-1, :] + Fu[1:, :])
	Fu_V[0, :] = Fu[0, :]
	Fu_V[-1, :] = Fu[-1, :]

	v_on_U_V = np.empty_like(Fu_V)
	v_on_U_V[1:-1, :] = 0.5 * (v_on_U[:-1, :] + v_on_U[1:, :])
	v_on_U_V[0, :] = v_on_U[0, :]
	v_on_U_V[-1, :] = v_on_U[-1, :]

	# x-flux on the EDGES (ny+1, nx+1), then "east − west" → (ny+1, nx)
	Gx_v = v_on_U_V * Fu_V                       # (ny+1, nx+1)
	dGx_v = Gx_v[:, 1:] - Gx_v[:, :-1]           # (ny+1, nx)

	# 2) y-flux directly on the north/south EDGES of V, then "north − south" (no roll in j)
	Gy_v = v * Fv                                # (ny+1, nx)
	dGy_v = np.empty_like(Gy_v)
	dGy_v[1:-1, :] = Gy_v[2:, :] - Gy_v[1:-1, :]
	dGy_v[0, :] = Gy_v[1, :] - Gy_v[0, :]
	dGy_v[-1, :] = Gy_v[-1, :] - Gy_v[-2, :]

	# Divergence / dual area Av : shapes (ny+1, nx) everywhere
	dMV = -(dGx_v + dGy_v) / np.maximum(Av, 1e-30)

	# Caps: ensure no tendency on U in cap rows (Fu is zero there already; this is a guard)
	if getattr(grid, "cap_rows", None) is not None and np.any(grid.cap_rows):
		dMU = dMU.copy()
		dMU[grid.cap_rows, :] = 0.0

	zc = np.zeros_like(state.M)
	return Tend(dM=zc, dT=zc, dqv=zc, dqc=zc, dqr=zc, dMU=dMU, dMV=dMV)