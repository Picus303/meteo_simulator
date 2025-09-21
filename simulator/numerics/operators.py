import numpy as np
from ..grid.metrics import _dx_dy_faces_from_centers


def grad_center_to_UV(A: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient métrique centres → faces :
      (∂xA)_U[j,i+1/2] = (A[j,i+1] - A[j,i]) / dx_u[j,i+1/2]   (périodique en i)
      (∂yA)_V[j+1/2,i] = (A[j+1,i] - A[j,i]) / dy_v[j+1/2,i]   (clamp en j: 0 aux pôles)
    Renvoie:
      Gx_U: (ny,   nx+1)
      Gy_V: (ny+1, nx)
    """
    ny, nx = A.shape
    dx_u, dy_v = _dx_dy_faces_from_centers(grid)  # (ny,nx+1), (ny+1,nx)

    # --- Zonal (vers U): construire explicitement les paires (gauche, droite) pour les nx+1 faces
    A_left  = np.concatenate([A[:, -1:], A], axis=1)   # (ny, nx+1)  [i-1, i, ..., nx-1]
    A_right = np.concatenate([A, A[:, :1]], axis=1)    # (ny, nx+1)  [0, 1, ..., 0]
    Gx = (A_right - A_left) / np.maximum(dx_u, 1e-30)  # (ny, nx+1)

    # --- Méridien (vers V): clamp aux pôles (no-flux)
    Gy = np.empty((ny + 1, nx), dtype=A.dtype)
    Gy[1:-1, :] = (A[1:, :] - A[:-1, :]) / np.maximum(dy_v[1:-1, :], 1e-30)
    Gy[0, :]    = 0.0
    Gy[-1, :]   = 0.0                               # (ny+1, nx)

    return Gx, Gy


def grad_half_g_h2_to_UV(M: np.ndarray, grid, g: float, rho_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient to U/V of Φ = 1/2 g (M/rho)^2 :
      returns ( ∂xΦ |_U , ∂yΦ |_V ).
    """
    Phi = 0.5 * g * (M / rho_ref) ** 2
    return grad_center_to_UV(Phi, grid)


def raw_div_c(Fu: np.ndarray, Fv: np.ndarray) -> np.ndarray:
    """
    *Non-area-normalized* divergence at centers.

    Fu: (ny,   nx+1)  zonal flux at U faces (positive eastward)
    Fv: (ny+1, nx  )  meridional flux at V faces (positive northward)

    Returns:
      (ny, nx)  = (Fu[:,1:] - Fu[:,:-1]) + (Fv[1:,:] - Fv[:-1,:])
    """
    return (Fu[:, 1:] - Fu[:, :-1]) + (Fv[1:, :] - Fv[:-1, :])

def div_c_from_fluxes(Fu: np.ndarray, Fv: np.ndarray, area: np.ndarray, *, tiny: float = 1e-30) -> np.ndarray:
    """
    Centered divergence *per unit area* (s⁻¹) on the C grid.

    Fu:   (ny,   nx+1)  mass flux or other quantity (kg/s) on U
    Fv:   (ny+1, nx  )  flux on V
    area: (ny,   nx  )  cell area (m²)
    tiny: small constant to avoid division by zero

    Returns:
      div/area of shape (ny, nx)
    """
    return raw_div_c(Fu, Fv) / np.maximum(area, tiny)