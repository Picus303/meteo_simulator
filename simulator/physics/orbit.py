from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..grid.sphere import Grid

TWOPI = 2.0 * np.pi

# -----------------------------
# Keplerian orbit + planetary spin
# -----------------------------
@dataclass
class KeplerOrbit:
    a: float = 1.0             # semi-major axis (arbitrary units, consistent with S0 scaling)
    e: float = 0.0             # eccentricity
    i_orb: float = 0.0         # inclination [rad]
    Omega: float = 0.0         # longitude of ascending node [rad]
    omega: float = 0.0         # argument of periapsis [rad]
    M0: float = 0.0            # mean anomaly at t=0 [rad]
    year_length: float = 365.0 * 86400.0  # orbital period [s]

@dataclass
class SpinConfig:
    day_length: float = 86400.0      # sidereal day [s]
    theta0: float = 0.0              # reference prime-meridian angle at t=0 [rad]
    # spin axis via right ascension / declination in inertial frame
    ra_pole: float = 0.0             # α_p [rad]
    dec_pole: float = np.radians(23.44)  # δ_p [rad] (Earth-like obliquity)

# -----------------------------
# Linear algebra helpers
# -----------------------------

def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    # Rotate vector v around unit axis by angle
    k = axis / np.linalg.norm(axis)
    vpar = np.dot(v, k) * k
    vperp = v - vpar
    kxv = np.cross(k, v)
    return vpar + vperp * np.cos(angle) + kxv * np.sin(angle)

# -----------------------------
# Kepler solver and geometry
# -----------------------------

def mean_motion(year_length: float) -> float:
    return TWOPI / year_length


def kepler_E(M: float, e: float, tol: float = 1e-12, itmax: int = 32) -> float:
    """Solve Kepler's equation M = E - e sin E for E (rad)."""
    M = (M + np.pi) % TWOPI - np.pi  # wrap
    E = M if e < 0.8 else np.pi
    for _ in range(itmax):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        d = -f / fp
        E += d
        if abs(d) < tol:
            break
    return E


def true_anomaly(E: float, e: float) -> float:
    """
    Robust true anomaly from E using atan2 form (avoids div by zero at E≈π).
    v = 2 atan2(√(1+e) sin(E/2), √(1-e) cos(E/2))
    """
    s = np.sqrt(1.0 + e) * np.sin(0.5 * E)
    c = np.sqrt(1.0 - e) * np.cos(0.5 * E)
    nu = 2.0 * np.arctan2(s, c)
    return nu


def radius_from_true_anomaly(a: float, e: float, nu: float) -> float:
    return a * (1 - e * e) / (1 + e * np.cos(nu))


def sun_dir_inertial_and_r(t: float, orb: KeplerOrbit) -> tuple[np.ndarray, float]:
    """Return unit sun direction in inertial frame (from planet to sun) and distance r."""
    n = mean_motion(orb.year_length)
    M = orb.M0 + n * t
    E = kepler_E(M, orb.e)
    nu = true_anomaly(E, orb.e)
    r = radius_from_true_anomaly(orb.a, orb.e, nu)
    # position vector in perifocal (PQW)
    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    # rotate to inertial: Rz(Omega) Rx(i) Rz(omega)
    R = _rot_z(orb.Omega) @ _rot_x(orb.i_orb) @ _rot_z(orb.omega)
    r_I = R @ r_pqw
    s_I = -r_I / np.linalg.norm(r_I)
    return s_I, r


def spin_axis_vector(spin: SpinConfig) -> np.ndarray:
    cp = np.cos(spin.dec_pole)
    return np.array([cp * np.cos(spin.ra_pole), cp * np.sin(spin.ra_pole), np.sin(spin.dec_pole)])


def subsolar_lonlat(t: float, orb: KeplerOrbit, spin: SpinConfig) -> tuple[float, float]:
    """
    Return (phi_s, lambda_s) of subsolar point in the rotating body frame.
    z-axis is the spin axis. Longitude increases with planetary rotation.
    """
    s_I, _ = sun_dir_inertial_and_r(t, orb)
    S = spin_axis_vector(spin)
    # Build equatorial basis (E,N,S) with E from inertial x projected on equator
    ex = np.array([1.0, 0.0, 0.0])
    E0 = ex - np.dot(ex, S) * S
    if np.linalg.norm(E0) < 1e-12:
        ex = np.array([0.0, 1.0, 0.0])
        E0 = ex - np.dot(ex, S) * S
    E0 /= np.linalg.norm(E0)
    N0 = np.cross(S, E0)
    # Components of sun direction in this body-fixed basis before rotation of the prime meridian
    sx, sy, sz = np.dot(s_I, E0), np.dot(s_I, N0), np.dot(s_I, S)
    lambda_s0 = np.arctan2(sy, sx)
    phi_s = np.arcsin(np.clip(sz, -1.0, 1.0))
    # Planetary rotation: prime meridian angle θ(t)
    theta = spin.theta0 + TWOPI * (t / spin.day_length)
    lambda_s = lambda_s0 - theta
    # wrap to [-pi, pi)
    lambda_s = (lambda_s + np.pi) % TWOPI - np.pi
    return phi_s, lambda_s


def mu0_field(grid: Grid, t: float, orb: KeplerOrbit, spin: SpinConfig) -> np.ndarray:
    phi_s, lam_s = subsolar_lonlat(t, orb, spin)
    lon = grid.lon_c[None, :]
    lat = grid.lat_c[:, None]
    mu = np.sin(lat) * np.sin(phi_s) + np.cos(lat) * np.cos(phi_s) * np.cos(lon - lam_s)
    return np.clip(mu, 0.0, 1.0)


def orbit_radius(t: float, orb: KeplerOrbit) -> float:
    _, r = sun_dir_inertial_and_r(t, orb)
    return r