"""
spp.py - single point positioning by least squares, used three ways.

  1. Pseudorange WLS -> position p, clock bias, Q = (H^T W H)^-1
                        -> HDOP -> sigma_h = HDOP * sigma_UERE
  2. Doppler WLS     -> velocity v, using the SAME H rows [-e_i^T, 1]
  3. Residual test   -> SSE = r^T W r / sigma^2 ~ chi2(n - 4); drop the worst
                        satellite and re-solve while the test fails (RAIM-lite)

plus the velocity filter: least squares with an IMU prior, which is
algebraically the Kalman update (see ls_with_prior / kalman_update).

Conventions
-----------
* Satellites and the Gauss-Newton iteration are in ECEF metres. Everything
  handed to boundary.py is ENU metres about params.ORIGIN_*.
* e_i is the unit line-of-sight vector from the receiver to satellite i.
* An H row is [-e_i^T, 1], the derivative of ||s_i - x|| + b w.r.t. [x, b].
* W = diag(w_i) holds dimensionless weights, w_i = sin^2(el_i), so a zenith
  satellite has weight 1. The measurement covariance is sigma^2 * W^-1 and
  cov(x_hat) = sigma^2 * Q with Q = (H^T W H)^-1. Q is the weighted DOP
  matrix, so sigma_h = HDOP * sigma_UERE holds exactly.

Only numpy is required. The input is one Epoch per GNSS epoch (see Epoch and
load_csv); UrbanNav RINEX has to be turned into satellite positions and
velocities first (your lab ephemeris code, RTKLIB, ...).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np

import params as P

# --------------------------------------------------------------------------
# Constants (WGS-84 / GPS ICD)
# --------------------------------------------------------------------------
C = 299_792_458.0             # m/s
OMEGA_E = 7.2921151467e-5     # rad/s, Earth rotation rate
F_L1 = 1575.42e6              # Hz
LAMBDA_L1 = C / F_L1          # 0.1903 m
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


# ==========================================================================
# Geodesy
# ==========================================================================
def geodetic_to_ecef(lat_deg, lon_deg, h):
    """Geodetic (deg, deg, m) -> ECEF (m). Accepts scalars or arrays."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(xyz):
    """ECEF (m) -> (lat deg, lon deg, h m). Iterative, converges in ~4 steps."""
    x, y, z = np.asarray(xyz, dtype=float)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    h = 0.0
    for _ in range(8):
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    return math.degrees(lat), math.degrees(lon), h


def enu_rotation(lat_deg, lon_deg):
    """R such that v_enu = R @ v_ecef."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


class LocalFrame:
    """ENU frame about a fixed origin (shared by spp.py and boundary.py)."""

    def __init__(self, lat_deg=P.ORIGIN_LAT, lon_deg=P.ORIGIN_LON, h=P.ORIGIN_H):
        self.lat, self.lon, self.h = lat_deg, lon_deg, h
        self.ecef0 = geodetic_to_ecef(lat_deg, lon_deg, h)
        self.R = enu_rotation(lat_deg, lon_deg)

    def to_enu(self, xyz):
        return (np.asarray(xyz) - self.ecef0) @ self.R.T

    def to_ecef(self, enu):
        return np.asarray(enu) @ self.R + self.ecef0

    def latlon_to_enu(self, lat_deg, lon_deg, h=None):
        h = self.h if h is None else h
        return self.to_enu(geodetic_to_ecef(lat_deg, lon_deg, h))


def elev_az(x_ecef, e_ecef):
    """Elevation and azimuth (rad) of unit LOS vectors seen from x_ecef."""
    lat, lon, _ = ecef_to_geodetic(x_ecef)
    e_enu = e_ecef @ enu_rotation(lat, lon).T
    el = np.arcsin(np.clip(e_enu[:, 2], -1.0, 1.0))
    az = np.mod(np.arctan2(e_enu[:, 0], e_enu[:, 1]), 2.0 * np.pi)
    return el, az


# ==========================================================================
# Measurement models
# ==========================================================================
def tropo_delay(el_rad):
    """Simple elevation-mapped troposphere delay (m): 2.47 / (sin el + 0.0121)."""
    return 2.47 / (np.sin(el_rad) + 0.0121)


def elevation_weights(el_rad, scheme=P.WEIGHTING):
    """Dimensionless weights. sin^2(el) means sigma_i = sigma_zenith / sin(el)."""
    if scheme == "equal":
        return np.ones_like(el_rad)
    return np.clip(np.sin(el_rad), 0.05, 1.0) ** 2


def doppler_to_range_rate(doppler_hz, wavelength=LAMBDA_L1):
    """Doppler is positive when the range shrinks, so rho_dot = -lambda * f_D."""
    return -wavelength * np.asarray(doppler_hz, dtype=float)


def rotate_z(vecs, theta):
    """Rotate row vectors about ECEF z by theta (per row). Earth-rotation fix."""
    c, s = np.cos(theta), np.sin(theta)
    x, y = vecs[:, 0], vecs[:, 1]
    return np.column_stack([c * x + s * y, -s * x + c * y, vecs[:, 2]])


def los_geometry(sat_pos, sat_vel, x, earth_rotation=P.APPLY_EARTH_ROTATION):
    """Satellite states in the receive-time ECEF frame, ranges and unit LOS e_i."""
    rng = np.linalg.norm(sat_pos - x, axis=1)
    if earth_rotation:
        theta = OMEGA_E * rng / C
        sat_pos = rotate_z(sat_pos, theta)
        sat_vel = rotate_z(sat_vel, theta)
        rng = np.linalg.norm(sat_pos - x, axis=1)
    e = (sat_pos - x) / rng[:, None]
    return sat_pos, sat_vel, rng, e


# ==========================================================================
# Least squares core
# ==========================================================================
def wls(H, z, w):
    """x = (H^T W H)^-1 H^T W z.  Returns x, Q = (H^T W H)^-1, residuals r."""
    HtW = H.T * w
    Q = np.linalg.inv(HtW @ H)
    x = Q @ (HtW @ z)
    return x, Q, z - H @ x


def dops(Q_enu):
    """DOPs from the 4x4 ENU cofactor matrix [E, N, U, clock]."""
    d = np.diag(Q_enu)
    return {
        "hdop": math.sqrt(d[0] + d[1]),
        "vdop": math.sqrt(d[2]),
        "pdop": math.sqrt(d[0] + d[1] + d[2]),
        "tdop": math.sqrt(d[3]),
        "gdop": math.sqrt(d.sum()),
    }


def ls_with_prior(H, z, r_inv, x_prior, P_prior):
    """
    argmin ||z - Hx||^2_(R^-1) + ||x - x_prior||^2_(P^-1)   (information form)

        x = (H^T R^-1 H + P^-1)^-1 (H^T R^-1 z + P^-1 x_prior)
        P = (H^T R^-1 H + P^-1)^-1

    r_inv is the diagonal of R^-1. This is exactly the Kalman update; it also
    works with fewer than 4 measurements because the prior fills the gap.
    """
    P_inv = np.linalg.inv(P_prior)
    HtRi = H.T * r_inv
    P_post = np.linalg.inv(HtRi @ H + P_inv)
    x_post = P_post @ (HtRi @ z + P_inv @ x_prior)
    return x_post, P_post


def kalman_update(H, z, R, x_prior, P_prior):
    """Textbook gain form, kept only to show it equals ls_with_prior."""
    S = H @ P_prior @ H.T + R
    K = P_prior @ H.T @ np.linalg.inv(S)
    x = x_prior + K @ (z - H @ x_prior)
    P_post = (np.eye(len(x_prior)) - K @ H) @ P_prior
    return x, P_post


# ==========================================================================
# Chi-square test (exact for integer dof, no scipy needed)
# ==========================================================================
def chi2_sf(x, k):
    """P(X > x) for X ~ chi2(k), integer k >= 1."""
    if x <= 0.0:
        return 1.0
    half = 0.5 * x
    if k % 2 == 0:
        term, total = 1.0, 1.0
        for j in range(1, k // 2):
            term *= half / j
            total += term
        return math.exp(-half) * total
    total, term = 0.0, 0.0
    for j in range(1, (k - 1) // 2 + 1):
        term = math.sqrt(x) if j == 1 else term * x / (2 * j - 1)
        total += term
    return math.erfc(math.sqrt(half)) + math.sqrt(2.0 / math.pi) * math.exp(-half) * total


def chi2_threshold(dof, p_fa=P.P_FA):
    """T such that P(chi2(dof) > T) = p_fa (bisection on chi2_sf)."""
    lo, hi = 0.0, dof + 10.0 * math.sqrt(2.0 * dof) + 50.0
    while chi2_sf(hi, dof) > p_fa:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if chi2_sf(mid, dof) > p_fa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class TestResult:
    stat: float = float("nan")       # r^T W r / sigma^2
    threshold: float = float("nan")
    dof: int = 0
    passed: bool | None = None       # None = no redundancy, cannot test
    excluded: list = field(default_factory=list)   # indices into the epoch


def fde(solve, mask, sigma, p_fa=P.P_FA, max_excl=P.MAX_EXCLUSIONS):
    """
    Fault detection and exclusion around any least-squares solve.

    solve(mask) must return an object with .H, .r, .w, .Q for the satellites
    in mask (in index order). While SSE = r^T W r / sigma^2 exceeds the
    chi2(n-4) threshold, drop the satellite with the largest standardised
    residual |r_i| / sqrt(diag(W^-1 - H Q H^T)) and solve again.
    """
    mask = mask.copy()
    test = TestResult()
    while True:
        sol = solve(mask)
        idx = np.flatnonzero(mask)
        dof = len(idx) - sol.H.shape[1]
        test.dof = dof
        if dof < 1:
            test.passed = None
            return sol, mask, test
        test.stat = float(np.sum(sol.w * sol.r ** 2) / sigma ** 2)
        test.threshold = chi2_threshold(dof, p_fa)
        if test.stat <= test.threshold:
            test.passed = True
            return sol, mask, test
        if dof < 2 or len(test.excluded) >= max_excl:
            test.passed = False          # fault seen, cannot isolate it
            return sol, mask, test
        c_r = 1.0 / sol.w - np.einsum("ij,jk,ik->i", sol.H, sol.Q, sol.H)
        std = np.abs(sol.r) / np.sqrt(np.clip(c_r, 1e-12, None))
        worst = int(idx[np.argmax(std)])
        mask[worst] = False
        test.excluded.append(worst)


# ==========================================================================
# 1. Position: iterated weighted least squares (Gauss-Newton)
# ==========================================================================
@dataclass
class PositionFix:
    x: np.ndarray        # ECEF (m)
    b: float             # receiver clock bias c*dt (m)
    e: np.ndarray        # (n,3) unit LOS, ECEF
    el: np.ndarray       # (n,) rad
    az: np.ndarray
    w: np.ndarray        # (n,) weights
    H: np.ndarray        # (n,4) rows [-e_i^T, 1]
    Q: np.ndarray        # (4,4) (H^T W H)^-1, ECEF
    r: np.ndarray        # (n,) residuals (m)
    iterations: int
    converged: bool


def solve_position(sat_pos, pr, x0=None, b0=0.0, weighting=P.WEIGHTING,
                   tropo=P.APPLY_TROPO, earth_rotation=P.APPLY_EARTH_ROTATION,
                   max_iter=15, tol=1e-4):
    """
    pr: pseudoranges already corrected for satellite clock (and iono if known).
    Linearise rho_i = ||s_i - x|| + b around the current guess:
        drho = rho_measured - rho_predicted = H dx,  H row = [-e_i^T, 1]
    and iterate dx = (H^T W H)^-1 H^T W drho until it stops moving.
    """
    x = np.zeros(3) if x0 is None else np.array(x0, dtype=float)
    b = float(b0)
    zeros = np.zeros_like(sat_pos)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        _, _, rng, e = los_geometry(sat_pos, zeros, x, earth_rotation)
        if np.linalg.norm(x) > 6.0e6:            # near the surface: el is meaningful
            el, _ = elev_az(x, e)
            w = elevation_weights(el, weighting)
            t = tropo_delay(el) if tropo else 0.0
        else:                                    # first steps from Earth centre
            w, t = np.ones(len(pr)), 0.0
        H = np.column_stack([-e, np.ones(len(pr))])
        dx, _, _ = wls(H, pr - t - (rng + b), w)
        x += dx[:3]
        b += dx[3]
        if np.linalg.norm(dx) < tol and np.linalg.norm(x) > 6.0e6:
            converged = True
            break

    _, _, rng, e = los_geometry(sat_pos, zeros, x, earth_rotation)
    el, az = elev_az(x, e)
    w = elevation_weights(el, weighting)
    t = tropo_delay(el) if tropo else 0.0
    H = np.column_stack([-e, np.ones(len(pr))])
    Q = np.linalg.inv((H.T * w) @ H)
    r = pr - t - (rng + b)
    return PositionFix(x, b, e, el, az, w, H, Q, r, it, converged)


# ==========================================================================
# 2. Velocity: Doppler least squares (linear, one shot)
# ==========================================================================
@dataclass
class VelocityFix:
    v: np.ndarray        # (3,) ENU m/s
    drift: float         # receiver clock drift c*dt_dot (m/s)
    H: np.ndarray        # (n,4) rows [-e_i_enu^T, 1]
    z: np.ndarray        # (n,) rho_dot - e_i^T v_sat (+ sat clock drift)
    w: np.ndarray
    Q: np.ndarray        # (4,4) ENU
    r: np.ndarray        # (n,) residuals (m/s)


def doppler_system(e_ecef, sat_vel, rr, R_enu, sat_clk_drift=None):
    """
    rho_dot_i = e_i^T (v_sat - v_u) + c*dt_dot_u - c*dt_dot_sat
    Move the known satellite terms left:
        z_i = rho_dot_i - e_i^T v_sat + c*dt_dot_sat = -e_i^T v_u + c*dt_dot_u
    Rows are [-e_i^T, 1] (rotated to ENU so v comes out in ENU directly).
    """
    z = rr - np.einsum("ij,ij->i", e_ecef, sat_vel)
    if sat_clk_drift is not None:
        z = z + sat_clk_drift
    H = np.column_stack([-(e_ecef @ R_enu.T), np.ones(len(rr))])
    return H, z


def solve_velocity(H, z, w):
    x, Q, r = wls(H, z, w)
    return VelocityFix(x[:3], float(x[3]), H, z, w, Q, r)


# ==========================================================================
# Velocity filter: least squares with an IMU prior (= Kalman filter)
# ==========================================================================
class VelocityFilter:
    """
    State [vE, vN, vU, c*dt_dot]. Each epoch:
      predict:  v-  = v + a_imu * dt          P- = P + diag((s_a dt)^2 x3, s_d^2 dt)
      update:   x   = argmin ||z - Hx||^2_W/sigma^2 + ||x - x-||^2_(P-^-1)
    With no IMU the prior is constant velocity with a bigger s_a.
    """

    def __init__(self, sigma_acc_imu=P.SIGMA_ACC_IMU,
                 sigma_acc_no_imu=P.SIGMA_ACC_NO_IMU, sigma_drift=P.SIGMA_DRIFT_RW):
        self.s_imu, self.s_free, self.s_d = sigma_acc_imu, sigma_acc_no_imu, sigma_drift
        self.x = None
        self.P = None

    @property
    def ready(self):
        return self.x is not None

    def predict(self, dt, accel_enu=None):
        if self.x is None or dt <= 0.0:
            return
        s_a = self.s_free
        if accel_enu is not None:
            self.x[:3] += np.asarray(accel_enu) * dt
            s_a = self.s_imu
        self.P = self.P + np.diag([(s_a * dt) ** 2] * 3 + [self.s_d ** 2 * dt])

    def update(self, H, z, w, sigma):
        r_inv = w / sigma ** 2
        if self.x is None:
            if len(z) < 4:
                return False
            self.P = np.linalg.inv((H.T * r_inv) @ H)
            self.x = self.P @ ((H.T * r_inv) @ z)
            return True
        if len(z) == 0:
            return True
        self.x, self.P = ls_with_prior(H, z, r_inv, self.x, self.P)
        return True


# ==========================================================================
# Epoch input / solution output
# ==========================================================================
@dataclass
class Epoch:
    t: float                          # s (GPS time of week, or any time tag)
    prn: list
    sat_pos: np.ndarray               # (n,3) ECEF at transmit time (m)
    sat_vel: np.ndarray               # (n,3) ECEF (m/s)
    pr: np.ndarray                    # (n,) raw pseudorange (m)
    rr: np.ndarray | None = None      # (n,) range rate (m/s) = -lambda * Doppler
    sat_clk: np.ndarray | None = None        # (n,) c*dt_sat (m), added to pr
    sat_clk_drift: np.ndarray | None = None  # (n,) c*dt_dot_sat (m/s)
    iono: np.ndarray | None = None           # (n,) iono delay (m), subtracted
    accel_enu: np.ndarray | None = None      # (3,) IMU accel, gravity removed (m/s^2)


@dataclass
class NavSolution:
    t: float
    ok: bool = False                  # a position fix exists
    integrity: bool = False           # chi2 passed with >= MIN_DOF_INTEGRITY dof left
    p_enu: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    v_enu: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    v_enu_raw: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    p_ecef: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    lat: float = float("nan")
    lon: float = float("nan")
    h: float = float("nan")
    clock_bias: float = float("nan")  # m
    clock_drift: float = float("nan") # m/s
    hdop: float = float("nan")        # weighted (matches the weights used)
    vdop: float = float("nan")
    pdop: float = float("nan")
    hdop_geom: float = float("nan")   # classic equal-weight HDOP, for reference
    sigma_h: float = float("nan")     # m, HDOP * sigma_UERE (drms)
    sigma_major: float = float("nan") # m, 1-sigma semi-major axis
    sigma_v_h: float = float("nan")   # m/s, of v_enu
    n_used: int = 0
    used: list = field(default_factory=list)
    excluded: list = field(default_factory=list)
    pos_test: TestResult = field(default_factory=TestResult)
    vel_test: TestResult = field(default_factory=TestResult)
    Q_enu: np.ndarray | None = None

    def line(self) -> str:
        if not self.ok:
            return f"t={self.t:9.1f}  NO FIX"
        flag = "ok " if self.integrity else "NOT VERIFIED"
        ex = ",".join(self.excluded) if self.excluded else "-"
        return (f"t={self.t:9.1f}  n={self.n_used:2d}  HDOP={self.hdop:4.2f}"
                f"  sig_h={self.sigma_h:5.1f} m"
                f"  v=({self.v_enu[0]:5.2f},{self.v_enu[1]:5.2f}) m/s"
                f"  chi2={self.pos_test.stat:6.1f}/{self.pos_test.threshold:5.1f} {flag}"
                f"  excl={ex}")


# ==========================================================================
# The per-epoch pipeline
# ==========================================================================
class SPP:
    """Stateful per-epoch solver: warm start, velocity filter, outputs p, v, sigma_h."""

    def __init__(self, frame: LocalFrame | None = None, use_filter=P.USE_VELOCITY_FILTER):
        self.frame = frame or LocalFrame()
        self.x_last = None
        self.b_last = 0.0
        self.t_last = None
        self.vf = VelocityFilter() if use_filter else None

    def step(self, ep: Epoch) -> NavSolution:
        sol = NavSolution(t=ep.t)
        n = len(ep.pr)
        pr = np.asarray(ep.pr, dtype=float).copy()
        if ep.sat_clk is not None:
            pr += ep.sat_clk
        if ep.iono is not None:
            pr -= ep.iono
        mask = np.isfinite(pr)

        # Elevation mask from a warm start (last fix, else the local origin).
        x0 = self.x_last if self.x_last is not None else self.frame.ecef0
        _, _, _, e0 = los_geometry(ep.sat_pos, np.zeros((n, 3)), x0)
        el0, _ = elev_az(x0, e0)
        mask &= el0 >= math.radians(P.ELEV_MASK_DEG)

        dt = None if self.t_last is None else ep.t - self.t_last
        self.t_last = ep.t
        if self.vf is not None and dt is not None:
            self.vf.predict(dt, ep.accel_enu)

        if mask.sum() < 4:
            self._coast(sol)
            return sol

        # ---- 1 + 3. position WLS inside the chi2 FDE loop -----------------
        def pos_solve(m):
            return solve_position(ep.sat_pos[m], pr[m], x0=x0, b0=self.b_last)

        fix, mask, ptest = fde(pos_solve, mask, P.SIGMA_UERE)
        self.x_last, self.b_last = fix.x, fix.b
        idx = np.flatnonzero(mask)

        R = self.frame.R
        H_enu = np.column_stack([-(fix.e @ R.T), np.ones(len(idx))])
        Q_enu = np.linalg.inv((H_enu.T * fix.w) @ H_enu)
        d = dops(Q_enu)
        Hg = H_enu
        Qg = np.linalg.inv(Hg.T @ Hg)

        sol.ok = True
        sol.integrity = ptest.passed is True and ptest.dof >= P.MIN_DOF_INTEGRITY
        sol.pos_test = ptest
        sol.p_ecef = fix.x
        sol.p_enu = self.frame.to_enu(fix.x)
        sol.lat, sol.lon, sol.h = ecef_to_geodetic(fix.x)
        sol.clock_bias = fix.b
        sol.hdop, sol.vdop, sol.pdop = d["hdop"], d["vdop"], d["pdop"]
        sol.hdop_geom = math.sqrt(Qg[0, 0] + Qg[1, 1])
        sol.sigma_h = sol.hdop * P.SIGMA_UERE
        sol.sigma_major = P.SIGMA_UERE * math.sqrt(np.linalg.eigvalsh(Q_enu[:2, :2])[-1])
        if not sol.integrity:              # fault not isolated, or too little redundancy
            sol.sigma_h = max(sol.sigma_h, P.SIGMA_H_FALLBACK)
            sol.sigma_major = max(sol.sigma_major, P.SIGMA_H_FALLBACK)
        sol.Q_enu = Q_enu
        sol.n_used = len(idx)
        sol.used = [ep.prn[i] for i in idx]
        sol.excluded = [ep.prn[i] for i in ptest.excluded]

        # ---- 2 + 3. Doppler WLS, same H, own chi2 test ----------------------
        if ep.rr is None:
            self._coast(sol, keep_ok=True)
            return sol
        rr = np.asarray(ep.rr, dtype=float)
        s_rot, v_rot, _, e = los_geometry(ep.sat_pos[idx], ep.sat_vel[idx], fix.x)
        clk_dr = None if ep.sat_clk_drift is None else ep.sat_clk_drift[idx]
        Hv, zv = doppler_system(e, v_rot, rr[idx], R, clk_dr)
        vmask = np.isfinite(zv)
        wv = fix.w

        if vmask.sum() >= 4:
            def vel_solve(m):
                return solve_velocity(Hv[m], zv[m], wv[m])

            vfix, vmask, vtest = fde(vel_solve, vmask, P.SIGMA_RR)
            sol.vel_test = vtest
            sol.excluded += [ep.prn[idx[i]] for i in vtest.excluded]
            sol.v_enu_raw = vfix.v
            sol.clock_drift = vfix.drift
            sol.v_enu = vfix.v
            sol.sigma_v_h = P.SIGMA_RR * math.sqrt(vfix.Q[0, 0] + vfix.Q[1, 1])

        if self.vf is not None:
            m = vmask
            self.vf.update(Hv[m], zv[m], wv[m], P.SIGMA_RR)
            if self.vf.ready:
                sol.v_enu = self.vf.x[:3].copy()
                sol.clock_drift = float(self.vf.x[3])
                sol.sigma_v_h = math.sqrt(self.vf.P[0, 0] + self.vf.P[1, 1])
        return sol

    def _coast(self, sol, keep_ok=False):
        """No (usable) Doppler this epoch: report the filter prediction, if any."""
        if self.vf is not None and self.vf.ready:
            sol.v_enu = self.vf.x[:3].copy()
            sol.sigma_v_h = math.sqrt(self.vf.P[0, 0] + self.vf.P[1, 1])
        if not keep_ok:
            sol.ok = False


# ==========================================================================
# CSV input (one row per satellite per epoch)
# ==========================================================================
CSV_COLUMNS = ("t", "prn", "x", "y", "z", "vx", "vy", "vz", "clk_m",
               "clk_drift_mps", "pr_m", "rr_mps", "doppler_hz", "iono_m",
               "acc_e", "acc_n", "acc_u")


def load_csv(path) -> list[Epoch]:
    """
    Required columns: t, prn, x, y, z, pr_m.
    Optional: vx, vy, vz (needed for velocity), clk_m, clk_drift_mps, iono_m,
    rr_mps or doppler_hz (rr = -lambda_L1 * doppler), acc_e/acc_n/acc_u (IMU).
    """
    groups: dict[float, list[dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            groups.setdefault(round(float(row["t"]), 3), []).append(row)

    def col(rows, name):
        vals = [r.get(name, "") for r in rows]
        if all(v in ("", None) for v in vals):
            return None
        return np.array([float(v) if v not in ("", None) else np.nan for v in vals])

    epochs = []
    for t in sorted(groups):
        rows = groups[t]
        n = len(rows)
        vel = [col(rows, k) for k in ("vx", "vy", "vz")]
        sat_vel = np.column_stack(vel) if all(v is not None for v in vel) else np.zeros((n, 3))
        rr = col(rows, "rr_mps")
        if rr is None:
            dop = col(rows, "doppler_hz")
            rr = None if dop is None else doppler_to_range_rate(dop)
        if all(v is None for v in vel):
            rr = None
        acc = [col(rows, k) for k in ("acc_e", "acc_n", "acc_u")]
        accel = None if any(a is None for a in acc) else np.array([a[0] for a in acc])
        epochs.append(Epoch(
            t=t,
            prn=[r["prn"] for r in rows],
            sat_pos=np.column_stack([col(rows, k) for k in ("x", "y", "z")]),
            sat_vel=sat_vel,
            pr=col(rows, "pr_m"),
            rr=rr,
            sat_clk=col(rows, "clk_m"),
            sat_clk_drift=col(rows, "clk_drift_mps"),
            iono=col(rows, "iono_m"),
            accel_enu=accel,
        ))
    return epochs


def run(epochs, frame=None, use_filter=P.USE_VELOCITY_FILTER) -> list[NavSolution]:
    solver = SPP(frame, use_filter)
    return [solver.step(ep) for ep in epochs]


# ==========================================================================
# Self-test on synthetic data (python spp.py [file.csv])
# ==========================================================================
def _report(name, sols, truths):
    ok = [i for i, s in enumerate(sols) if s.ok]
    pe = np.array([sols[i].p_enu[:2] - truths[i]["p_enu"][:2] for i in ok])
    ve = np.array([sols[i].v_enu_raw[:2] - truths[i]["v_enu"][:2] for i in ok
                   if np.all(np.isfinite(sols[i].v_enu_raw))])
    vf = np.array([sols[i].v_enu[:2] - truths[i]["v_enu"][:2] for i in ok])
    hd = np.array([sols[i].hdop for i in ok])
    sh = np.array([sols[i].sigma_h for i in ok])
    excl = sum(len(sols[i].excluded) for i in ok)
    fails = sum(1 for i in ok if not sols[i].integrity)
    # the number that matters for the fence: true error outside k * sigma_h
    misleading = int(np.sum(np.linalg.norm(pe, axis=1) > P.K_SIGMA * sh)) if len(pe) else 0
    rms = lambda a: float(np.sqrt(np.mean(np.sum(a ** 2, axis=1)))) if len(a) else float("nan")
    print(name)
    print(f"  fixes {len(ok)}/{len(sols)}, sats used {np.mean([sols[i].n_used for i in ok]):.1f},"
          f" HDOP {hd.mean():.2f} (max {hd.max():.2f})")
    print(f"  horiz pos error rms {rms(pe):6.2f} m   mean predicted sigma_h {sh.mean():6.2f} m")
    print(f"  horiz vel error rms {rms(ve):6.3f} m/s raw Doppler LS -> {rms(vf):6.3f} m/s filtered")
    print(f"  chi2: {excl} exclusions, {fails} epochs not verified (sigma_h -> fallback),"
          f" {misleading} epochs with error > k*sigma_h")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for s in run(load_csv(sys.argv[1])):
            print(s.line())
        sys.exit()

    from sim import Canyon, Sim

    frame = LocalFrame()
    heading = math.radians(30.0)
    cruise = 8.0 * np.array([math.sin(heading), math.cos(heading), 0.0])

    def scenario(name, canyon=None, fault=None, epochs=180):
        sim = Sim(frame, canyon=canyon, seed=3)
        if fault is not None:                       # bias one satellite in view
            prn = sim.measure()[0].prn[2]
            sim.faults = [(prn, sim.t + 30, sim.t + 120, *fault)]
            name += f" on {prn}"
        solver = SPP(frame)
        sols, truths = [], []
        for _ in range(epochs):
            ep, truth = sim.step(1.0, cruise)
            sols.append(solver.step(ep))
            truths.append(truth)
        _report(name, sols, truths)
        print()

    scenario("1. open sky")
    scenario("2. open sky, +60 m / +2 m/s fault for 90 s", fault=(60.0, 2.0))
    scenario("3. urban canyon: 30 m buildings, 30 m street, NLOS reflections",
             canyon=Canyon())

    # Least squares with a prior == Kalman update
    rng = np.random.default_rng(0)
    H = rng.normal(size=(6, 4))
    z = rng.normal(size=6)
    Rd = rng.uniform(0.1, 1.0, 6)
    xp = rng.normal(size=4)
    A = rng.normal(size=(4, 4))
    Pp = A @ A.T + np.eye(4)
    x1, P1 = ls_with_prior(H, z, 1.0 / Rd, xp, Pp)
    x2, P2 = kalman_update(H, z, np.diag(Rd), xp, Pp)
    print(f"LS-with-prior vs Kalman: max |dx| = {np.abs(x1 - x2).max():.1e},"
          f" max |dP| = {np.abs(P1 - P2).max():.1e}")
    print(f"chi2 thresholds (P_fa={P.P_FA:g}): "
          + ", ".join(f"dof{k}={chi2_threshold(k):.2f}" for k in (1, 2, 4, 6)))
