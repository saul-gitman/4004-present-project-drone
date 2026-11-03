"""
location.py - where am I, how fast am I going, and where are the obstacles?

Two flows, because the two kinds of threat are measured differently.

FLOW 1  GNSS + IMU   absolute position, compared with the eSUA map (zones)
    pseudorange - Klobuchar iono   -> gnss_lib_py wls()   -> p (ECEF -> NED), clock bias
    rows H = [-e_i, 1] at the fix  -> Q = (H^T H)^-1      -> HDOP, sigma_h = HDOP * SIGMA_UERE
    Doppler range rate, same H     -> least squares       -> v_gnss, clock drift
    IMU acceleration, body -> NED  -> v_imu = v + a * dt
    weighted:  v = w * v_gnss + (1 - w) * v_imu,   w = s_imu^2 / (s_imu^2 + s_gnss^2)

FLOW 2  radar        relative position, no GNSS needed (buildings)
    (range, bearing) in the sensor frame -> body frame (mount angle) -> NED (heading)

Frames: ECEF for satellites; NED metres about params.ORIGIN_* for the map,
zones and aircraft; body = (forward, right) for the IMU and the radar. The
flight is level, so everything after the GNSS fix uses (north, east) only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import params as P
from gnss_lib_py_subset import LocalCoord, ecef_to_el_az, wls
from gnss_lib_py_subset import _calculate_iono_delay as klobuchar

FRAME = LocalCoord.from_geodetic(np.array([[P.ORIGIN_LAT, P.ORIGIN_LON, P.ORIGIN_H]]))


def rot(angle):
    """2-D rotation by `angle` (rad, clockwise seen from above).

    rot(heading):    body (forward, right) -> NED (north, east)
    rot(mount):      sensor frame -> body frame
    rot(...).T:      the way back
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


# ==========================================================================
# Flow 1: GNSS position (pseudorange) and velocity (Doppler)
# ==========================================================================
def gnss_fix(ep, x0):
    """
    Single point positioning for one epoch. x0 = (4,1) [ECEF x, y, z; clock bias]
    is the starting guess (last fix). Returns None if fewer than 4 satellites
    are above the elevation mask.
    """
    el, _ = ecef_to_el_az(x0[:3], ep["sv_pos"].T)
    use = el >= P.ELEV_MASK_DEG
    if use.sum() < 4:
        return None
    sv_pos = ep["sv_pos"][use]
    iono = klobuchar(ep["t"], ep["iono_params"], x0[:3], sv_pos.T)      # m, one per satellite
    x = wls(x0, sv_pos, (ep["pr"][use] - iono).reshape(-1, 1), sv_rx_time=True)

    # Geometry at the fix: one row [-e_i, 1] per satellite, rotated into NED.
    los = sv_pos - x[:3, 0]
    e = los / np.linalg.norm(los, axis=1, keepdims=True)                 # receiver -> satellite
    H = np.column_stack([-e @ FRAME.ecef_to_ned_matrix.T, np.ones(len(e))])
    Q = np.linalg.inv(H.T @ H)
    return {"x": x, "p": FRAME.ecef_to_ned(x[:3].T)[0, :2],
            "hdop": math.sqrt(Q[0, 0] + Q[1, 1]), "e": e, "use": use, "iono": iono}


def doppler_velocity(ep, fix):
    """
    Velocity from Doppler, with the same rows H = [-e_i, 1] as the position:
        rr_i = e_i . (v_sat_i - v) + c*dt_dot
        z_i  = rr_i - e_i . v_sat_i  =  -e_i . v + c*dt_dot
    Linear, so one solve: [v; c*dt_dot] = (H^T H)^-1 H^T z.
    Returns v (north, east) in m/s.
    """
    e = fix["e"]
    z = ep["rr"][fix["use"]] - np.einsum("ij,ij->i", e, ep["sv_vel"][fix["use"]])
    H = np.column_stack([-e, np.ones(len(e))])
    sol = np.linalg.solve(H.T @ H, H.T @ z)
    return FRAME.ecef_to_nedv(sol[:3].reshape(1, 3))[0, :2]


@dataclass
class NavState:
    t: float
    fix: bool              # a GNSS position this epoch
    p: np.ndarray | None   # (2,) north, east (m); None before the first fix
    v: np.ndarray          # (2,) weighted GNSS + IMU velocity (m/s)
    a: np.ndarray          # (2,) IMU acceleration in NED (m/s^2)
    heading: float         # rad, from the IMU
    sigma_h: float         # m, horizontal position sigma
    hdop: float
    n_sats: int            # satellites used
    w_gnss: float          # weight on the GNSS velocity (0 = IMU only)
    v_gnss: np.ndarray | None
    v_imu: np.ndarray | None
    iono_mean: float       # m, mean Klobuchar correction
    used: list


class Navigator:
    """Flow 1 with memory: warm start for the fix, last velocity for the IMU."""

    def __init__(self):
        self.x = np.vstack([FRAME.init_ecef.reshape(3, 1), [[0.0]]])
        self.p = None
        self.v = None
        self.sigma_fix = math.inf
        self.t_out = 0.0                  # s since the last GNSS fix

    def update(self, ep, imu, dt) -> NavState:
        a = rot(imu["heading"]) @ imu["acc_body"]                 # body -> NED
        v_imu = None if self.v is None else self.v + a * dt
        fix = gnss_fix(ep, self.x)

        if fix is not None:
            self.x, self.p, self.t_out = fix["x"], fix["p"], 0.0
            self.sigma_fix = fix["hdop"] * P.SIGMA_UERE
            sigma_h = self.sigma_fix
            v_gnss = doppler_velocity(ep, fix)
            if v_imu is None:
                w, self.v = 1.0, v_gnss
            else:
                s_gnss = fix["hdop"] * P.SIGMA_RR             # same DOP as the position
                s_imu = P.SIGMA_ACC_IMU * dt
                w = s_imu ** 2 / (s_imu ** 2 + s_gnss ** 2)
                self.v = w * v_gnss + (1.0 - w) * v_imu
        else:                                                 # outage: IMU dead reckoning
            v_gnss, w = None, 0.0
            if v_imu is not None:
                self.v = v_imu
            if self.p is not None:
                self.p = self.p + self.v * dt
                self.t_out += dt
            sigma_h = self.sigma_fix + 0.5 * P.SIGMA_ACC_IMU * self.t_out ** 2

        return NavState(
            t=ep["t"], fix=fix is not None, p=self.p,
            v=np.zeros(2) if self.v is None else self.v, a=a, heading=imu["heading"],
            sigma_h=sigma_h, hdop=fix["hdop"] if fix else math.nan,
            n_sats=int(fix["use"].sum()) if fix else 0, w_gnss=w, v_gnss=v_gnss, v_imu=v_imu,
            iono_mean=float(fix["iono"].mean()) if fix else math.nan,
            used=[prn for prn, u in zip(ep["prn"], fix["use"]) if u] if fix else [])


# ==========================================================================
# Flow 2: radar returns -> obstacle range and direction in NED
# ==========================================================================
@dataclass
class Obstacle:
    range: float           # m, as measured
    u: np.ndarray          # (2,) unit vector aircraft -> obstacle, NED


def radar_to_ned(returns, heading):
    """Each return: sensor frame -> body frame (mount) -> NED (heading)."""
    R_bs = rot(math.radians(P.SENSOR_MOUNT_DEG))      # sensor -> body
    R_nb = rot(heading)                               # body -> NED
    out = []
    for r in returns:
        u_sensor = np.array([math.cos(r["bearing"]), math.sin(r["bearing"])])
        out.append(Obstacle(r["range"], R_nb @ R_bs @ u_sensor))
    return out
