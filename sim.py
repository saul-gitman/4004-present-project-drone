"""
sim.py - the real world, simulated. It only makes measurements; it is test
data, not course content.

    satellites  GPS-like constellation: 6 planes x 5, 55 deg, 26 560 km
    GNSS epoch  pseudorange = range + receiver clock + ionosphere + noise
                range rate  = e . (v_sat - v) + receiver clock drift + noise
                (satellite clocks already corrected, Earth rotation ignored)
    sky         buildings hide every satellite below params.SKY_MASK_DEG
    IMU         acceleration in the body frame, and heading, with noise
    radar       range and bearing (sensor frame) to the nearest point of
                every building within params.SENSOR_RANGE
    aircraft    follows a velocity command: speed changes at most A_BRAKE,
                turning radius R_TURN (tighter only when slow)
"""

from __future__ import annotations

import math

import numpy as np

import params as P
from boundary import closest_point
from gnss_lib_py_subset import _calculate_iono_delay as klobuchar
from gnss_lib_py_subset import ecef_to_el_az
from location import FRAME, rot

# --------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------
T0 = 4 * 86400 + 11760.0          # GPS time of week: Thursday 11:16 in Hong Kong
                                  # (8 satellites above 15 deg, 5 above 40, 1 above 45)

# Klobuchar parameters as broadcast in a GPS navigation message (typical values)
IONO_PARAMS = {"gps": np.array([[1.1176e-08, -7.4506e-09, -5.9605e-08, 1.1921e-07],    # alpha
                                [1.1674e+05, -2.2938e+05, -1.3107e+05, 1.0486e+06]])}  # beta
IONO_TRUE_SCALE = 1.5             # the real ionosphere is 1.5 x what the broadcast model says

# One obstacle, taller than the flight altitude. EXAMPLE, north/east metres.
BUILDINGS = {"EXAMPLE tower": np.array([[150.0, -190.0], [150.0, -130.0],
                                        [200.0, -130.0], [200.0, -190.0]])}

# --------------------------------------------------------------------------
# Satellites
# --------------------------------------------------------------------------
GM = 3.986004418e14
A_ORBIT = 26_559_700.0
INC = math.radians(55.0)
N_MOTION = math.sqrt(GM / A_ORBIT ** 3)
OMEGA_E = 7.2921151467e-5
PRNS = [f"G{k:02d}" for k in range(1, 31)]


def _sat_pos(t):
    pos = []
    for plane in range(6):
        raan = math.radians(60.0 * plane + 17.0)
        for slot in range(5):
            u = N_MOTION * t + math.radians(72.0 * slot + 15.0 * plane)
            xo, yo = A_ORBIT * math.cos(u), A_ORBIT * math.sin(u)
            x1, y1, z1 = xo, yo * math.cos(INC), yo * math.sin(INC)
            pos.append([x1 * math.cos(raan) - y1 * math.sin(raan),
                        x1 * math.sin(raan) + y1 * math.cos(raan), z1])
    eci = np.array(pos)
    c, s = math.cos(OMEGA_E * t), math.sin(OMEGA_E * t)          # inertial -> ECEF
    return np.column_stack([c * eci[:, 0] + s * eci[:, 1], -s * eci[:, 0] + c * eci[:, 1], eci[:, 2]])


def satellites(t):
    """ECEF positions and velocities (30, 3) at GPS time t."""
    return _sat_pos(t), (_sat_pos(t + 0.5) - _sat_pos(t - 0.5))


# --------------------------------------------------------------------------
# Aircraft and its sensors
# --------------------------------------------------------------------------
class World:
    def __init__(self, start=(0.0, 0.0), heading_deg=0.0, seed=1):
        self.rng = np.random.default_rng(seed)
        self.p = np.array(start, dtype=float)          # north, east (m)
        self.speed = 0.0                               # m/s
        self.heading = math.radians(heading_deg)       # rad, clockwise from north
        self.t = T0
        self.clock_bias, self.clock_drift = 15_000.0, 80.0   # receiver clock: m, m/s
        self.acc = np.zeros(2)                         # mean NED acceleration, last step
        self.track = [self.p.copy()]                   # positions during the last step

    @property
    def v(self):
        return self.speed * np.array([math.cos(self.heading), math.sin(self.heading)])

    def fly(self, v_cmd, dt, h=0.05):
        """Follow the commanded velocity (north, east) for dt seconds."""
        v0 = self.v
        s_cmd = float(np.hypot(*v_cmd))
        hdg_cmd = math.atan2(v_cmd[1], v_cmd[0]) if s_cmd > 0.1 else self.heading
        self.track = []
        for _ in range(max(1, round(dt / h))):
            err = (hdg_cmd - self.heading + math.pi) % (2.0 * math.pi) - math.pi
            target = 0.0 if abs(err) > math.pi / 2 else s_cmd          # sharp turn: stop first
            self.speed += min(max(target - self.speed, -P.A_BRAKE * h), P.A_BRAKE * h)
            w_max = max(P.A_BRAKE / max(self.speed, 0.1), self.speed / P.R_TURN)
            self.heading += min(max(err, -w_max * h), w_max * h)       # radius >= min(R, v^2/a)
            self.p = self.p + self.v * h
            self.track.append(self.p.copy())
        self.acc = (self.v - v0) / dt
        self.t += dt
        self.clock_bias += self.clock_drift * dt

    def gnss(self):
        """One GNSS epoch: what the receiver gets from the satellites it can see."""
        x = FRAME.ned_to_ecef(np.array([[self.p[0], self.p[1], -P.FLIGHT_ALT]]))[0]
        v = FRAME.ned_to_ecefv(np.array([[self.v[0], self.v[1], 0.0]]))[0]
        sv_pos, sv_vel = satellites(self.t)
        el, az = ecef_to_el_az(x.reshape(3, 1), sv_pos.T)
        seen = el > max(P.SKY_MASK_DEG, 0.0)
        los = sv_pos - x
        rng = np.linalg.norm(los, axis=1)
        e = los / rng[:, None]
        iono = IONO_TRUE_SCALE * klobuchar(self.t, IONO_PARAMS, x, sv_pos.T)
        n = len(PRNS)
        pr = rng + self.clock_bias + iono + self.rng.normal(size=n) * P.SIGMA_UERE
        rr = (np.einsum("ij,ij->i", e, sv_vel - v) + self.clock_drift
              + self.rng.normal(size=n) * P.SIGMA_RR)
        return {"t": self.t, "prn": [p for p, s in zip(PRNS, seen) if s],
                "sv_pos": sv_pos[seen], "sv_vel": sv_vel[seen], "pr": pr[seen], "rr": rr[seen],
                "iono_params": IONO_PARAMS,
                "sky": [(PRNS[k], float(el[k]), float(az[k]), bool(seen[k]))
                        for k in np.flatnonzero(el > 0)]}

    def imu(self):
        """Mean acceleration over the last step in the body frame, and heading."""
        acc_body = rot(self.heading).T @ self.acc + self.rng.normal(size=2) * P.SIGMA_ACC_IMU
        heading = self.heading + math.radians(P.SIGMA_YAW_DEG) * self.rng.normal()
        return {"acc_body": acc_body, "heading": heading}

    def radar(self):
        """Magic radar: range and bearing (sensor frame) to every building in range."""
        out = []
        mount = math.radians(P.SENSOR_MOUNT_DEG)
        for poly in BUILDINGS.values():
            q, inside = closest_point(poly, self.p)
            d = 0.0 if inside else float(np.hypot(*(q - self.p)))
            if d > P.SENSOR_RANGE:
                continue
            bearing_ned = math.atan2(q[1] - self.p[1], q[0] - self.p[0])
            out.append({"range": max(0.0, d + P.SIGMA_RANGE * self.rng.normal()),
                        "bearing": bearing_ned - self.heading - mount
                        + math.radians(P.SIGMA_BEARING_DEG) * self.rng.normal()})
        return out

    def clearance(self, zones):
        """True distances (m) to the nearest zone edge (negative inside) and building."""
        dz, db = math.inf, math.inf
        for z in zones:
            q, inside = closest_point(z.ne, self.p)
            dz = min(dz, -float(np.hypot(*(q - self.p))) if inside else float(np.hypot(*(q - self.p))))
        for poly in BUILDINGS.values():
            q, inside = closest_point(poly, self.p)
            db = min(db, 0.0 if inside else float(np.hypot(*(q - self.p))))
        return dz, db
