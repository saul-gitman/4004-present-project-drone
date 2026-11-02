"""
sim.py - synthetic GNSS epochs with known truth, for testing spp.py and
boundary.py before (or alongside) the UrbanNav data.

Not course content: it only manufactures measurements. It models
  * a GPS-like constellation (6 planes x 5 satellites, 55 deg, 26 560 km),
  * light time and Earth rotation, satellite clocks, troposphere,
  * pseudorange / range-rate noise with sigma / sin(el),
  * an optional street canyon that blocks satellites, a fraction of which
    still arrive by reflection (NLOS) with a positive range bias,
  * a drone that tracks a commanded ENU velocity (first-order lag) and an IMU
    that reports its mean acceleration over each epoch (delta-v / dt) with noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import params as P
from spp import (C, OMEGA_E, Epoch, LocalFrame, ecef_to_geodetic, enu_rotation,
                 rotate_z, tropo_delay)

GM = 3.986004418e14
A_ORBIT = 26_559_700.0
INC = math.radians(55.0)
N_MOTION = math.sqrt(GM / A_ORBIT ** 3)
N_PLANES, N_SLOTS = 6, 5
PRNS = [f"G{k:02d}" for k in range(1, N_PLANES * N_SLOTS + 1)]


def constellation_ecef(t):
    """ECEF positions (30,3) of a GPS-like Walker constellation at time t."""
    pos = []
    for plane in range(N_PLANES):
        raan = math.radians(360.0 / N_PLANES * plane + 17.0)
        for slot in range(N_SLOTS):
            u = N_MOTION * t + math.radians(360.0 / N_SLOTS * slot + 15.0 * plane)
            xo, yo = A_ORBIT * math.cos(u), A_ORBIT * math.sin(u)
            x1, y1, z1 = xo, yo * math.cos(INC), yo * math.sin(INC)
            pos.append([x1 * math.cos(raan) - y1 * math.sin(raan),
                        x1 * math.sin(raan) + y1 * math.cos(raan), z1])
    eci = np.array(pos)
    return rotate_z(eci, np.full(len(eci), OMEGA_E * t))


def constellation_vel(t, h=0.5):
    return (constellation_ecef(t + h) - constellation_ecef(t - h)) / (2.0 * h)


@dataclass
class Canyon:
    """Straight street; a satellite is blocked below atan(H |sin(az - street)| / w)."""
    street_az_deg: float = 30.0
    height_m: float = 30.0
    half_width_m: float = 15.0
    nlos_fraction: float = 0.4
    nlos_pr_bias_m: tuple = (20.0, 80.0)
    nlos_rr_bias_mps: float = 1.5

    def mask_el(self, az_rad):
        s = np.abs(np.sin(az_rad - math.radians(self.street_az_deg)))
        return np.arctan2(self.height_m * s, self.half_width_m)


class Sim:
    def __init__(self, frame: LocalFrame | None = None, p0_enu=(0.0, 0.0, 30.0),
                 v0_enu=(0.0, 0.0, 0.0), t0=345_600.0, seed=1, canyon: Canyon | None = None,
                 sigma_pr=P.SIGMA_UERE, sigma_rr=P.SIGMA_RR, tau_v=0.8, sigma_imu=0.15,
                 clock_bias=15_000.0, clock_drift=80.0, faults=()):
        self.frame = frame or LocalFrame()
        self.p = np.array(p0_enu, dtype=float)
        self.v = np.array(v0_enu, dtype=float)
        self.a = np.zeros(3)
        self.t = t0
        self.rng = np.random.default_rng(seed)
        self.canyon = canyon
        self.sigma_pr, self.sigma_rr = sigma_pr, sigma_rr
        self.tau_v, self.sigma_imu = tau_v, sigma_imu
        self.b, self.d = clock_bias, clock_drift
        self.faults = list(faults)      # (prn, t_from, t_to, pr_bias_m, rr_bias_mps)
        n = len(PRNS)
        self.sat_clk = self.rng.uniform(-3000.0, 3000.0, n)      # c*dt_sat (m)
        self.sat_clk_drift = self.rng.uniform(-0.01, 0.01, n)    # m/s
        self.is_nlos_sat = self.rng.uniform(size=n) < (canyon.nlos_fraction if canyon else 0)
        lo, hi = canyon.nlos_pr_bias_m if canyon else (0.0, 0.0)
        self.nlos_pr = self.rng.uniform(lo, hi, n)
        self.nlos_rr = self.rng.choice([-1.0, 1.0], n) * (canyon.nlos_rr_bias_mps if canyon else 0)

    def step(self, dt, v_cmd_enu):
        """Fly dt seconds towards the commanded velocity, then measure."""
        v_cmd = np.asarray(v_cmd_enu, dtype=float)
        h = 0.01
        v_start = self.v.copy()
        for _ in range(max(1, int(round(dt / h)))):
            self.v = self.v + (v_cmd - self.v) / self.tau_v * h
            self.p = self.p + self.v * h
        self.a = (self.v - v_start) / dt     # mean accel over the epoch (IMU delta-v / dt)
        self.t += dt
        self.b += self.d * dt
        return self.measure()

    def measure(self):
        x = self.frame.to_ecef(self.p)
        v_ecef = self.frame.R.T @ self.v

        # light-time iteration: satellite at transmit time, expressed in the
        # ECEF frame of the receive time
        tau = np.full(len(PRNS), 0.075)
        for _ in range(3):
            s_tx = np.array([constellation_ecef(self.t - ti)[k] for k, ti in enumerate(tau)])
            s_rot = rotate_z(s_tx, OMEGA_E * tau)
            tau = np.linalg.norm(s_rot - x, axis=1) / C
        v_tx = np.array([constellation_vel(self.t - ti)[k] for k, ti in enumerate(tau)])
        v_rot = rotate_z(v_tx, OMEGA_E * tau)
        rng = np.linalg.norm(s_rot - x, axis=1)
        e = (s_rot - x) / rng[:, None]

        lat, lon, _ = ecef_to_geodetic(x)
        e_enu = e @ enu_rotation(lat, lon).T
        el = np.arcsin(e_enu[:, 2])
        az = np.mod(np.arctan2(e_enu[:, 0], e_enu[:, 1]), 2 * np.pi)

        visible = el > math.radians(2.0)
        nlos = np.zeros(len(PRNS), dtype=bool)
        if self.canyon is not None:
            blocked = el < self.canyon.mask_el(az)
            nlos = blocked & self.is_nlos_sat
            visible &= ~blocked | nlos

        sin_el = np.clip(np.sin(el), 0.05, 1.0)
        pr = (rng + self.b - self.sat_clk + tropo_delay(el)
              + self.rng.normal(size=len(PRNS)) * self.sigma_pr / sin_el
              + np.where(nlos, self.nlos_pr, 0.0))
        rr = (np.einsum("ij,ij->i", e, v_rot - v_ecef) + self.d - self.sat_clk_drift
              + self.rng.normal(size=len(PRNS)) * self.sigma_rr / sin_el
              + np.where(nlos, self.nlos_rr, 0.0))
        for prn, t_from, t_to, pr_bias, rr_bias in self.faults:
            if t_from <= self.t <= t_to and prn in PRNS:
                k = PRNS.index(prn)
                pr[k] += pr_bias
                rr[k] += rr_bias

        idx = np.flatnonzero(visible)
        imu = self.a + self.rng.normal(size=3) * self.sigma_imu
        ep = Epoch(t=self.t, prn=[PRNS[k] for k in idx], sat_pos=s_tx[idx],
                   sat_vel=v_tx[idx], pr=pr[idx], rr=rr[idx], sat_clk=self.sat_clk[idx],
                   sat_clk_drift=self.sat_clk_drift[idx], accel_enu=imu)
        truth = {"t": self.t, "p_enu": self.p.copy(), "v_enu": self.v.copy(),
                 "nlos": [PRNS[k] for k in idx if nlos[k]],
                 "el_deg": np.degrees(el[idx]), "az_deg": np.degrees(az[idx])}
        return ep, truth


def write_csv(epochs, path):
    """Dump epochs in the spp.load_csv format (handy template for UrbanNav)."""
    import csv
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "prn", "x", "y", "z", "vx", "vy", "vz", "clk_m",
                     "clk_drift_mps", "pr_m", "rr_mps", "acc_e", "acc_n", "acc_u"])
        for ep in epochs:
            for k, prn in enumerate(ep.prn):
                wr.writerow([f"{ep.t:.3f}", prn, *(f"{v:.4f}" for v in ep.sat_pos[k]),
                             *(f"{v:.5f}" for v in ep.sat_vel[k]), f"{ep.sat_clk[k]:.4f}",
                             f"{ep.sat_clk_drift[k]:.5f}", f"{ep.pr[k]:.4f}",
                             f"{ep.rr[k]:.5f}", *(f"{a:.4f}" for a in ep.accel_enu)])
