"""
boundary.py - one decision for two kinds of threat.

    threat      distance d comes from        margin
    eSUA zone   GNSS position vs the map     R_DRONE + k * sigma_h      (sigma_h = HDOP * sigma_UERE)
    obstacle    radar range                  R_DRONE + k * SIGMA_RANGE

The radar measures the distance directly, so GNSS error does not enter the
obstacle margin. A zone exists only on the map, so the GNSS error does.

For each threat, with the closing speed v_c = -v . n_hat (v from GNSS + IMU,
n_hat = unit vector pointing away from the threat):

    gap      = d - margin                       room left before the margin
    d_react  = v_c * DT                         flown before a decision acts
    d_brake  = d_react + v_c^2 / (2 A_BRAKE)    distance to brake to a stop
    d_turn   = d_react + R_TURN (1 - sin a)     distance to turn until parallel to the edge
                                                (a = angle already turned off head-on,
                                                 sin a = v_side / |v|; head-on: R_TURN)
    v_allow  = fastest v_c with d_brake <= gap  = A (sqrt(DT^2 + 2 gap / A) - DT)

The most urgent threat (largest v_c - v_allow) sets the state:

    CLEAR    v_c <= v_allow                   pilot command passes
    BRAKE    v_c >  v_allow and gap <  d_turn  closing part of the command cut to v_allow
    TURN     v_c >  v_allow and gap >= d_turn  too late to brake, room to turn: fly along the edge
    RETURN   gap <= 0                         inside the margin: back off at V_RETURN

Below v = sqrt(2 A R) braking is the shorter escape, above it turning is.
The pilot's command always has its closing speed capped at v_allow, so the
state also reads BRAKE whenever that cap changes the command.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

import params as P
from location import FRAME

ZONES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zones.geojson")


# --------------------------------------------------------------------------
# Zones: GeoJSON lat/lon -> NED polygons
# --------------------------------------------------------------------------
@dataclass
class Zone:
    name: str
    ne: np.ndarray         # (k, 2) polygon corners, north/east metres


def load_zones(path=ZONES_FILE):
    """No-entry zones from GeoJSON Polygon features ([lon, lat] order, outer ring)."""
    with open(path) as f:
        gj = json.load(f)
    zones = []
    for i, ft in enumerate(gj["features"]):
        ring = ft["geometry"]["coordinates"][0]
        if ring[0] == ring[-1]:
            ring = ring[:-1]
        ne = [FRAME.geodetic_to_ned(np.array([[lat, lon, P.ORIGIN_H]]))[0, :2]
              for lon, lat, *_ in ring]
        name = (ft.get("properties") or {}).get("name", f"zone {i + 1}")
        zones.append(Zone(name, np.array(ne)))
    return zones


def closest_point(poly, p):
    """Nearest point to p on the polygon's edges, and whether p is inside (even-odd rule)."""
    best, best_d2, inside = None, math.inf, False
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        t = min(max(float((p - a) @ ab / (ab @ ab)), 0.0), 1.0)
        q = a + t * ab
        d2 = float((p - q) @ (p - q))
        if d2 < best_d2:
            best, best_d2 = q, d2
        if (a[1] > p[1]) != (b[1] > p[1]) and \
                p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
            inside = not inside
    return best, inside


# --------------------------------------------------------------------------
# Threats
# --------------------------------------------------------------------------
@dataclass
class Threat:
    kind: str              # "zone" or "obstacle"
    name: str
    d: float               # m, distance (negative: inside a zone)
    n_hat: np.ndarray      # (2,) unit vector pointing away from the threat, NED
    margin: float          # m
    v_c: float = 0.0       # filled in by assess()
    gap: float = 0.0
    d_react: float = 0.0
    d_brake: float = 0.0
    d_turn: float = 0.0
    v_allow: float = 0.0


def zone_threats(zones, nav):
    """Flow 1 output (GNSS position) against the map."""
    margin = P.R_DRONE + P.K_SIGMA * nav.sigma_h
    out = []
    for z in zones:
        q, inside = closest_point(z.ne, nav.p)
        away = nav.p - q
        d = float(np.hypot(*away))
        n_hat = away / max(d, 1e-9)
        if inside:
            d, n_hat = -d, -n_hat
        out.append(Threat("zone", z.name, d, n_hat, margin))
    return out


def obstacle_threats(obstacles):
    """Flow 2 output (radar range and direction): no GNSS position needed."""
    margin = P.R_DRONE + P.K_SIGMA * P.SIGMA_RANGE
    return [Threat("obstacle", f"radar return {i + 1}", o.range, -o.u, margin)
            for i, o in enumerate(obstacles)]


def allowed_speed(gap):
    """Fastest closing speed that can still brake to a stop within `gap` (negative: back off)."""
    if gap <= 0.0:
        return -P.V_RETURN
    return P.A_BRAKE * (math.sqrt(P.DT ** 2 + 2.0 * gap / P.A_BRAKE) - P.DT)


def assess(th, v):
    speed = float(np.hypot(*v))
    th.v_c = max(0.0, -float(v @ th.n_hat))
    sin_a = math.sqrt(max(speed ** 2 - th.v_c ** 2, 0.0)) / speed if speed > 0 else 0.0
    th.gap = th.d - th.margin
    th.d_react = th.v_c * P.DT
    th.d_brake = th.d_react + th.v_c ** 2 / (2.0 * P.A_BRAKE)
    th.d_turn = th.d_react + P.R_TURN * (1.0 - sin_a)
    th.v_allow = allowed_speed(th.gap)
    return th


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------
def cap_closing(v_cmd, n_hat, v_allow):
    """Closing part of v_cmd (towards the threat) at most v_allow; the sideways part is kept."""
    closing = -float(v_cmd @ n_hat)
    return v_cmd + (closing - v_allow) * n_hat if closing > v_allow else v_cmd


@dataclass
class Decision:
    state: str             # CLEAR / BRAKE / TURN / RETURN / HOLD
    threats: list
    worst: Threat | None
    v_cmd: np.ndarray      # (2,) the command actually sent (north, east)


def decide(v_pilot, nav, zones, obstacles) -> Decision:
    v_pilot = np.asarray(v_pilot, dtype=float)
    if nav.p is None:                                   # no position yet
        return Decision("HOLD", [], None, np.zeros(2))
    threats = [assess(th, nav.v) for th in zone_threats(zones, nav) + obstacle_threats(obstacles)]
    if not threats:
        return Decision("CLEAR", [], None, v_pilot)

    worst = max(threats, key=lambda th: th.v_c - th.v_allow)
    late = worst.v_c > worst.v_allow                    # braking alone no longer fits
    if worst.gap <= 0.0:
        state = "RETURN"
    elif late and worst.gap >= worst.d_turn:
        state = "TURN"
    elif late:
        state = "BRAKE"
    else:
        state = "CLEAR"

    v_cmd = v_pilot
    if state == "TURN":                                 # same speed, along the edge
        along = np.array([-worst.n_hat[1], worst.n_hat[0]])
        if along @ nav.v < 0.0:
            along = -along
        v_cmd = along * float(np.hypot(*nav.v))
    for th in threats:
        v_cmd = cap_closing(v_cmd, th.n_hat, th.v_allow)
    if state == "CLEAR" and not np.allclose(v_cmd, v_pilot):
        state = "BRAKE"
    return Decision(state, threats, worst, v_cmd)
