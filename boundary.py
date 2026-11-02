"""
boundary.py - fences as lat/lon polygons -> local ENU metres -> a precomputed
signed distance field, plus the decision logic that turns spp.py output into
a velocity command, including how the drone is returned to safety.

    d        = signed distance from p_hat to the nearest fence edge (+ = allowed side)
    r_buffer = R_DRONE + K_SIGMA * sigma_h + v_close * DT
    v_allow  = clip((d - R_DRONE - K_SIGMA * sigma_h) / DT, -V_RETURN, V_MAX)

v_allow is the signed velocity field: the fastest the drone may close on the
nearest fence. Positive means "approach at most this fast", negative means
"you are inside the stop line, move away at this speed". The two are the same
inequality, d >= r_buffer, solved for the closing speed.

Verified fix (chi2 passed with enough redundancy) - n_hat is trusted:
    CLEAR    v_close <= v_allow            pilot command passes
    SLOW     0 <= v_allow < v_close        closing part cut to v_allow, sideways kept
    RETURN   v_allow < 0                   closing part forced to v_allow: pushed out along n_hat

Fix not verified (sigma_h = fallback) - n_hat is NOT trusted:
    CLEAR / SLOW                           whole speed |v| capped at v_allow
    RETRACE  v_allow < 0                   fly back to the last verified fix, steering by
                                           dead reckoning with Doppler velocity
    HOLD     no fix, or nothing to retrace hover

The fences below are EXAMPLES placed around the default origin so the demo
has something to fly at. They are not real eSUA zones: export the real
restricted zones (GeoJSON) and load them with load_geojson().
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

import params as P
from spp import LocalFrame, NavSolution

# --------------------------------------------------------------------------
# Fences
# --------------------------------------------------------------------------
EXAMPLE_FENCES = [
    # (name, kind, [(lat, lon), ...])  -- illustrative only
    ("EXAMPLE operating area", "keep_in",
     [(22.299985, 114.175148), (22.299985, 114.186793),
      (22.310821, 114.186794), (22.310821, 114.175147)]),
    ("EXAMPLE restricted zone A", "keep_out",
     [(22.306919, 114.180156), (22.305971, 114.181921), (22.306449, 114.182778),
      (22.307514, 114.182542), (22.308215, 114.180848)]),
    ("EXAMPLE restricted zone B", "keep_out",
     [(22.306143, 114.178106), (22.305872, 114.178610),
      (22.306420, 114.178950), (22.306691, 114.178446)]),
]


@dataclass
class Fence:
    name: str
    kind: str              # "keep_out" (restricted zone) or "keep_in" (operating area)
    latlon: np.ndarray     # (k,2) degrees, open ring
    xy: np.ndarray         # (k,2) ENU metres about params.ORIGIN_*


def make_fence(name, kind, latlon, frame: LocalFrame) -> Fence:
    if kind not in ("keep_out", "keep_in"):
        raise ValueError(f"fence kind must be keep_out or keep_in, got {kind!r}")
    ll = np.asarray(latlon, dtype=float)
    if np.allclose(ll[0], ll[-1]):
        ll = ll[:-1]
    xy = frame.latlon_to_enu(ll[:, 0], ll[:, 1])[:, :2]
    return Fence(name, kind, ll, xy)


def example_fences(frame: LocalFrame | None = None) -> list[Fence]:
    frame = frame or LocalFrame()
    return [make_fence(n, k, ll, frame) for n, k, ll in EXAMPLE_FENCES]


def load_geojson(path, frame: LocalFrame | None = None, default_kind="keep_out"):
    """
    Polygons / MultiPolygons from a GeoJSON FeatureCollection (outer rings).
    properties.kind = "keep_in" marks an operating area; anything else is a
    restricted zone. GeoJSON order is [lon, lat].
    """
    frame = frame or LocalFrame()
    with open(path) as f:
        gj = json.load(f)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    fences = []
    for i, ft in enumerate(feats):
        props = ft.get("properties") or {}
        name = str(props.get("name") or props.get("Name") or f"zone {i}")
        kind = props.get("kind", default_kind)
        geom = ft["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for j, poly in enumerate(polys):
            ring = np.asarray(poly[0], dtype=float)[:, :2]
            label = name if len(polys) == 1 else f"{name} [{j}]"
            fences.append(make_fence(label, kind, ring[:, ::-1], frame))
    return fences


# --------------------------------------------------------------------------
# Signed distance (exact, vectorised)
# --------------------------------------------------------------------------
def _edges(poly):
    return zip(poly, np.roll(poly, -1, axis=0))


def _inside(px, py, poly):
    """Even-odd ray casting."""
    inside = np.zeros(np.shape(px), dtype=bool)
    for (ax, ay), (bx, by) in _edges(poly):
        crosses = (ay > py) != (by > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = ax + (py - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (px < x_at)
    return inside


def _edge_distance(px, py, poly):
    d = np.full(np.shape(px), np.inf)
    for (ax, ay), (bx, by) in _edges(poly):
        dx, dy = bx - ax, by - ay
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        d = np.minimum(d, np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return d


def signed_distance(fences, px, py):
    """Distance to the nearest fence edge; positive on the allowed side."""
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    out = np.full(px.shape, np.inf)
    for f in fences:
        d = _edge_distance(px, py, f.xy)
        ins = _inside(px, py, f.xy)
        out = np.minimum(out, np.where(ins, -d, d) if f.kind == "keep_out"
                         else np.where(ins, d, -d))
    return out


class DistanceField:
    """Signed distance precomputed on a grid; bilinear lookups at run time."""

    def __init__(self, fences, res=P.FIELD_RES, margin=P.FIELD_MARGIN):
        self.fences = list(fences)
        pts = np.vstack([f.xy for f in self.fences])
        lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
        self.res = res
        self.x = np.arange(lo[0], hi[0] + res, res)
        self.y = np.arange(lo[1], hi[1] + res, res)
        X, Y = np.meshgrid(self.x, self.y)               # (ny, nx)
        self.d = signed_distance(self.fences, X, Y)
        self.gy, self.gx = np.gradient(self.d, res)

    def _bilinear(self, grid, px, py):
        fx = (px - self.x[0]) / self.res
        fy = (py - self.y[0]) / self.res
        i0 = np.clip(np.floor(fx).astype(int), 0, len(self.x) - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, len(self.y) - 2)
        tx, ty = fx - i0, fy - j0
        v = (grid[j0, i0] * (1 - tx) * (1 - ty) + grid[j0, i0 + 1] * tx * (1 - ty)
             + grid[j0 + 1, i0] * (1 - tx) * ty + grid[j0 + 1, i0 + 1] * tx * ty)
        inside = (fx >= 0) & (fy >= 0) & (fx <= len(self.x) - 1) & (fy <= len(self.y) - 1)
        return v, inside

    def distance(self, p):
        """p: (2,) or (m,2) ENU metres -> signed distance (m)."""
        p = np.atleast_2d(p)
        d, ok = self._bilinear(self.d, p[:, 0], p[:, 1])
        if not np.all(ok):                               # off the grid: exact
            d = np.where(ok, d, signed_distance(self.fences, p[:, 0], p[:, 1]))
        return d if len(d) > 1 else float(d[0])

    def normal(self, p):
        """Unit gradient of d: points away from the nearest fence."""
        p = np.atleast_2d(p)
        gx, _ = self._bilinear(self.gx, p[:, 0], p[:, 1])
        gy, _ = self._bilinear(self.gy, p[:, 0], p[:, 1])
        g = np.column_stack([gx, gy])
        g /= np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-9, None)
        return g if len(g) > 1 else g[0]


# --------------------------------------------------------------------------
# Closing the loop: buffer, signed speed field, state, command
# --------------------------------------------------------------------------
def buffer_radius(sigma_h, v_close):
    return P.R_DRONE + P.K_SIGMA * sigma_h + max(v_close, 0.0) * P.DT


def allowed_closing_speed(d, sigma_h):
    """Signed speed field: + max approach speed, - required retreat speed."""
    return float(np.clip((d - P.R_DRONE - P.K_SIGMA * sigma_h) / P.DT, -P.V_RETURN, P.V_MAX))


@dataclass
class FenceCheck:
    state: str             # CLEAR / SLOW / RETURN / RETRACE / HOLD
    d: float               # signed distance from the estimated position (m)
    n_hat: np.ndarray      # unit normal, away from the nearest fence
    v_close: float         # m/s: closing speed (verified) or |v| (not verified)
    sigma_h: float
    r_buffer: float
    v_allow: float         # signed allowed closing speed (m/s)
    verified: bool         # True: n_hat is trusted


def check(field: DistanceField, sol: NavSolution) -> FenceCheck:
    if not sol.ok:
        return FenceCheck("HOLD", float("nan"), np.zeros(2), 0.0, float("nan"),
                          float("nan"), 0.0, False)
    p = sol.p_enu[:2]
    d = field.distance(p)
    n_hat = field.normal(p)
    v = sol.v_enu[:2] if np.all(np.isfinite(sol.v_enu[:2])) else np.zeros(2)
    verified = bool(sol.integrity)
    if verified and P.USE_CLOSING_SPEED:
        v_close = max(0.0, -float(v @ n_hat))
    else:                                      # direction not trusted: use the full speed
        v_close = float(np.hypot(*v))
    v_allow = allowed_closing_speed(d, sol.sigma_h)
    if v_allow < 0.0:
        state = "RETURN" if verified else "RETRACE"
    elif v_allow < P.V_MAX and v_close > v_allow:
        state = "SLOW"
    else:
        state = "CLEAR"
    return FenceCheck(state, d, n_hat, v_close, sol.sigma_h,
                      buffer_radius(sol.sigma_h, v_close), v_allow, verified)


def limit_velocity(v_cmd, n_hat, v_allow):
    """
    Enforce closing speed <= v_allow along n_hat; keep the sideways part.
    With v_allow < 0 this pushes the drone out at |v_allow|.
    """
    v_cmd = np.asarray(v_cmd, dtype=float).copy()
    closing = -float(v_cmd[:2] @ n_hat)
    if closing > v_allow:
        v_cmd[:2] += (closing - v_allow) * n_hat
    return v_cmd


def cap_speed(v_cmd, v_max):
    """Scale the whole horizontal velocity down to v_max (direction unchanged)."""
    v_cmd = np.asarray(v_cmd, dtype=float).copy()
    s = float(np.hypot(*v_cmd[:2]))
    if s > v_max:
        v_cmd[:2] *= max(v_max, 0.0) / s
    return v_cmd


def safe_velocity(v_goal, chk: FenceCheck):
    """Stateless part of the command logic (RETRACE needs ReturnGuard)."""
    if chk.state in ("HOLD", "RETRACE"):
        return np.zeros(3)
    if chk.verified:
        return limit_velocity(v_goal, chk.n_hat, chk.v_allow)
    return cap_speed(v_goal, chk.v_allow)


class ReturnGuard:
    """
    The fence logic with memory. It keeps the last verified fix and a dead-
    reckoned position (last verified fix + integral of the Doppler velocity).
    When the fix is not verified and the fallback buffer says "move away",
    n_hat cannot be trusted, so the drone retraces its own path back to the
    last verified fix: it got there safely, so going back is safe.
    """

    def __init__(self, field: DistanceField):
        self.field = field
        self.p_last_ok = None      # last verified p_hat (ENU, m)
        self.p_dr = None           # dead-reckoned position since then
        self.arrived = False       # retrace done: hold until a verified fix

    def step(self, sol: NavSolution, v_goal, dt):
        chk = check(self.field, sol)
        if sol.ok and sol.integrity:
            self.p_last_ok = sol.p_enu[:2].copy()
            self.p_dr = self.p_last_ok.copy()
            self.arrived = False
        elif self.p_dr is not None and np.all(np.isfinite(sol.v_enu[:2])):
            self.p_dr = self.p_dr + sol.v_enu[:2] * dt

        if chk.state == "RETRACE":
            if self.p_last_ok is None or self.arrived:
                chk.state = "HOLD"                     # nothing to go back to / already there
                return np.zeros(3), chk
            to_go = self.p_last_ok - self.p_dr
            dist = float(np.hypot(*to_go))
            if dist < P.RETRACE_TOL:
                # Stop steering by dead reckoning once there: a biased velocity
                # would otherwise make a "hovering" drone wander.
                self.arrived = True
                chk.state = "HOLD"
                return np.zeros(3), chk
            v = np.zeros(3)
            v[:2] = to_go / dist * min(P.V_RETURN, dist / P.DT)
            return v, chk
        v_cmd = safe_velocity(v_goal, chk)
        if chk.state in ("CLEAR", "SLOW"):             # SLOW = the logic cut the command
            cut = np.hypot(*(v_cmd[:2] - np.asarray(v_goal, dtype=float)[:2])) > 1e-6
            chk.state = "SLOW" if cut else "CLEAR"
        return v_cmd, chk


# --------------------------------------------------------------------------
# Demo: fly at restricted zone A and let the logic stop and return the drone
# --------------------------------------------------------------------------
def closed_loop(label, canyon=None, canyon_from=0.0, seconds=70, seed=5, verbose=False):
    from sim import Sim
    from spp import SPP

    frame = LocalFrame()
    field = DistanceField(example_fences(frame))
    sim = Sim(frame, canyon=canyon, seed=seed)
    if canyon_from > 0:                              # open sky first, canyon later
        sim.canyon = None
    solver = SPP(frame)
    guard = ReturnGuard(field)
    heading = np.radians(30.0)
    v_goal = P.V_MAX * np.array([np.sin(heading), np.cos(heading), 0.0])
    v_cmd = np.zeros(3)                              # hold until the first fix
    d_true_min, rows, states = np.inf, [], {}
    t0 = sim.t
    for _ in range(seconds):
        if canyon_from > 0 and sim.t - t0 >= canyon_from:
            sim.canyon = canyon
        ep, truth = sim.step(P.DT_GNSS, v_cmd)       # command held for one epoch
        sol = solver.step(ep)
        v_cmd, chk = guard.step(sol, v_goal, P.DT_GNSS)
        d_true = field.distance(truth["p_enu"][:2])
        d_true_min = min(d_true_min, d_true)
        states[chk.state] = states.get(chk.state, 0) + 1
        rows.append((truth["t"], d_true, chk, sol, float(np.hypot(*truth["v_enu"][:2])), truth))

    print(label)
    if verbose:
        print("     t   d_true  d_est  HDOP  sig_h  v_true  v_close  r_buf  v_allow  state")
        for t, dt_, c, s, vt, _ in rows[::5]:
            print(f"  {t - t0:4.0f}  {dt_:6.1f}  {c.d:5.1f}  {s.hdop:4.2f}  {c.sigma_h:5.1f}"
                  f"  {vt:6.2f}  {c.v_close:6.2f}  {c.r_buffer:5.1f}  {c.v_allow:6.2f}  {c.state}")
    first_slow = next((r[1] for r in rows if r[2].state != "CLEAR"), float("nan"))
    print(f"  states: " + ", ".join(f"{k} {v}" for k, v in states.items()))
    print(f"  first slow-down {first_slow:.0f} m from the fence; ended {rows[-1][1]:.1f} m away"
          f" (closest {d_true_min:.1f} m, drone radius {P.R_DRONE} m)"
          f" -> {'BREACH' if d_true_min < P.R_DRONE else 'no breach'}")
    print()
    return rows


if __name__ == "__main__":
    from sim import Canyon

    frame = LocalFrame()
    fences = example_fences(frame)
    field = DistanceField(fences)
    print(f"field: {len(field.x)} x {len(field.y)} cells at {field.res} m")
    for f in fences:
        print(f"  {f.kind:8s} {f.name}: {len(f.xy)} vertices,"
              f" signed distance from origin {float(signed_distance([f], 0.0, 0.0)):.1f} m")
    print()
    closed_loop("open sky: pilot keeps pushing at the fence", verbose=True)
    closed_loop("street canyon, 15 m buildings (LOS only)", Canyon(height_m=15.0, nlos_fraction=0.0))
    closed_loop("deep canyon from the start (never verified)", Canyon())
    rows = closed_loop("open sky, then deep canyon from t = 17 s (retrace)", Canyon(), canyon_from=17.0)
    last_ok = max((r for r in rows if r[3].integrity), key=lambda r: r[0])
    gap = np.hypot(*(rows[-1][5]["p_enu"][:2] - last_ok[5]["p_enu"][:2]))
    print(f"  retrace ended {gap:.1f} m from where the last verified fix was taken")
