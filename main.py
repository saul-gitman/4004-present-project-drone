"""
main.py - the whole flow, one GNSS epoch at a time.

    sim.World --GNSS epoch, IMU--> location.Navigator.update   flow 1: p, v, sigma_h
              --radar returns----> location.radar_to_ned       flow 2: obstacle range, direction
    boundary.decide(pilot command, nav, zones, obstacles)   ->  state + safe command
    sim.World.fly(safe command)

    python main.py      four scripted flights and a summary

demo.html calls web_reset() / web_step() below; its parameter values reach
params.py through set_params().
"""

from __future__ import annotations

import json
import math

import numpy as np

import boundary
import location
import params as P
import sim

START = (0.0, 0.0)
ZONE_TARGET = (230.0, 230.0)        # inside the EXAMPLE zone
BUILDING_TARGET = (350.0, -350.0)   # behind the EXAMPLE tower

_DEFAULTS = {k: v for k, v in vars(P).items() if k.isupper()}


def set_params(overrides):
    """Restore the params.py values, then apply `overrides` (names must exist in params.py)."""
    for k, v in _DEFAULTS.items():
        setattr(P, k, v)
    for k, v in overrides.items():
        if k not in _DEFAULTS:
            raise KeyError(f"params.py has no {k}")
        setattr(P, k, type(_DEFAULTS[k])(v))


def pilot(nav, target):
    """The pilot flies straight at the target at V_MAX and slows down near it."""
    if nav.p is None:
        return np.zeros(2)
    to_go = np.asarray(target, dtype=float) - nav.p
    dist = float(np.hypot(*to_go))
    return to_go / max(dist, 1e-9) * min(P.V_MAX, dist / 2.0)


class Flight:
    def __init__(self, start=START, target=ZONE_TARGET, seed=1):
        heading = math.degrees(math.atan2(target[1] - start[1], target[0] - start[0]))
        self.world = sim.World(start, heading, seed)
        self.nav = location.Navigator()
        self.zones = boundary.load_zones()
        self.target = np.asarray(target, dtype=float)

    def step(self):
        dt = 1.0 / P.GNSS_RATE_HZ
        ep, imu, returns = self.world.gnss(), self.world.imu(), self.world.radar()
        nav = self.nav.update(ep, imu, dt)                           # flow 1
        obstacles = location.radar_to_ned(returns, nav.heading)      # flow 2
        v_pilot = pilot(nav, self.target)
        dec = boundary.decide(v_pilot, nav, self.zones, obstacles)
        clearance = self.world.clearance(self.zones)                 # truth, before moving
        self.world.fly(dec.v_cmd, dt)
        return {"ep": ep, "nav": nav, "returns": returns, "obstacles": obstacles,
                "v_pilot": v_pilot, "dec": dec, "clearance": clearance}


# ==========================================================================
# Command line: four flights
# ==========================================================================
def run(label, target, overrides=None, seconds=50, change=None, every=4):
    set_params(overrides or {})
    f = Flight(START, target)
    states, rows = {}, []
    for k in range(seconds):
        if change and k == change[0]:
            for name, value in change[1].items():
                setattr(P, name, value)
        r = f.step()
        nav, dec = r["nav"], r["dec"]
        states[dec.state] = states.get(dec.state, 0) + 1
        rows.append((k, r))

    print(label)
    print("     t  zone  bldg  sats  HDOP  sig_h   v    w   threat    gap  d_brake d_turn  state")
    for k, r in rows[::every]:
        nav, w = r["nav"], r["dec"].worst
        th = (f"{w.kind:8s} {w.gap:6.1f}  {w.d_brake:6.1f} {w.d_turn:6.1f}" if w
              else f"{'-':8s} {'':6s}  {'':6s} {'':6s}")
        print(f"  {k:4d} {r['clearance'][0]:5.0f} {r['clearance'][1]:5.0f}  {nav.n_sats:4d}"
              f"  {nav.hdop:4.2f} {nav.sigma_h:6.1f} {np.hypot(*nav.v):4.1f} {nav.w_gnss:4.2f}"
              f"   {th}  {r['dec'].state}")
    zmin = min(r["clearance"][0] for _, r in rows)
    bmin = min(r["clearance"][1] for _, r in rows)
    first = next((k for k, r in rows if r["dec"].state != "CLEAR"), None)
    print("  states: " + ", ".join(f"{s} {n}" for s, n in states.items()))
    print(f"  first intervention at t = {first} s; closest true distance: zone {zmin:.1f} m,"
          f" building {bmin:.1f} m (aircraft radius {P.R_DRONE} m)"
          f" -> {'BREACH' if min(zmin, bmin) < P.R_DRONE else 'no breach'}")
    print()
    return rows


def main():
    run(f"1. Cruise ({P.V_MAX:.0f} m/s) at the eSUA zone, open city sky (mask 15 deg)",
        ZONE_TARGET)
    run("2. 20 m/s at the building: too fast to brake, so it turns",
        BUILDING_TARGET, {"V_MAX": 20.0})
    run("3. Same as 1 in a street canyon (mask 40 deg): fewer satellites, bigger zone margin",
        ZONE_TARGET, {"SKY_MASK_DEG": 40.0})
    run("4. Same as 1, sky blocked (mask 50 deg) from t = 10 s: IMU only, margin keeps growing",
        ZONE_TARGET, change=(10, {"SKY_MASK_DEG": 50.0}))
    set_params({})


# ==========================================================================
# demo.html: JSON in, JSON out
# ==========================================================================
_flight = None


def web_reset(arg):
    """arg: JSON {"params": {...}, "target": [n, e]}. Returns the static scene."""
    global _flight
    cfg = json.loads(arg)
    set_params(cfg.get("params", {}))
    _flight = Flight(START, cfg.get("target", ZONE_TARGET))
    return json.dumps({
        "start": list(START), "zone_target": list(ZONE_TARGET), "building_target": list(BUILDING_TARGET),
        "zones": [{"name": z.name, "ne": z.ne.tolist()} for z in _flight.zones],
        "buildings": [{"name": n, "ne": poly.tolist()} for n, poly in sim.BUILDINGS.items()],
        "defaults": _DEFAULTS,
    })


def web_params(arg):
    """arg: JSON {...}. Apply new values now, without restarting the flight."""
    set_params(json.loads(arg))
    return "ok"


def web_step(arg):
    """arg: JSON {"target": [n, e]} (optional). One epoch; returns everything the page draws."""
    cfg = json.loads(arg)
    if "target" in cfg:
        _flight.target = np.asarray(cfg["target"], dtype=float)
    w = _flight.world
    heading_before = w.heading
    r = _flight.step()
    nav, dec = r["nav"], r["dec"]
    lst = lambda a: None if a is None else [float(x) for x in a]
    used = set(nav.used)
    return json.dumps({
        "t": float(nav.t - sim.T0),
        "truth": {"track": [lst(p) for p in w.track], "heading": w.heading,
                  "heading_before": heading_before, "speed": w.speed,
                  "zone": r["clearance"][0], "building": r["clearance"][1]},
        "nav": {"fix": nav.fix, "p": lst(nav.p), "v": lst(nav.v), "v_gnss": lst(nav.v_gnss),
                "v_imu": lst(nav.v_imu), "w": nav.w_gnss, "a": lst(nav.a),
                "heading": nav.heading, "sigma_h": nav.sigma_h,
                "hdop": None if math.isnan(nav.hdop) else nav.hdop,
                "n_sats": nav.n_sats, "n_seen": len(r["ep"]["prn"]),
                "iono": None if math.isnan(nav.iono_mean) else nav.iono_mean},
        "sky": [[prn, el, az, seen, prn in used] for prn, el, az, seen in r["ep"]["sky"]],
        "radar": [{"range": ret["range"], "bearing": ret["bearing"], "u": lst(o.u)}
                  for ret, o in zip(r["returns"], r["obstacles"])],
        "threats": [{"kind": th.kind, "name": th.name, "d": th.d, "n": lst(th.n_hat),
                     "margin": th.margin, "gap": th.gap, "v_c": th.v_c, "v_allow": th.v_allow,
                     "d_react": th.d_react, "d_brake": th.d_brake, "d_turn": th.d_turn,
                     "worst": th is dec.worst} for th in dec.threats],
        "state": dec.state, "v_pilot": lst(r["v_pilot"]), "v_cmd": lst(dec.v_cmd),
    })


if __name__ == "__main__":
    main()
