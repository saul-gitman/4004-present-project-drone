# Future work

The first version of this repo did more than a course project needs. It was
rebuilt around one idea (two navigation flows into one decision), and the
features below were taken out. Each one is a planned improvement, roughly in
the order it would pay off. The old code still exists in git history:

```
git show f156a3b:spp.py          # chi-square FDE, elevation weights, Kalman velocity filter, CSV loader
git show f156a3b:boundary.py     # distance field, keep-in areas, RETRACE / HOLD
git show f156a3b:sim.py          # NLOS street canyon, troposphere, Earth rotation, satellite clocks
git show f156a3b:drone_game.html # browser game with every parameter editable
git show f156a3b:velocity_demo.html
```

Effort: **S** = an afternoon, **M** = a few days, **L** = a week or more.

## Phase 1: trust the GNSS fix

| # | Improvement | Why | Old code | Effort |
|---|---|---|---|---|
| 1 | **Residual test (RAIM-lite).** After the least-squares fix, `SSE = rᵀr/σ² ~ χ²(n−4)`. If it fails, drop the satellite with the largest standardised residual and solve again. | A reflected (NLOS) signal biases the fix without changing HDOP, so σ_h alone can be too optimistic in a street canyon. | `spp.fde`, `chi2_threshold` | M |
| 2 | **Verified / not verified fix.** Trust σ_h only if the test passes with ≥ 3 spare satellites; otherwise use a large fallback σ_h (50 m). | Two NLOS satellites can hide in 1–2 spare satellites. The old self-test needed 3 to have zero "error > k·σ_h" epochs. | `params.MIN_DOF_INTEGRITY`, `SIGMA_H_FALLBACK` | S (after 1) |
| 3 | **Elevation weighting** `W = diag(sin² el)` and the matching weighted DOP. | Low satellites are noisier and more often reflected. | `spp.elevation_weights` | S |
| 4 | **Use the error ellipse, not the circle.** Street canyons stretch the ellipse across the street; use its semi-major axis instead of σ_h. | `k = √(−ln P)` assumes a circular error. | `NavSolution.sigma_major` | S |
| 5 | **Multi-constellation** (BeiDou, Galileo): one extra clock column in H per system. | In Hong Kong this roughly triples the satellites in view, which helps both HDOP and the residual test. | – | M |

## Phase 2: a more realistic world (sim.py)

| # | Improvement | Why | Old code | Effort |
|---|---|---|---|---|
| 6 | **Street canyon with NLOS reflections** instead of one sky mask: buildings block by azimuth, and some blocked satellites arrive reflected with a +20–80 m range bias. | This is what makes item 1 necessary. | `sim.Canyon` | M |
| 7 | **Troposphere, Earth rotation (Sagnac) and light time, satellite clocks** in the simulated measurements, with the matching corrections in `location.py` (`wls(..., sv_rx_time=False)` already handles Earth rotation). | Closer to real receiver data. | `spp.tropo_delay`, `los_geometry`, `sim.Sim.measure` | S |
| 8 | **Time-correlated GNSS errors** instead of white noise. | Real multipath changes slowly, so errors last longer than in the simulator. | – | S |

## Phase 3: better velocity and position fusion

| # | Improvement | Why | Old code | Effort |
|---|---|---|---|---|
| 9 | **Kalman filter for velocity** (least squares with the IMU prediction as a prior). It replaces the fixed one-step weight and keeps track of how uncertain v is. | The current weight treats the previous velocity as exact. | `spp.VelocityFilter`, `ls_with_prior` | M |
| 10 | **De-weight Doppler that failed its own residual test.** | In the old NLOS canyon, raw Doppler velocity was up to 4 m/s wrong. | `params.UNVERIFIED_RR_SCALE` | S (after 1, 9) |
| 11 | **GNSS/IMU position fusion** (loosely coupled EKF), so position also coasts smoothly through short outages. | Today the position is the raw fix, or dead reckoning with a growing σ_h. | e.g. rtklibexplorer/GNSS_IMU | L |

## Phase 4: what to do when the fix cannot be trusted

| # | Improvement | Why | Old code | Effort |
|---|---|---|---|---|
| 12 | **RETRACE and HOLD.** If the fix is not verified and the margin says "move away", the direction away from the zone cannot be trusted either. Fly back to the last verified fix by dead reckoning, then hover. | Today an outage makes the aircraft back away along a dead-reckoned direction, which slowly drifts. | `boundary.ReturnGuard` | M |

## Phase 5: sensors

| # | Improvement | Why | Old code | Effort |
|---|---|---|---|---|
| 13 | **Sensor network.** Several sensors (front, sides, rear), each with its own mount rotation `R_bs` and field of view; `radar_to_ned` already applies `R_nb · R_bs`. | Real aircraft use several narrow sensors, not one 360° "magic radar". | – | M |
| 14 | **Realistic sensor model:** noise growing with range, missed detections, false alarms, clutter from the ground. | The magic radar always sees everything in range. | – | M |
| 15 | **Sensor-range speed limit:** a building is only seen at `SENSOR_RANGE`, so the speed must satisfy `v ≤ v_allow(SENSOR_RANGE − margin)`, about 23 m/s with the current numbers. | You cannot brake for what you have not seen yet. | – | S |
| 16 | **Add the sensed buildings to the map,** so they are avoided even after they leave the sensor's view. | Today an obstacle exists only while it is in range. | – | M |

## Phase 6: real maps and real data

| # | Improvement | Why | Old code | Effort |
|---|---|---|---|---|
| 17 | **Real restricted zones** (e.g. HK CAD eSUA, CAAS in Singapore) as GeoJSON: MultiPolygon, holes, keep-in operating areas. | `load_zones` reads only the outer ring of Polygon features, as no-entry zones. | `boundary.load_geojson` (MultiPolygon, keep_in) | S |
| 18 | **Many or complex zones:** precompute the signed distance on a grid. | The exact polygon distance is fine for a few zones, slow for hundreds. | `boundary.DistanceField` | S |
| 19 | **3D:** zone altitude bands, climbing as a third escape (next to braking and turning), wind (ground speed ≠ airspeed). | Restricted zones have height limits, and a tailwind lengthens braking. | – | L |
| 20 | **UrbanNav data** (PolyU IPNL): read RINEX with the full `gnss_lib_py` package or IPNL's `pyrtklib`, then feed the same `location.py`. | Shows the method on real Hong Kong canyon data. | `spp.load_csv` (CSV template) | M |

## Demo

| # | Improvement | Old code | Effort |
|---|---|---|---|
| 21 | Upload your own GeoJSON zone in `demo.html`. | – | S |
| 22 | Expose more parameters, or bring back the full game and the slide deck on where v comes from. | `drone_game.html`, `velocity_demo.html` | S |
| 23 | Work offline: bundle Pyodide and numpy next to `demo.html` instead of loading them from the CDN. | – | S |
