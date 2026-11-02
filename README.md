# SPP geofence: least squares, used three ways

A drone geofence where every safety number comes from the course's
single-point-positioning least squares:

| LS output | becomes | used for |
|---|---|---|
| position `p`, cofactor `Q = (HᵀWH)⁻¹` | `HDOP`, `σ_h = HDOP·σ_UERE` | buffer width `k·σ_h` |
| Doppler LS velocity `v` (same `H`) | closing speed `v_close` | buffer growth `v·Δt` and the speed cap |
| residuals `r`, `SSE ~ χ²(n−4)` | exclude a bad satellite, or flag the fix | trusting `σ_h` at all |

```
r_buffer = r_drone + k·σ_h + v_close·Δt
v_cap    = (d − r_drone − k·σ_h) / Δt
```

Urban canyon → fewer satellites, worse geometry → higher HDOP → bigger buffer →
the drone slows down earlier near fences, without any extra rule.

```mermaid
flowchart LR
  raw["GNSS epoch<br/>pseudorange ρ, Doppler f_D<br/>sat pos/vel, clocks"] --> pos["Pseudorange WLS<br/>p, Q → HDOP → σ_h"]
  pos --> chi["χ² residual test<br/>exclude / flag"]
  chi --> pos
  pos --> vel["Doppler WLS (same H)<br/>v, clock drift"]
  vel --> kf["LS with IMU prior<br/>= Kalman update"]
  imu["IMU accel"] --> kf
  pos --> bnd["boundary.py<br/>d, n̂ from distance field"]
  kf --> bnd
  bnd --> cap["r_buffer, v_cap<br/>CLEAR / SLOW / STOP"]
  cap --> field["field.py"]
```

## Files

| file | job |
|---|---|
| `params.py` | Design inputs only: drone radius, timing (`Δt`), risk → `k`, noise sigmas, χ² false-alarm rate, filter noise. Nothing measured lives here any more. |
| `spp.py` | Pseudorange WLS, Doppler WLS, DOPs, χ² fault detection and exclusion, the velocity filter. Outputs `NavSolution` with `p_enu`, `v_enu`, `sigma_h` per epoch. |
| `boundary.py` | Fence polygons (lat/lon) → local metres → precomputed signed distance field. Buffer radius, speed cap, state, safe velocity command. |
| `sim.py` | Synthetic epochs with known truth (constellation, canyon, NLOS, drone, IMU). Test data only, not course content. |
| `velocity_demo.html` | Presentation deck that walks through the velocity idea with live demos. Open it in a browser. |
| `field.py` | Unchanged. It is not in this repo yet; copy it in. See "Hooking up field.py" below. |

Only `numpy` is needed (Python 3.10+).

```
python params.py      # derived k, Δt and a buffer table
python spp.py         # self-test: open sky, injected fault, urban canyon
python spp.py my.csv  # run on your own epochs (format below)
python boundary.py    # closed loop: fly at a fence and let the buffer stop the drone
```

---

## 1. Position, covariance and the fence buffer

For satellite `i` with corrected pseudorange `ρ_i` (satellite clock added back,
troposphere removed, Earth rotation applied to the satellite position):

```
ρ_i = ‖s_i − x‖ + b + ε_i          b = c·δt (receiver clock, metres)
e_i = (s_i − x) / ‖s_i − x‖         unit line of sight, receiver → satellite
H   = [ −e_iᵀ  1 ]                  one row per satellite
Δρ  = ρ − (‖s − x₀‖ + b₀)
Δx  = (HᵀWH)⁻¹ HᵀW Δρ               iterate until Δx is tiny (spp.solve_position)
```

Weights are dimensionless, `w_i = sin²(el_i)`, so a zenith satellite has
weight 1 and satellite `i` has `σ_i = σ_UERE / sin(el_i)`. With that scaling

```
cov(x̂) = σ_UERE² · Q,     Q = (HᵀWH)⁻¹                    (4×4)
Q_ENU  : the same Q built from rows [ −(R·e_i)ᵀ  1 ]    (R = ECEF→ENU)
HDOP   = √(Q_EE + Q_NN)
σ_h    = HDOP · σ_UERE                                  (2-D rms, "drms")
```

`spp.py` also reports the classic equal-weight HDOP (`hdop_geom`) so you can
compare with the lecture definition.

**Choosing k.** For a circular 2-D Gaussian, `P(|error| > k·drms) = e^(−k²)`.
`params.P_EXCEED = 1e-4` gives `k = √(−ln 1e-4) = 3.03`.

## 2. Velocity from Doppler (how `v` is calculated)

The receiver measures Doppler `f_D` in Hz. Convert it to range rate (positive
Doppler means the satellite is getting closer, so the range is shrinking):

```
ρ̇_i = −λ · f_D,i          λ = c / f_L1 = 0.1903 m
```

The range rate is the relative velocity along the line of sight, plus the
clock drifts:

```
ρ̇_i = e_iᵀ (v_sat,i − v_u) + c·δṫ_u − c·δṫ_sat,i
```

Everything about the satellite is known from the ephemeris, so move it to the
left. What is left is linear in the unknowns `[v_u, c·δṫ_u]`:

```
z_i = ρ̇_i − e_iᵀ v_sat,i + c·δṫ_sat,i  =  −e_iᵀ v_u + c·δṫ_u
z   = H · [v_u ; c·δṫ_u]                with the SAME rows H = [ −e_iᵀ  1 ]
v̂   = (HᵀWH)⁻¹ HᵀW z                    one shot, no iteration
cov = σ_ρ̇² · Q                          σ_v,h = HDOP · σ_ρ̇
```

Code path (`spp.SPP.step`):

1. `los_geometry` gives `e_i` and Earth-rotation-corrected satellite velocities at the final position fix.
2. `doppler_system` builds `z` and `H` (rows rotated into ENU so `v` comes out as east/north/up).
3. `fde(solve_velocity, …)` solves and runs the χ² test on the Doppler residuals with `σ_ρ̇`.
4. The filter (section 4) blends that with the IMU prediction.

Same geometry, same weights, same HDOP. Bad geometry inflates velocity noise
exactly as it inflates position noise.

In the buffer, `v` is the **closing speed** `v_close = max(0, −v·n̂)`, where
`n̂` is the distance-field gradient (pointing away from the nearest fence).
Flying parallel to a fence does not grow the buffer. Set
`params.USE_CLOSING_SPEED = False` to use `|v|` instead.

## 3. Filter 1: residual χ² test (RAIM-lite)

After any solve, the weighted residuals tell you whether the measurements
agree with each other:

```
r   = Δρ − H·Δx
SSE = rᵀ W r / σ²   ~  χ²(n − 4)     if all satellites are healthy
```

`spp.fde` (used for both pseudoranges and Doppler):

1. If `n − 4 < 1`: no redundancy, nothing to test.
2. If `SSE ≤ T(n−4, P_fa)`: pass. The threshold is the exact χ² quantile (`chi2_threshold`, no scipy).
3. Otherwise drop the satellite with the largest **standardised** residual
   `|r_i| / √(W⁻¹ − H Q Hᵀ)_ii` and solve again. Exclusion needs `n ≥ 6`, so the reduced set can still be tested.

The position fix is **verified** only if the test passes with at least
`params.MIN_DOF_INTEGRITY = 3` redundant satellites left. An unverified fix
still reports a position, but `σ_h` is raised to `SIGMA_H_FALLBACK = 50 m`.
The buffer then grows to about 150 m, so the drone stays well away from fences
until the sky opens up.

Why 3 and not 1: in the synthetic deep canyon (5–6 satellites, 2–3 of them
NLOS) two biased satellites can hide inside a 1–2 dof residual space. With
`MIN_DOF_INTEGRITY = 1` the self-test had 117 of 180 epochs where the true
error was larger than `k·σ_h`. With 3 there were none, and open sky is
unaffected because it normally has 4–5 dof.

## 4. Filter 2: least squares with an IMU prior (= Kalman filter)

State `x = [v_E, v_N, v_U, c·δṫ]`. Each epoch (`spp.VelocityFilter`):

```
predict   x⁻ = x + [a_imu·Δt ; 0]
          P⁻ = P + diag((σ_a·Δt)², (σ_a·Δt)², (σ_a·Δt)², σ_d²·Δt)

update    x̂ = argmin  ‖z − Hx‖²_(W/σ_ρ̇²)  +  ‖x − x⁻‖²_(P⁻⁻¹)
            = (HᵀR⁻¹H + P⁻⁻¹)⁻¹ (HᵀR⁻¹z + P⁻⁻¹x⁻)       R⁻¹ = W/σ_ρ̇²
          P = (HᵀR⁻¹H + P⁻⁻¹)⁻¹
```

That is the Kalman update in information form: `spp.py` checks it against
`x⁻ + K(z − Hx⁻)` and they agree to 1e-14. Two consequences you can show:

* **The trade-off is a single knob.** `σ_a` small means trusting the IMU, large means trusting GNSS. Down-weighting low-elevation satellites in `W` is the course-legit version of "trust GNSS less".
* **It keeps working with fewer than 4 satellites.** The prior adds four pseudo-measurements, so 1–3 Doppler rows still improve `v`. With 0 rows the filter coasts on the IMU and its `σ` grows.

Without IMU data the prior is constant velocity with `SIGMA_ACC_NO_IMU`.
Position stays a snapshot fix, so `σ_h` keeps its direct HDOP meaning.

## 5. Closing the loop (`boundary.py`)

```
d        = signed distance from p̂ to the nearest fence edge (+ = allowed side)
n̂        = ∇d / |∇d|
r_buffer = r_drone + k·σ_h + v_close·Δt
v_cap    = clip((d − r_drone − k·σ_h) / Δt, 0, V_MAX)
```

| state | condition | command (`safe_velocity`) |
|---|---|---|
| CLEAR | `d ≥ r_buffer` | as planned |
| SLOW | `d < r_buffer`, same as `v_close > v_cap` | closing component cut to `v_cap`, sideways part kept |
| STOP | `d ≤ r_drone + k·σ_h` | hover |
| NO-FIX | fewer than 4 satellites | hover |

STOP hovers rather than sliding along the fence: once the confidence disc
touches a fence, `n̂` at the *estimated* position can point the wrong way.
An earlier version kept the sideways component and flew straight through a
fence in the NLOS canyon run.

`Δt = DT_GNSS + DT_COMPUTE + DT_REACT = 1.0 + 0.1 + 0.5 = 1.6 s`: the time the
drone keeps flying at `v` before a new cap takes effect.

The distance field is computed once on a 2 m grid (`DistanceField`); look-ups
are bilinear, with an exact fallback off the grid. Fences are `keep_out`
(restricted zones) or `keep_in` (approved operating area). `EXAMPLE_FENCES`
are made-up polygons around the default origin. For real zones, export them
as GeoJSON and use `load_geojson()`.

## Self-test results

`python spp.py` (synthetic, seed 3, 180 s at 8 m/s):

| case | sats | HDOP | pos rms | predicted σ_h | vel rms raw → filtered | χ² |
|---|---|---|---|---|---|---|
| open sky | 9.0 | 1.76 | 8.85 m | 8.82 m | 0.36 → 0.27 m/s | 1 exclusion |
| + 60 m fault on one satellite for 90 s | 8.5 | 1.86 | 9.18 m | 9.28 m | 0.38 → 0.29 m/s | 91 exclusions, error unchanged |
| deep canyon, NLOS | 5.5 | 2.83 | 56 m | 50 m (fallback) | 1.66 → 0.94 m/s | every epoch unverified, 0 misleading |

`python boundary.py` (fly at 10 m/s towards restricted zone A):

| case | slow-down starts | stops | breach |
|---|---|---|---|
| open sky | 48 m out | 16 m out | no |
| street canyon, 15 m buildings | 48 m out | 19 m out | no |
| deep canyon, NLOS | immediately (unverified fix) | never leaves the start, 209 m out | no |

Across seeds 1–8 none of the 24 runs breached. Deep-canyon runs usually
hovered 127–209 m out. Seed 8 was the exception: its sky kept 6–8 satellites,
56 of 70 fixes were verified, and the drone ended 17.8 m out.

## Running on UrbanNav

UrbanNav ships RINEX observation and navigation files. `spp.py` takes epochs
that already have satellite positions, so compute those with your lab's
ephemeris code (or RTKLIB) and write one CSV row per satellite per epoch:

| column | unit | required |
|---|---|---|
| `t` | s | yes (time tag; rows with the same `t` form one epoch) |
| `prn` | text | yes |
| `x, y, z` | m, ECEF at transmit time | yes |
| `pr_m` | m, raw pseudorange | yes |
| `vx, vy, vz` | m/s, ECEF | for velocity |
| `rr_mps` **or** `doppler_hz` | m/s or Hz (`rr = −λ·f_D`) | for velocity |
| `clk_m` | m, `c·δt_sat` (added to `pr`) | recommended |
| `clk_drift_mps` | m/s, `c·δṫ_sat` | optional |
| `iono_m` | m, e.g. Klobuchar (subtracted) | optional |
| `acc_e, acc_n, acc_u` | m/s², IMU in ENU, gravity removed | optional (filter) |

`sim.write_csv()` writes this format if you want a template. Move
`params.ORIGIN_*` to the start of your run so ENU stays small.

Two things to check on real data:

* If your satellite positions are already corrected for Earth rotation, set `APPLY_EARTH_ROTATION = False`, or the correction is applied twice.
* GPS only: `H` has one clock column. Adding Galileo or BeiDou means one inter-system bias column per extra constellation, and it also gives the χ² test the redundancy it needs in canyons.

## Hooking up field.py

`field.py` is not in this repo, so its interface was not available when this
was written. Whatever it reads, the values now come from here:

```python
from spp import SPP
from boundary import DistanceField, example_fences, check, safe_velocity

solver = SPP()
dfield = DistanceField(example_fences())       # or load_geojson("esua.geojson")

sol = solver.step(epoch)       # sol.p_enu, sol.v_enu, sol.sigma_h, sol.integrity
chk = check(dfield, sol)       # chk.d, chk.n_hat, chk.r_buffer, chk.v_cap, chk.state
v_cmd = safe_velocity(v_goal, chk)
```

If `field.py` used to read `p`, `v`, `sigma` from `params.py`, point those
reads at `sol.p_enu`, `sol.v_enu` and `sol.sigma_h` instead.

## Limits

* **NLOS is not geometry.** HDOP covers geometry only. A reflected signal adds a bias that HDOP cannot see, which is why the χ² test and `MIN_DOF_INTEGRITY` exist. In a deep GPS-only canyon the honest answer is "not verified", so the drone keeps a big buffer.
* **Elongated ellipses.** `k` assumes a circular error. Canyon geometry stretches the ellipse across the street, which makes the tail heavier along that axis. `NavSolution.sigma_major` (1-σ semi-major axis) is reported; using it instead of `σ_h` is the conservative option.
* **White-noise simulator.** Real multipath is time-correlated, so real residuals look smoother and biases persist for longer than in `sim.py`.

## Future work: sensors

Sensors (camera, lidar, radar) stay on the future-work slide. Teaser: the
same buffer logic gives "you can't outrun your headlights": with sensing
range `R_s`, the speed must satisfy `R_s ≥ r_drone + v·Δt + v²/(2a)`.
