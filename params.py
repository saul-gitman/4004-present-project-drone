"""
params.py - design inputs for the SPP geofence.

Everything in this file is *chosen*, not measured. The measured state
(position p, velocity v, horizontal sigma sigma_h) no longer lives here: it
comes out of spp.py every GNSS epoch, and boundary.py turns it into a buffer
radius and a speed cap.

    r_buffer = R_DRONE + K_SIGMA * sigma_h + v * DT
    v_allow  = clip((d - R_DRONE - K_SIGMA * sigma_h) / DT, -V_RETURN, V_MAX)
"""

import math

# --------------------------------------------------------------------------
# Local frame. spp.py reports ENU metres about this point and boundary.py
# converts fence lat/lon into the same frame, so they must share it.
# (Default: PolyU campus, Hung Hom. Move it to the start of your UrbanNav run.)
# --------------------------------------------------------------------------
ORIGIN_LAT = 22.3045      # deg
ORIGIN_LON = 114.1800     # deg
ORIGIN_H = 10.0           # m, ellipsoidal height

# --------------------------------------------------------------------------
# Drone
# --------------------------------------------------------------------------
R_DRONE = 0.5             # m, physical radius including props
V_MAX = 10.0              # m/s, horizontal cruise limit

# --------------------------------------------------------------------------
# Timing: how long the drone keeps flying at v before a fence decision
# takes effect. This is the Delta-t in r_buffer.
# --------------------------------------------------------------------------
GNSS_RATE_HZ = 1.0                 # UrbanNav receivers log at 1 Hz
DT_GNSS = 1.0 / GNSS_RATE_HZ       # worst-case age of the fix
DT_COMPUTE = 0.1                   # SPP + field lookup
DT_REACT = 0.5                     # controller and motors respond to a new cap
DT = DT_GNSS + DT_COMPUTE + DT_REACT

# --------------------------------------------------------------------------
# Risk -> k.  sigma_h = HDOP * sigma_UERE is the 2-D rms error (drms).
# For a circular 2-D Gaussian, P(|error| > k * drms) = exp(-k^2), so the
# multiplier follows from the risk you accept. 1e-4 gives k = 3.03.
# Elongated canyon ellipses have heavier tails; see README "Limits".
# --------------------------------------------------------------------------
P_EXCEED = 1e-4
K_SIGMA = math.sqrt(-math.log(P_EXCEED))

# --------------------------------------------------------------------------
# GNSS measurement model (zenith-equivalent sigmas; a satellite at elevation
# el gets sigma / sin(el), i.e. weight w = sin^2(el)).
# --------------------------------------------------------------------------
SIGMA_UERE = 5.0          # m, pseudorange (consumer receiver, open sky)
SIGMA_RR = 0.2            # m/s, range rate from Doppler
ELEV_MASK_DEG = 10.0
WEIGHTING = "sin2el"      # "sin2el" or "equal"
APPLY_TROPO = True        # simple elevation-mapped troposphere model
APPLY_EARTH_ROTATION = True   # Sagnac: rotate satellite by omega_e * tau

# --------------------------------------------------------------------------
# Integrity (chi-square residual test, RAIM-lite)
# --------------------------------------------------------------------------
P_FA = 1e-3               # false-alarm probability per epoch
MAX_EXCLUSIONS = 2        # satellites the FDE loop may drop per epoch
MIN_DOF_INTEGRITY = 3     # redundancy (n - 4) still needed after exclusion to
                          # trust sigma_h. With 1-2 dof, two NLOS satellites can
                          # hide inside the residuals (see README "Limits").
SIGMA_H_FALLBACK = 50.0   # m, sigma_h used whenever the fix is not verified

# --------------------------------------------------------------------------
# Velocity filter = least squares with an IMU prior (= Kalman update)
# --------------------------------------------------------------------------
USE_VELOCITY_FILTER = True
SIGMA_ACC_IMU = 0.3       # m/s^2, IMU accel error -> prior velocity growth
SIGMA_ACC_NO_IMU = 2.0    # m/s^2, manoeuvre allowance when no IMU is logged
SIGMA_DRIFT_RW = 0.5      # m/s per sqrt(s), receiver clock-drift random walk
UNVERIFIED_RR_SCALE = 5.0 # Doppler sigma multiplier when the velocity chi2 test
                          # is not verified (lean on the IMU prior instead)

# --------------------------------------------------------------------------
# Fence distance field
# --------------------------------------------------------------------------
FIELD_RES = 2.0           # m, grid spacing
FIELD_MARGIN = 200.0      # m, grid extends this far beyond the fences
USE_CLOSING_SPEED = True  # v in the buffer = speed towards the fence, not |v|

# --------------------------------------------------------------------------
# Return logic (boundary.py)
# --------------------------------------------------------------------------
V_RETURN = 2.0            # m/s, top speed when pushing out (RETURN) or retracing
RETRACE_TOL = 3.0         # m, retrace target reached: hover and wait for a verified fix


def summary() -> str:
    return (
        f"k = {K_SIGMA:.2f}  (P_exceed = {P_EXCEED:g})\n"
        f"dt = {DT:.2f} s  (GNSS {DT_GNSS:.2f} + compute {DT_COMPUTE:.2f}"
        f" + react {DT_REACT:.2f})\n"
        f"r_drone = {R_DRONE} m, v_max = {V_MAX} m/s\n"
        f"sigma_UERE = {SIGMA_UERE} m, sigma_rr = {SIGMA_RR} m/s,"
        f" mask = {ELEV_MASK_DEG} deg, P_fa = {P_FA:g}"
    )


if __name__ == "__main__":
    print(summary())
    print()
    print(" HDOP   v(m/s)  sigma_h(m)  r_buffer(m)")
    for hdop in (0.8, 1.5, 3.0, 6.0):
        for v in (0.0, 5.0, 10.0):
            sh = hdop * SIGMA_UERE
            rb = R_DRONE + K_SIGMA * sh + v * DT
            print(f" {hdop:4.1f}   {v:5.1f}   {sh:9.1f}   {rb:10.1f}")
