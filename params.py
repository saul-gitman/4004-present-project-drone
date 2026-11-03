"""
params.py - predefined numbers for the LAE guardrail.

Everything here is chosen, not measured. demo.html hands its own values for a
few of these to the Python at run time (aircraft speed and size, k, latency,
sky blockage, sigma_UERE); this file itself never changes. Every module reads
`P.NAME` when it runs, so an override takes effect immediately.

    python params.py      derived k and latency, and braking vs turning distance
"""

import math

# --------------------------------------------------------------------------
# Local frame: positions are NED metres (north, east, down) about this point.
# (PolyU campus, Hung Hom, Hong Kong.)
# --------------------------------------------------------------------------
ORIGIN_LAT = 22.3045      # deg
ORIGIN_LON = 114.1800     # deg
ORIGIN_H = 10.0           # m, ellipsoidal height
FLIGHT_ALT = 100.0        # m above the origin; the flight is level

# --------------------------------------------------------------------------
# Aircraft (eVTOL)
# --------------------------------------------------------------------------
R_DRONE = 3.0             # m, physical radius including rotors
V_MAX = 10.0              # m/s, cruise speed the pilot asks for
A_BRAKE = 2.5             # m/s^2, braking deceleration (also its max acceleration)
R_TURN = 50.0             # m, turning radius (it can turn tighter when slow)
V_RETURN = 2.0            # m/s, back-off speed when it is inside a margin

# --------------------------------------------------------------------------
# Latency: how long the aircraft keeps flying at v before a decision acts.
# --------------------------------------------------------------------------
GNSS_RATE_HZ = 1.0        # one GNSS epoch = one decision per second
DT_COMPUTE = 0.1          # s, positioning + decision
DT_REACT = 0.5            # s, controller and motors respond
DT = 1.0 / GNSS_RATE_HZ + DT_COMPUTE + DT_REACT     # 1.6 s

# --------------------------------------------------------------------------
# Risk -> k. For a circular 2-D Gaussian, P(|error| > k * sigma_h) = exp(-k^2),
# so P_EXCEED = 1e-4 gives k = 3.03.
# --------------------------------------------------------------------------
P_EXCEED = 1e-4
K_SIGMA = math.sqrt(-math.log(P_EXCEED))

# --------------------------------------------------------------------------
# GNSS receiver
# --------------------------------------------------------------------------
SIGMA_UERE = 5.0          # m, pseudorange error (sigma_h = HDOP * SIGMA_UERE)
SIGMA_RR = 0.2            # m/s, range-rate error from Doppler
ELEV_MASK_DEG = 10.0      # deg, satellites below this are not used

# --------------------------------------------------------------------------
# IMU
# --------------------------------------------------------------------------
SIGMA_ACC_IMU = 0.3       # m/s^2, accelerometer noise
SIGMA_YAW_DEG = 1.0       # deg, heading noise

# --------------------------------------------------------------------------
# Obstacle sensor. For now one "magic radar": 360 deg, it reports every
# obstacle within range. A sensor network would have one mount angle each.
# --------------------------------------------------------------------------
SENSOR_RANGE = 150.0      # m
SENSOR_MOUNT_DEG = 0.0    # deg, sensor x-axis relative to the body's forward axis
SIGMA_RANGE = 0.5         # m
SIGMA_BEARING_DEG = 1.0   # deg

# --------------------------------------------------------------------------
# Environment (used by sim.py only): buildings around the aircraft hide every
# satellite below this elevation. 15 = open city, 35+ = street canyon.
# --------------------------------------------------------------------------
SKY_MASK_DEG = 15.0


if __name__ == "__main__":
    print(f"k  = {K_SIGMA:.2f}   (P_exceed = {P_EXCEED:g})")
    print(f"dt = {DT:.2f} s  (GNSS {1 / GNSS_RATE_HZ:.1f} + compute {DT_COMPUTE} + react {DT_REACT})")
    print(f"brake and turn need the same distance at v = sqrt(2 a R) = "
          f"{math.sqrt(2 * A_BRAKE * R_TURN):.1f} m/s")
    print()
    print(" v (m/s)  react (m)  brake (m)  turn (m)  shorter")
    for v in (5, 10, 15, 20, 25):
        react = v * DT
        brake = react + v * v / (2 * A_BRAKE)
        turn = react + R_TURN
        print(f"  {v:5.0f}   {react:8.1f}   {brake:8.1f}   {turn:7.1f}   "
              f"{'brake' if brake <= turn else 'turn'}")
