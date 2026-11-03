"""
gnss_lib_py_subset.py - the parts of gnss_lib_py that this project uses.

Source: https://github.com/Stanford-NavLab/gnss_lib_py (Stanford NAV Lab),
commit 3a5c43c (14 Sep 2026), MIT licence (copied below).

Each function below is copied from the file named in its banner. Function
bodies are unchanged; the only edits are the import lines (replaced by the
small `consts` namespace here) and the first lines of the Klobuchar function,
marked "# CHANGED:", so it takes satellite positions directly instead of the
library's NavData objects.

    algorithms/snapshot.py    wls                      pseudorange least squares (SPP)
    utils/coordinates.py      geodetic_to_ecef         LLH  -> ECEF
                              ecef_to_geodetic         ECEF -> LLH
                              LocalCoord               ECEF <-> NED about an origin
                              ecef_to_el_az            elevation / azimuth of satellites
    utils/gnss_models.py      _calculate_iono_delay    Klobuchar ionosphere model
    utils/constants.py        the six constants in `consts`

Why a copy and not `pip install gnss-lib-py`: the full package also installs
pandas, scipy, matplotlib, plotly, georinex and more, and demo.html runs these
files in the browser, where only numpy is available.

MIT License

Copyright (c) 2021 Stanford-NavLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The coordinate functions are themselves "based on code from
https://github.com/commaai/laika", MIT License, Copyright (c) 2018 comma.ai,
under the same permission notice as above.
"""

import warnings
from types import SimpleNamespace

import numpy as np

# ---- from utils/constants.py (values unchanged) --------------------------
consts = SimpleNamespace(
    A=6378137.,                  # WGS-84 semi-major axis [m]
    B=6356752.3145,              # WGS-84 semi-minor axis [m]
    E1SQ=6.69437999014 * 0.001,  # first eccentricity squared
    E2SQ=6.73949674228 * 0.001,  # second eccentricity squared
    OMEGA_E_DOT=7.2921151467e-5, # Earth rotation rate [rad/s]
    C=299792458.,                # speed of light [m/s]
)

# ==========================================================================
# utils/coordinates.py
# ==========================================================================
EPSILON = 1e-7

def geodetic_to_ecef(geodetic, radians=False):
    """LLA to ECEF conversion.

    Parameters
    ----------
    geodetic : np.ndarray
        Float with WGS-84 LLA coordinates.
    radians : bool
        Flag of whether input [rad].

    Returns
    -------
    ecef : np.ndarray
        ECEF coordinates corresponding to input LLA.

    Notes
    -----
    Based on code from https://github.com/commaai/laika.

    """

    ratio = 1.0 if radians else (np.pi / 180.0)
    geodetic = np.array(geodetic)
    input_shape = geodetic.shape
    geodetic = np.atleast_2d(geodetic)
    if input_shape[0]==3:
        lat = ratio*geodetic[0,:]
        lon = ratio*geodetic[1,:]
        alt = geodetic[2,:]
    elif input_shape[1]==3:
        lat = ratio*geodetic[:,0]
        lon = ratio*geodetic[:,1]
        alt = geodetic[:,2]
    else:  # pragma: no cover
        raise ValueError('geodetic is incorrect shape ', geodetic.shape,
                        ' should be (N,3) or (3,N)')
    xi = np.sqrt(1 - consts.E1SQ * np.sin(lat)**2)
    x = (consts.A / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (consts.A / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (consts.A / xi * (1 - consts.E1SQ) + alt) * np.sin(lat)
    ecef = np.array([x, y, z]).T
    if input_shape[0]==3:
        ecef = ecef.T
    return ecef


def ecef_to_geodetic(ecef, radians=False):
    """ECEF to LLA conversion using Ferrari's method.

    Parameters
    ----------
    ecef : np.ndarray
        array where ECEF x, ECEF y, and ECEF z are either independent
        rows or independent columns, values should be floats
    radians : bool
        If False (default), output of lat/lon is returned in degrees.
        If True, output of lat/lon is returned in radians.

    Returns
    -------
    geodetic : np.ndarray
        Float with WGS-84 LLA coordinates corresponding to input ECEF.
        Order is returned as (lat, lon, h) and is returned in the same
        shape as the input. Height is in meters above the the WGS-84
        ellipsoid.

    Notes
    -----
    Based on code from https://github.com/commaai/laika.

    """

    ecef = np.atleast_2d(ecef)
    ecef = ecef.astype(np.float64)
    input_shape = ecef.shape
    if input_shape[0]==3:
        x_ecef, y_ecef, z_ecef = ecef[0, :], ecef[1, :], ecef[2, :]
    elif input_shape[1]==3:
        x_ecef, y_ecef, z_ecef = ecef[:, 0], ecef[:, 1], ecef[:, 2]
    else:  # pragma: no cover
        raise ValueError('Input ECEF vector has incorrect shape ', ecef.shape,
                        ' should be (N,3) or (3,N)')
    ratio = 1.0 if radians else (180.0 / np.pi)

    # Convert from ECEF to geodetic using Ferrari's methods
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Ferrari.27s_solution
    r = np.sqrt(x_ecef * x_ecef + y_ecef * y_ecef)
    E1SQ = consts.A * consts.A - consts.B * consts.B
    F = 54 * consts.B * consts.B * z_ecef * z_ecef
    G = r * r + (1 - consts.E1SQ) * z_ecef * z_ecef - consts.E1SQ * E1SQ
    C = (consts.E1SQ * consts.E1SQ * F * r * r) / (pow(G, 3))
    S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C + EPSILON))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = np.sqrt(1 + 2 * consts.E1SQ * consts.E1SQ * P)
    r_0 =  -(P * consts.E1SQ * r) / (1 + Q) + np.sqrt(0.5 * consts.A * consts.A*(1 + 1.0 / Q) - \
          P * (1 - consts.E1SQ) * z_ecef * z_ecef / (Q * (1 + Q)) - 0.5 * P * r * r)
    U = np.sqrt(pow((r - consts.E1SQ * r_0), 2) + z_ecef * z_ecef)
    V = np.sqrt(pow((r - consts.E1SQ * r_0), 2) + (1 - consts.E1SQ) * z_ecef * z_ecef)
    Z_0 = consts.B * consts.B * z_ecef / (consts.A * V)
    h = U * (1 - consts.B * consts.B / ((consts.A * V)))
    lat = ratio*np.arctan((z_ecef + consts.E2SQ * Z_0) / (r))
    lon = ratio*np.arctan2(y_ecef, x_ecef)
    # stack the new columns and return to the original shape
    geodetic = np.column_stack((lat, lon, h))
    if input_shape[0]==3:
        geodetic = np.vstack((lat, lon, h))
    return geodetic

class LocalCoord(object):
    """Class for conversions to NED (North-East-Down).

    Attributes
    ----------
    init_ecef : np.ndarray
        ECEF of origin of NED.
    ned_to_ecef_matrix : np.ndarray
        Rotation matrix to convert from NED to ECEF.
    ecef_to_ned_matrix : np.ndarray
        Rotation matrix to convert from ECEF to NED.

    Notes
    -----
    Based on code from https://github.com/commaai/laika.

    """

    def __init__(self, init_geodetic, init_ecef):
        self.init_ecef = init_ecef
        if init_geodetic.shape[0]==3:
            lat = (np.pi/180.)*init_geodetic[0, 0]
            lon = (np.pi/180.)*init_geodetic[1, 0]
        elif init_geodetic.shape[1]==3:
            lat = (np.pi/180.)*init_geodetic[0, 0]
            lon = (np.pi/180.)*init_geodetic[0, 1]
        else:  # pragma: no cover
            raise ValueError('init_geodetic has incorrect size', len(init_geodetic),
                            ' must be of size 3')
        self.ned_to_ecef_matrix = np.array([[-np.sin(lat)*np.cos(lon), -np.sin(lon), -np.cos(lat)*np.cos(lon)],
                                            [-np.sin(lat)*np.sin(lon), np.cos(lon), -np.cos(lat)*np.sin(lon)],
                                            [np.cos(lat), 0, -np.sin(lat)]])
        self.ecef_to_ned_matrix = self.ned_to_ecef_matrix.T

    @classmethod
    def from_geodetic(cls, init_geodetic):
        """Instantiate class using NED origin in geodetic coordinates.

        Parameters
        ----------
        init_geodetic : np.ndarray
            Float with WGS-84 LLA coordinates of the NED origin

        Returns
        -------
        local_coord : LocalCoord
            Instance of LocalCoord object with corresponding NED origin.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        init_ecef = geodetic_to_ecef(init_geodetic)
        local_coord = LocalCoord(init_geodetic, init_ecef)
        return local_coord

    @classmethod
    def from_ecef(cls, init_ecef):
        """Instantiate class using the NED origin in ECEF coordinates.

        Parameters
        ----------
        init_ecef : np.ndarray
            Float with ECEF coordinates of the NED origin.

        Returns
        -------
        local_coord : LocalCoord
            Instance of LocalCoord object with corresponding NED origin.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        init_geodetic = ecef_to_geodetic(init_ecef)
        local_coord = LocalCoord(init_geodetic, init_ecef)
        return local_coord

    def ecef_to_ned(self, ecef):
        """Convert ECEF position vectors to NED position vectors.

        Parameters
        ----------
        ecef : np.ndarray
            Float with ECEF position vectors.

        Returns
        -------
        ned : np.ndarray
            Converted NED position vectors.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ecef = np.array(ecef)
        # Convert to column vectors for calculation before returning in the same shape as the input
        input_shape = ecef.shape
        if input_shape[0] == 3:
            ned =  np.matmul(self.ecef_to_ned_matrix, (ecef - np.reshape(self.init_ecef, [3, -1])))
        elif input_shape[1]==3:
            ned = np.matmul(self.ecef_to_ned_matrix, (ecef.T - np.reshape(self.init_ecef, [3, -1])))
            ned = np.transpose(ned)
        return ned

    def ecef_to_nedv(self, ecef):
        """Convert ECEF free vectors to NED free vectors.

        Parameters
        ----------
        ecef : np.ndarray
            Float with free vectors in the ECEF frame of reference.

        Returns
        -------
        ned : np.ndarray
            Converted free vectors in the NED frame of reference.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ecef = np.array(ecef)
        # Convert to column vectors for calculation before returning in the same shape as the input
        input_shape = ecef.shape
        if input_shape[0] == 3:
            ned =  np.matmul(self.ecef_to_ned_matrix, ecef)
        elif input_shape[1]==3:
            ned = np.matmul(self.ecef_to_ned_matrix, ecef.T)
            ned = ned.T
        return ned

    def ned_to_ecef(self, ned):
        """Convert NED position vectors to ECEF position vectors.

        Parameters
        ----------
        ned : np.ndarray
            Float with position vectors in the NED frame of reference.

        Returns
        -------
        ecef : np.ndarray
            Converted position vectors in the ECEF frame of reference.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ned = np.array(ned)
        # Convert to column vectors for calculation before returning in the same shape as the input
        input_shape = ned.shape
        if input_shape[0] == 3:
            ecef =  np.matmul(self.ned_to_ecef_matrix, ned) + np.reshape(self.init_ecef, [3, -1])
        elif input_shape[1]==3:
            ecef = np.matmul(self.ned_to_ecef_matrix, ned.T) + np.reshape(self.init_ecef, [3, -1])
            ecef = ecef.T
        return ecef

    def ned_to_ecefv(self, ned):
        """Convert NED free vectors to ECEF free vectors.

        Parameters
        ----------
        ned : np.ndarray
            Float with free vectors in the NED frame of reference.

        Returns
        -------
        ecef : np.ndarray
            Converted free vectors in the ECEF frame of reference.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ned = np.array(ned)
        # Convert to column vectors for calculation before returning in the same shape as the input
        input_shape = ned.shape
        if input_shape[0] == 3:
            ecef =  np.matmul(self.ned_to_ecef_matrix, ned)
        elif input_shape[1]==3:
            ecef = np.matmul(self.ned_to_ecef_matrix, ned.T)
            ecef = ecef.T
        return ecef

    def geodetic_to_ned(self, geodetic):
        """Convert geodetic position vectors to NED position vectors.

        Parameters
        ----------
        geodetic : np.ndarray
            Float with geodetic position vectors.

        Returns
        -------
        ned : np.ndarray
            Converted NED position vectors.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ecef = geodetic_to_ecef(geodetic)
        ned = self.ecef_to_ned(ecef)
        return ned

    def ned_to_geodetic(self, ned):
        """Convert geodetic position vectors to NED position vectors.

        Parameters
        ----------
        ned : np.ndarray
            Float with NED position vectors.

        Returns
        -------
        geodetic : np.ndarray
            Converted geodetic position vectors.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ecef = self.ned_to_ecef(ned)
        geodetic = ecef_to_geodetic(ecef)
        return geodetic

def ecef_to_el_az(rx_pos, sv_pos):
    """Calculate the elevation and azimuth from receiver to satellites.

    Vectorized to be able to be able to output the elevation and azimuth
    for multiple satellites at the same time.

    Parameters
    ----------
    rx_pos : np.ndarray
        1x3 vector containing ECEF [X, Y, Z] coordinate of receiver
    sv_pos : np.ndarray
        3xN array  containing ECEF [X, Y, Z] coordinates of satellites

    Returns
    -------
    el_az : np.ndarray
        2XN array containing the elevation and azimuth from the
        receiver to the requested satellites. Elevation and azimuth are
        given in decimal degrees.

    Notes
    -----
    Code based on method by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    """

    # conform receiver position to correct shape
    rx_pos = np.atleast_2d(rx_pos)
    if rx_pos.shape == (1,3):
        rx_pos = rx_pos.T
    if rx_pos.shape != (3, 1):
        raise RuntimeError("Receiver ECEF position must be a " \
                          + "np.ndarray of shape 3x1.")

    # conform satellite position to correct shape
    sv_pos = np.atleast_2d(sv_pos)
    if sv_pos.shape[0] != 3:
        raise RuntimeError("Satellite ECEF position(s) must be a " \
                          + "np.ndarray of shape 3xN.")

    # Convert the receiver location to WGS84
    rx_lla = ecef_to_geodetic(rx_pos)

    # Create variables with the latitude and longitude in radians
    rx_lat, rx_lon = np.deg2rad(rx_lla[:2,0])

    # Create the 3 x 3 transform matrix from ECEF to VEN
    ecef_to_ven = np.array([[ np.cos(rx_lat)*np.cos(rx_lon),
                              np.cos(rx_lat)*np.sin(rx_lon),
                              np.sin(rx_lat)],
                            [-np.sin(rx_lon),
                              np.cos(rx_lon),
                              0.     ],
                            [-np.sin(rx_lat)*np.cos(rx_lon),
                             -np.sin(rx_lat)*np.sin(rx_lon),
                              np.cos(rx_lat)]])

    # Replicate the rx_pos array to be the same size as the satellite array
    rx_array = np.tile(rx_pos,(1, sv_pos.shape[1]))

    # Calculate the normalized pseudorange for each satellite
    pseudorange = (sv_pos - rx_array) / np.linalg.norm(sv_pos - rx_array,
                                             axis=0, keepdims=True)

    # Perform the transform of the normalized pseudorange from ECEF to VEN
    p_ven = np.dot(ecef_to_ven, pseudorange)
    # Calculate elevation and azimuth in degrees
    el_az = np.zeros([2, sv_pos.shape[1]])
    el_az[0,:] = np.rad2deg((np.pi/2. - np.arccos(p_ven[0,:])))
    el_az[1,:] = np.rad2deg(np.arctan2(p_ven[1,:],p_ven[2,:]))

    # wrap from 0 to 360
    while np.any(el_az[1, :] < 0):
        el_az[1, :][el_az[1, :] < 0] += 360

    return el_az


# ==========================================================================
# algorithms/snapshot.py
# ==========================================================================
def wls(rx_est_m, pos_sv_m, corr_pr_m, weights = None,
        only_bias = False, tol = 1e-7, max_count = 20, sv_rx_time=False):
    """Weighted least squares solver for GNSS measurements.

    The option for only_bias allows the user to only calculate the clock
    bias if the receiver position is already known. Only the bias term
    in rx_est_m will be updated if only_bias is set to True.

    Parameters
    ----------
    rx_est_m : np.ndarray
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters in an
        array with shape (4 x 1) and the following order:
        x_rx_m, y_rx_m, z_rx_m, b_rx_m.
    pos_sv_m : np.ndarray
        Satellite ECEF positions as an array of shape [# svs x 3] where
        the columns contain in order x_sv_m, y_sv_m, and z_sv_m.
    corr_pr_m : np.ndarray
        Corrected pseudoranges for all satellites with shape of
        [# svs x 1]
    weights : np.array
        Weights as an array of shape [# svs x 1] where the column
        is in the same order as pos_sv_m and corr_pr_m
    only_bias : bool
        If True, then only the receiver clock bias is estimated.
        Otherwise, both position and clock bias are estimated.
    tol : float
        Tolerance used for the convergence check.
    max_count : int
        Number of maximum iterations before process is aborted and
        solution returned.
    sv_rx_time : bool
        Flag to indicate if the satellite positions at the time of
        transmission should be used as is or if they should be transformed
        to the ECEF frame of reference at the time of reception. For real
        measurements, use ``sv_rx_time=False`` to account for the Earth's
        rotation and convert SV positions from the ECEF frame at the time
        of signal transmission to the ECEF frame at the time of signal
        reception. If the SV positions should be used as is, set
        ``sv_rx_time=True`` to indicate that the given positions are in
        the ECEF frame of reference for when the signals are received.
        By default, ``sv_rx_time=False``.

    Returns
    -------
    rx_est_m : np.ndarray
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters in an
        array with shape (4 x 1) and the following order:
        x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    Notes
    -----
    This function internally updates the used SV position to account for
    the time taken for the signal to travel to the Earth from the GNSS
    satellites.
    Since the SV and receiver positions are calculated in an ECEF frame
    of reference, which is moving with the Earth's rotation, the reference
    frame is slightly (about 30 m along longitude) different when the
    signals are received than when the signals were transmitted. Given
    the receiver's position is estimated when the signal is received,
    the SV positions need to be updated to reflect the change in the
    frame of reference in which their position is calculated.

    This update happens after every Gauss-Newton update step and is
    adapted from [1]_.

    References
    ----------
    .. [1] https://github.com/google/gps-measurement-tools/blob/master/opensource/FlightTimeCorrection.m

    """

    rx_est_m = rx_est_m.copy() # don't change referenced value

    count = 0
    # Store the SV position at the original receiver time.
    # This position will be modified by the time taken by the signal to
    # travel to the receiver.
    rx_time_pos_sv_m = pos_sv_m.copy()
    num_svs = pos_sv_m.shape[0]
    if num_svs < 4 and not only_bias:
        raise RuntimeError("Need at least four satellites for WLS.")
    pos_x_delta = np.inf*np.ones((4,1))

    # load weights
    if weights is None:
        weight_matrix = np.eye(num_svs)
    elif isinstance(weights, np.ndarray):
        if weights.ndim != 0 and weights.size == num_svs:
            weight_matrix = np.eye(num_svs)*weights
        else:
            raise TypeError("WLS weights must be the same length"\
                            + " as number of satellites.")
    else:
        raise TypeError("WLS weights must be None or np.ndarray.")

    while np.linalg.norm(pos_x_delta) > tol:
        pos_rx_m = np.tile(rx_est_m[0:3,:].T, (num_svs, 1))

        gt_r_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                                 keepdims = True)

        if only_bias:
            geometry_matrix = np.ones((num_svs,1))
        else:
            geometry_matrix = np.ones((num_svs,4))
            geometry_matrix[:,:3] = np.divide(pos_rx_m - pos_sv_m,
                                              gt_r_m.reshape(-1,1))

        # assumes the use of corrected pseudoranges with the satellite
        # clock bias already removed
        pr_delta = corr_pr_m - gt_r_m - rx_est_m[3,0]
        try:
            pos_x_delta = np.linalg.pinv(geometry_matrix.T @ weight_matrix @ geometry_matrix) \
                        @ geometry_matrix.T @ weight_matrix @ pr_delta
        except np.linalg.LinAlgError as exception:
            print(exception)
            break

        if only_bias:
            rx_est_m[3,0] += pos_x_delta[0,0]
        else:
            rx_est_m += pos_x_delta

        if not sv_rx_time:
            # Update the satellite positions based on the time taken for
            # the signal to reach the Earth and the satellite clock bias.
            delta_t = (corr_pr_m.reshape(-1) - rx_est_m[3,0])/consts.C
            dtheta = consts.OMEGA_E_DOT*delta_t
            pos_sv_m[:, 0] = np.cos(dtheta)*rx_time_pos_sv_m[:,0] + \
                             np.sin(dtheta)*rx_time_pos_sv_m[:,1]
            pos_sv_m[:, 1] = -np.sin(dtheta)*rx_time_pos_sv_m[:,0] + \
                              np.cos(dtheta)*rx_time_pos_sv_m[:,1]

        count += 1

        if count >= max_count:
            warnings.warn("Newton Raphson did not converge.", RuntimeWarning)
            break

    return rx_est_m

# ==========================================================================
# utils/gnss_models.py
# ==========================================================================
def _calculate_iono_delay(gps_tow, iono_params, rx_ecef, sv_pos,     # CHANGED: inputs
                          constellation="gps"):
    """Calculate the ionospheric delay in pseudorange using the Klobuchar
    model Section 5.3.2 [1]_.

    Parameters
    ----------
    gps_tow : float
        GPS time of week [s].                                 (CHANGED)
    iono_params : np.ndarray
        Ionospheric atmospheric delay parameters for Klobuchar model,
        passed in 2x4 array, use None if not available.
    rx_ecef : np.ndarray
        3x1 receiver position in ECEF frame of reference [m], use None
        if not available.
    sv_pos : np.ndarray
        3xN satellite ECEF positions [m].                   (CHANGED)
    constellation : string
        Constellation used for the ionospheric parameters addition.

    Returns
    -------
    iono_delay : np.ndarray
        Estimated delay caused by the ionosphere [m].

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    References
    ----------
    ..  [1] Misra, P. and Enge, P,
        "Global Positioning System: Signals, Measurements, and Performance."
        2nd Edition, Ganga-Jamuna Press, 2006.

    """
    #Reshape receiver position to 3x1
    rx_ecef = np.reshape(rx_ecef, [3,1])

    # Determine the satellite locations
    sv_pos = np.asarray(sv_pos, dtype=float)    # CHANGED: positions passed in
    sv_pos = sv_pos.reshape(3, -1)
    el_az = ecef_to_el_az(rx_ecef, sv_pos)
    el_r = np.deg2rad(el_az[0, :])
    az_r = np.deg2rad(el_az[1, :])

    # Calculate the WGS-84 latitude/longitude of the receiver
    wgs_llh = ecef_to_geodetic(rx_ecef)
    lat_r = np.deg2rad(wgs_llh[0, :])
    lon_r = np.deg2rad(wgs_llh[1, :])

    # Parse the ionospheric parameters
    alpha = iono_params[constellation][0,:]
    beta = iono_params[constellation][1,:]

    # Calculate the psi angle
    psi = 0.1356/(el_r+0.346) - 0.0691

    # Calculate the ionospheric geodetic latitude
    lat_i = lat_r + psi * np.cos(az_r)

    # Make sure values are in bounds
    ind = np.argwhere(np.abs(lat_i) > 1.3090)
    if len(ind) > 0:
        lat_i[ind] = 1.3090 * np.sign(lat_i[ind])  # pragma: no cover
    # Calculate the ionospheric geodetic longitude
    lon_i = lon_r + psi * np.sin(az_r)/np.cos(lat_i)

    # Calculate the solar time corresponding to the gps_tow
    solar_time = 1.3751e4 * lon_i + gps_tow

    # Make sure values are in bounds
    solar_time = np.mod(solar_time,86400)

    # Calculate the geomagnetic latitude (semi-circles)
    lat_m = (lat_i + 2.02e-1 * np.cos(lon_i - 5.08))/np.pi
    # Calculate the period
    period = beta[0]+beta[1]*lat_m+beta[2]*lat_m**2+beta[3]*lat_m**3

    # Make sure values are in bounds
    ind = np.argwhere(period < 72000).flatten()
    if len(ind) > 0:
        period[ind] = 72000  # pragma: no cover

    # Calculate the local time angle
    theta = 2*np.pi*(solar_time - 50400) / period

    # Calculate the amplitude term
    amp = (alpha[0]+alpha[1]*lat_m+alpha[2]*lat_m**2+alpha[3]*lat_m**3)

    # Make sure values are in bounds
    ind = np.argwhere(amp < 0).flatten()
    if len(ind) > 0:
        amp[ind] = 0  # pragma: no cover

    # Calculate the slant factor
    slant_fact = 1.0 + 5.16e-1 * (1.6755-el_r)**3

    # Calculate the ionospheric delay
    iono_delay = slant_fact * 5.0e-9
    ind = np.argwhere(np.abs(theta) < np.pi/2.).flatten()
    if len(ind) > 0:
        iono_delay[ind] = slant_fact[ind]* \
            (5e-9+amp[ind]*(1-theta[ind]**2/2.+theta[ind]**4/24.))

    # Convert ionospheric delay to equivalent meters
    iono_delay = consts.C*iono_delay
    return iono_delay
