# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import chi2

import pandapower as pp
from pandapower.estimation.ppc_conversion import ExtendedPPCI


def perform_chi2_test(
        net: pp.pandapowerNet,
        eppci: ExtendedPPCI,
        confidence_level: float = 0.05,
):
    """
    The function perform_chi2_test performs a Chi^2 test for bad data and topology error
    detection. The Chi^2 test is optionally called and performed after state estimation is run.
    The test uses the number of measurements and states to return a range defined by the outputs j_min/j_max.
    If the value of the objective function is within the calculated range, no bad data alarm should be raised.
    The confidence of not raising bad data alarm is equal to the confidence_level value set.


    INPUT:
        net (pp.pandapowerNet) - Pandapower network containing measurements to be tested.
        eppci (ExtendedPPCI) - Bus branch model with measurements.

    OPTIONAL:
        confidence_level (float)  - default value = 0.05.

    OUTPUT:
        chi2_max_threshold (float) - Upper threshold of the Chi^2 statistic (quantile at 1 - confidence_level, df = m - n).
        chi2_min_threshold (float) - Lower threshold of the Chi^2 statistic (quantile at confidence_level, df = m - n).

    """

    # Number of measurements
    m = len(net.measurement)

    # Number of state variables (the -1 is due to the reference bus)
    n = len(eppci.v) + len(eppci.delta) - 1

    # Chi^2 test threshold

    j_max = chi2.ppf(1 - confidence_level, m - n)
    j_min = chi2.ppf(confidence_level, m - n)

    return j_max, j_min


def perform_rn_max_test(
        eppci: ExtendedPPCI,
        H: csr_matrix,
        Gm: csc_matrix,
        r: csc_matrix):
    """
    The function perform_rn_max_test creates a list of normalized residuals in descending order.
    The routine transforms raw measurement residuals into normalized residuals. Each residual
    is divided by its post-fit standard deviation derived from the gain-corrected covariance.
    The function then orders them so the most statistically significant (largest |rᶰ|) measurements appear first,
    ready for bad-data detection or investigative plotting.


    INPUT:
        eppci (ExtendedPPCI)  - Bus branch model with measurements.
        H     (csr_matrix)    –  Measurement Jacobian ∂h/∂x evaluated at the estimated state vector.
        Gm    (csc_matrix)    –  Gain matrix.
        r     (csc_matrix)    –  Vector of raw measurement residuals (measured − calculated).


    OUTPUT:
        rn    (np.ndarray)    – 1-D array of normalized residuals sorted by descending |rᶰ|.

    """

    R = (eppci.r_cov ** 2)

    S = R - (H @ np.linalg.inv(Gm.toarray()) @ H.T)

    Omega = S @ R

    Omega = np.sqrt(np.absolute(Omega))
    # Compute normalized residuals (r^N_i = |r_i|/sqrt{Omega_ii}):
    rN = np.abs(r.toarray()) / Omega[:, np.newaxis]
    rN = np.sort(rN, axis=0)
    rN = rN[::-1, :]

    return rN
