# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
from scipy.stats import chi2

import pandapower as pp
from pandapower.estimation.ppc_conversion import ExtendedPPCI

a = """
    Perform a Chi^2 test for bad data and topology error detection.

    Parameters:
        net (pp.pandapowerNet): "\
"Pandapower network containing measurements to be tested.
        eppci (ExtendedPPCI): "\
"State estimation result with voltage magnitudes `v` and angles `delta`.
        chi2_prob_false (float, optional): "\
"Significance level for false alarms (default: 0.05).
        chi2_prob_true (float, optional): "\
"Significance level for detection (default: 0.95).

    Returns:
        tuple:
            chi2_max_threshold (float): Upper threshold of the Chi^2 "\
"statistic (quantile at 1 - chi2_prob_false, df = m - n).
            chi2_min_threshold (float): Lower threshold of the Chi^2 "\
"statistic (quantile at 1 - chi2_prob_true, df = m - n).
    """


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
