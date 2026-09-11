# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from datetime import datetime
from typing import overload, Literal

import numpy as np
from typing_extensions import deprecated
from scipy.stats import chi2

from pandapower import pandapowerNet
from pandapower.estimation.algorithm.base import (WLSAlgorithm,
                                                  WLSZeroInjectionConstraintsAlgorithm,
                                                  IRWLSAlgorithm)
from pandapower.estimation.algorithm.lp import LPAlgorithm
from pandapower.estimation.algorithm.optimization import OptAlgorithm
from pandapower.estimation.ppc_conversion import pp2eppci, _initialize_voltage
from pandapower.estimation.results import eppci2pp
from pandapower.estimation.util import set_bb_switch_impedance, reset_bb_switch_impedance

import logging
std_logger = logging.getLogger(__name__)

ALGORITHM_MAPPING = {"wls": WLSAlgorithm,
                     "wls_with_zero_constraint": WLSZeroInjectionConstraintsAlgorithm,
                     "opt": OptAlgorithm,
                     "irwls": IRWLSAlgorithm,
                     "lp": LPAlgorithm,
                     "af-wls": WLSAlgorithm,
                     "af-lp": LPAlgorithm}
ALGORITHM_SE = Literal["wls", "wls_with_zero_constraint", "opt", "irwls", "lp", "af-wls", "af-lp"]
OPT_VAR_DEFAULTS = {
    "estimator": "wls",
    "linprog_method": "highs",
    "wlav": False,
    "with_ortools": True,
    "with_af_constraints": True,
    "af_init_value": .5,
    "af_target_value": None,
    "af_std_value": None
}


def estimate(
        net: pandapowerNet,
        algorithm: ALGORITHM_SE ="wls",
        init: str = "flat",
        tolerance: float = 1e-6,
        maximum_iterations: int = 50,
        zero_injection = "aux_bus",
        fuse_buses_with_bb_switch = "all",
        debug_mode: bool = False,
        **opt_vars
) -> dict[str, object]:
    """
    Wrapper function for WLS state estimation.

    Parameters:
        net (pandapowerNet): The net within this line should be created
        algorithm: Algorithm used for state estimation:

            - "wls": classical weighted least squares (default)
            - "wls_with_zero_constraint": WLS with zero-injection constraints
            - "opt": optimization-based algorithm
            - "irwls": iteratively reweighted WLS (robust estimation)
            - "lp": linear programming–based approach
            - "af-wls": WLS using allocation factors (af) -> set type for sgen, load, etc.
            - "af-lp": LP-based approach using af -> set type for sgen, load, etc.

            The algorithms with allocation factor (af), "af-wls" and "af-lp", internally call upon the classes "wls"
            :class:`WLSAlgorithm` and "lp" :class:`LPAlgorithm`. The allocation factor :math:`\\alpha` is handled in
            `estimation/ppc_conversion.py` and `estimation/algorithm/matrix_base.py`. When creating generators and
            loads, a type must be specified for af.

        init (string): Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all buses, 'results' uses the
            values from *res_bus* if available and 'slack' considers the slack bus voltage (and optionally, angle) as
            the initial values. Default is 'flat'
        tolerance (float): When the maximum state change between iterations is less than tolerance, the process stops.
            Default is 1e-6
        maximum_iterations (integer): Maximum number of iterations. Default is 50
        zero_injection (str, iterable, None): Defines which buses are zero injection bus or the method to identify zero
            injection bus, with 'wls_estimator' virtual measurements will be added, with 'wls_estimator with zero
            constraints' the buses will be handled as constraints

                - None: no bus will be identified as zero injection bus
                - "aux_bus": only aux bus will be identified as zero injection bus
                - "no_inj_bus": aux bus and bus without p,q measurement and without any connected injection \
                    (load, sgen...) will be identified as zero injection bus
                - "zero_pwr_bus": aux bus and all bus without p,q measurement that have either no connected injection \
                    (load, sgen...) or a connected injection (load, sgen...) equal to zero will be identified as \
                    zero injection bus
                - iterable: the iterable should contain index of the zero injection bus and also aux bus will be \
                    identified as zero-injection bus

        fuse_buses_with_bb_switch (str, iterable, None): Defines how buses with closed bb switches should be handled,
            if fuse buses will only fuse to one for calculation, if not fuse, an auxiliary bus and auxiliary line will
            be automatically added to the network to make the buses with different p,q injection measurements
            identifiable

                - "all": all buses with bb-switches will be fused, the same as the default behaviour in load flow
                - None: buses with bb-switches and individual p,q measurements will be reconfigurated \
                    by auxiliary elements
                - iterable: the iterable should contain the index of buses to be fused, the behaviour is contigous \
                    e.g. if one of the bus among the buses connected through bb switch is given, then all of them will \
                    still be fused

        debug_mode (bool):
            If ``True``, additional diagnostic information is logged, including the current state update norm,
            objective function value, and possible ill-conditioning of the gain matrix (only for WLS).

    Keyword Args:
        estimator (Literal["wls", "smgm"], "wls"): Used with ``algorithm="irwls"``.
        linprog_method (Literal["highs", "highs-ds", "highs-ipm"], "highs"): supported for algorithm='lav'
        wlav (bool, False): Perform LAV with weights.
        with_ortools (bool, True):
            Alternative solver to scipy for (W)LAV using `OR-Tools solver <https://github.com/google/or-tools>`_
        with_af_constraints: Constraints for allocation factors between 0 and 1 for (W)LAV algorithm.
        af_init_value (float | np.ndarray, .5):
            Initial values for the af in E vector. One Value for all or an array for explicite. The
            value 0.5 is default.
        af_target_value (float | np.ndarray | None, None):
            Optional pseudo-measurements (soft priors) for the af. Can be a scalar, a numpy array or None. Acts as a
            preferred value for the af (e.g. ``alpha=0.4``) in weakly observable cases and regularizes the estimation
            toward this value, but does not enforce it if it conflicts with the overall optimization of the state
            estimator. Specific value 0.4 used in a grid-operator use case.
        af_std_value (float | np.ndarray | None, None):
            Optional standard deviation(s) corresponding to `af_target_value`. Can be a scalar, a numpy array or None.
            Controls how strongly the pseudo-measurements for the af (e.g. 0.15) influence the
            estimation: smaller values imply a stronger pull toward `af_target_value`, larger values make the
            prior weaker. Specific value 0.15 used in a grid-operator use case.

    Returns:
        dict:
            Dictionary with state estimation results. The exact content depends on the chosen algorithm. For
            ``algorithm`` in ``["wls", "af-wls", "lp", "af-lp"]`` the dictionary contains:

                - ``"success"`` (bool): Flag indicating whether the state estimation finished successfully.
                - ``"num_iterations"`` (int): Number of iterations performed by the solver.
                - ``"objective_function_value"`` (float): Final value of the objective function.
                - ``"allocation_factors"`` (DataFrame): Estimated af (only meaningful for AF-based algorithms).
                - ``"time"`` (str): Timestamp of the estimation in the format ``"YYYY-MM-DD HH:MM:SS"``.

            For all other algorithms the dictionary contains only:

                - ``"success"`` (bool): Flag indicating whether the state estimation finished successfully.
    """
    if algorithm not in ALGORITHM_MAPPING:
        raise UserWarning("Algorithm {} is not a valid estimator".format(algorithm))

    se = StateEstimation(net, tolerance, maximum_iterations, algorithm=algorithm)
    v_start, delta_start = _initialize_voltage(net, init)

    return se.estimate(v_start=v_start,
                       delta_start=delta_start,
                       zero_injection=zero_injection,
                       fuse_buses_with_bb_switch=fuse_buses_with_bb_switch,
                       debug_mode=debug_mode,
                       init=init,
                       tolerance=tolerance,
                       maximum_iterations=maximum_iterations,
                       **opt_vars
                       )


def remove_bad_data(
        net,
        init="flat",
        tolerance=1e-6,
        maximum_iterations=10,
        calculate_voltage_angles=True,
        rn_max_threshold=3.0
) -> bool:
    """
    Wrapper function for bad data removal.

    Parameters:
        net (pandapowerNet): The net within this line should be created
        init (string): Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all buses, 'results' uses the
            values from *res_bus_est* if available and 'slack' considers the slack bus voltage (and optionally, angle)
            as the initial values. Defaults to 'flat'
        tolerance (float): When the maximum state change between iterations is less than tolerance, the process stops.
            Defaults to 1e-6
        maximum_iterations (integer): Maximum number of iterations. Default is 10
        calculate_voltage_angles (boolean): Take into account absolute voltage angles and phase shifts in transformers,
            if init is 'slack'. Defaults to True
        rn_max_threshold (float): Identification threshold to determine if the largest normalized residual reflects a
            bad measurement Defaults to 3.0

    Returns:
        bool: Was the state estimation successful?
    """
    wls_se = StateEstimation(net, tolerance, maximum_iterations, algorithm="wls")
    v_start, delta_start = _initialize_voltage(net, init)
    return wls_se.perform_rn_max_test(v_start, delta_start, calculate_voltage_angles, rn_max_threshold)


def chi2_analysis(
        net,
        init="flat", tolerance=1e-6,
        maximum_iterations=10,
        calculate_voltage_angles=True,
        chi2_prob_false=0.05
) -> bool | None:
    """
    Wrapper function for the chi-squared test. Only for WLS

    Parameters:
        net (pandapowerNet): The net within this line should be created.
        init (string): Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all buses, 'results' uses the
            values from *res_bus_est* if available and 'slack' considers the slack bus voltage (and optionally, angle)
            as the initial values. Defaults to 'flat'
        tolerance (float): When the maximum state change between iterations is less than tolerance, the process stops.
            Defaults to 1e-6
        maximum_iterations (integer): Maximum number of iterations. Defaults to 10
        calculate_voltage_angles (boolean): Take into account absolute voltage angles and phase shifts in transformers,
            if init is 'slack'. Defaults to True
        chi2_prob_false (float): probability of error / false alarms. Defaults to 0.05

    Returns:
        bool: Returns true if bad data has been detected
    """
    wls_se = StateEstimation(net, tolerance, maximum_iterations, algorithm="wls")
    v_start, delta_start = _initialize_voltage(net, init)
    return wls_se.perform_chi2_test(v_start, delta_start, calculate_voltage_angles, chi2_prob_false)


class StateEstimation:
    """
    Any user of the estimation module only needs to use the class state_estimation. It contains all relevant functions
    to control and operator the module. Two functions are used to configure the system according to the users needs
    while one function is used for the actual estimation process.
    """

    def __init__(
            self,
            net,
            tolerance=1e-6,
            maximum_iterations=50,
            algorithm: ALGORITHM_SE = "wls",
            logger=None,
            recycle=False
    ):
        self.logger = logger
        if self.logger is None:
            self.logger = std_logger
            # self.logger.setLevel(logging.DEBUG)
        self.net = net
        self.solver = ALGORITHM_MAPPING[algorithm](tolerance,
                                                   maximum_iterations, self.logger)
        self.ppc = None
        self.eppci = None
        self.recycle = recycle
        self.algorithm = algorithm

        # variables for chi^2 / rn_max tests
        self.delta = None
        self.bad_data_present = None

    @overload
    def estimate(self, v_start="flat", delta_start="flat", zero_injection=None,
                 fuse_buses_with_bb_switch="all", debug_mode=False, **opt_vars):
        ...

    @overload
    @deprecated("algorithm should be set via init. Use of algorithm key is deprecated.")
    def estimate(self, v_start="flat", delta_start="flat", zero_injection=None,
                 fuse_buses_with_bb_switch="all", algorithm="wls", debug_mode=False, **opt_vars):
        ...

    def estimate(
            self,
             v_start="flat",
             delta_start="flat",
             zero_injection=None,
             fuse_buses_with_bb_switch="all",
             init: str = "flat",
             tolerance: float = 1e-6,
             maximum_iterations: int = 50,
             debug_mode=False,
             **opt_vars
    ) -> dict[str, object]:
        """
        The function estimate is the main function of the module. It takes the inputs v_start and delta_start to
        initialize the state variables for the estimation process. Usually they can be initialized in a "flat-start"
        condition: All voltages being 1.0 pu and all voltage angles being 0 degrees. In this case, the parameters can be
        left at their default values (None). If the estimation is applied continuously, using the results from the last
        estimation as the starting condition for the current estimation can decrease the amount of iterations needed to
        estimate the current state.
        Returned is a boolean value, which is true after a successful estimation and false otherwise. The resulting
        complex voltage will be written into the pandapower network. The result fields are found res_bus_est of the
        pandapower network.

        Parameters:
            v_start (np.array, shape=(1,), optional):
                Vector with initial values for all voltage magnitudes in p.u.(sorted by bus index)
            delta_start (np.array, shape=(1,), optional):
                Vector with initial values for all voltage angles in degrees (sorted by bus index)
            zero_injection (str, iterable, None):
                Defines which buses are zero injection bus or the method to identify zero injection bus, with
                'wls_estimator' virtual measurements will be added, with 'wls_estimator with zero constraints' the buses
                will be handled as constraints

                    - None: no bus will be identified as zero injection bus
                    - "aux_bus": only aux bus will be identified as zero injection bus
                    - "no_inj_bus": aux bus and bus without p,q measurement and without any connected injection
                        (load, sgen...) will be identified as zero injection bus
                    - "zero_pwr_bus": aux bus and all bus without p,q measurement that have either no connected
                        injection (load, sgen...) or a connected injection (load, sgen...) equal to zero will be
                        identified as zero injection bus
                    - iterable: the iterable should contain index of the zero injection bus and also aux bus will be
                        identified as zero-injection bus

            init (string):
                Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all buses, 'results' uses the values
                from *res_bus* if available and 'slack' considers the slack bus voltage (and optionally, angle) as the
                initial values. Default is 'flat'
            tolerance (float):
                When the maximum state change between iterations is less than tolerance, the process stops. Default is
                1e-6
            maximum_iterations (integer): Maximum number of iterations. Default is 50
            fuse_buses_with_bb_switch (str, iterable, None):
                Defines how buses with closed bb switches should be handled, if fuse buses will only fuse to one for
                calculation, if not fuse, an auxiliary bus and auxiliary line will be automatically added to the network
                to make the buses with different p,q injection measurements identifiable

                    - "all": all buses with bb-switches will be fused, the same as the default behaviour in load flow
                    - None:
                        buses with bb-switches and individual p,q measurements will be reconfigurated by auxiliary
                        elements
                    - iterable:
                        the iterable should contain the index of buses to be fused, the behaviour is contigous e.g. if
                        one of the bus among the buses connected through bb switch is given, then all of them will still
                        be fused

            debug_mode (bool):
                If ``True``, additional diagnostic information is logged, including the current state update norm,
                objective function value, and possible ill-conditioning of the gain matrix (only for WLS).

        Keyword Args:
            tolerance (float):
                When the maximum state change between iterations is less than tolerance, the process stops. Default is
                1e-6.
            maximum_iterations (integer): Maximum number of iterations. Default is 50.


        Returns:
            For "wls", "af-wls", "lp", "af-lp" will give back a dictionary with:

                - "success": self.solver.successful,
                - "num_iterations": self.solver.iterations,
                - "objective_function_value": self.solver.obj_func,
                - "allocation factor": self.solver.af,
                - "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        """
        # check if all parameter are allowed
        for var_name in opt_vars.keys():
            if var_name not in OPT_VAR_DEFAULTS:
                self.logger.warning("Caution! %s is not allowed as parameter" % var_name \
                                    + " for estimate and will be ignored!")

        if self.net is None:
            raise UserWarning("SE Component was not initialized with a network.")

        # change the configuration of the pp net to avoid auto fusing of buses connected
        # through bb switch with elements on each bus if this feature enabled
        bus_to_be_fused = None
        if fuse_buses_with_bb_switch != "all" and not self.net.switch.empty:
            if isinstance(fuse_buses_with_bb_switch, str):
                raise UserWarning("fuse_buses_with_bb_switch parameter is not correctly initialized")
            elif hasattr(fuse_buses_with_bb_switch, "__iter__"):
                bus_to_be_fused = fuse_buses_with_bb_switch
            set_bb_switch_impedance(self.net, bus_to_be_fused)  # runpp() performed

        af_target_value = opt_vars.get("af_target_value", OPT_VAR_DEFAULTS["af_target_value"])
        af_std_value = opt_vars.get("af_std_value", OPT_VAR_DEFAULTS["af_std_value"])

        if (af_target_value is None) != (af_std_value is None):
            raise ValueError("af_target_value and af_std_value are both None or both not None.")
        self.net, self.ppc, self.eppci = pp2eppci(
            self.net,
            v_start=v_start,
            delta_start=delta_start,
            calculate_voltage_angles=True,
            ppc=self.ppc,
            eppci=self.eppci,
            algorithm=self.algorithm,
            init=init,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            zero_injection=zero_injection,
            fuse_buses_with_bb_switch=fuse_buses_with_bb_switch,
            debug_mode=debug_mode,
            estimator=opt_vars.get("estimator", OPT_VAR_DEFAULTS["estimator"]),
            linprog_method=opt_vars.get("linprog_method", OPT_VAR_DEFAULTS["linprog_method"]),
            wlav=opt_vars.get("wlav", OPT_VAR_DEFAULTS["wlav"]),
            with_ortools=opt_vars.get("with_ortools", OPT_VAR_DEFAULTS["with_ortools"]),
            with_af_constraints= opt_vars.get("with_af_constraints", OPT_VAR_DEFAULTS["with_af_constraints"]),
            af_init_value=opt_vars.get("af_init_value", OPT_VAR_DEFAULTS["af_init_value"]),
            af_target_value=af_target_value,
            af_std_value=af_std_value
        )

        # Estimate voltage magnitude and angle with the given estimator
        self.eppci = self.solver.estimate(self.eppci, debug_mode=debug_mode, **opt_vars)

        if self.solver.successful:
            self.net = eppci2pp(self.net, self.ppc, self.eppci)
            if self.algorithm in ["af-wls", "af-lp"]:
                self.net["res_cluster_est"] = self.eppci.clusters
        else:
            self.logger.warning("Estimation failed! Pandapower network failed to update!")

        # clear the aux elements and calculation results created for the substitution of bb switches
        if fuse_buses_with_bb_switch != "all" and not self.net.switch.empty:
            reset_bb_switch_impedance(self.net)

        # if recycle is not wished, reset ppc, ppci
        if not self.recycle:
            self.ppc, self.eppci = None, None
        
        if self.algorithm in ["wls", "af-wls", "lp", "af-lp"]:
            se_results = {
                "success": self.solver.successful,
                "num_iterations": self.solver.iterations,
                "objective_function_value": self.solver.obj_func,
                "allocation_factors": self.solver.af,
                "completion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        else:
            se_results = {"success":self.solver.successful}

        return se_results

    def perform_chi2_test(
            self,
            v_in_out=None,
            delta_in_out=None,
            calculate_voltage_angles :bool = True,
            chi2_prob_false=0.05
    ) -> bool | None:
        """
        The function perform_chi2_test performs a Chi^2 test for bad data and topology error detection.

        The function can be called with the optional input arguments v_in_out and delta_in_out. Then, the Chi^2 test is
        performed after calling the function estimate using them as input arguments. It can also be called without these
        arguments if it is called from the same object with which estimate had been called beforehand. Then, the Chi^2
        test is performed for the states estimated by the funtion estimate and the result, the existence of bad data, is
        given back as a boolean. As a optional argument the probability of a false measurement can be provided
        additionally. For bad data detection, the function perform_rn_max_test is more powerful and should be the
        function of choice. For topology error detection, however, perform_chi2_test should be used.

        Parameters:
            v_in_out (np.array, shape=(1,), optional):
                Vector with initial values for all voltage magnitudes in p.u. (sorted by bus index)
            delta_in_out (np.array, shape=(1,), optional):
                Vector with initial values for all voltage angles in degrees (sorted by bus index)
            calculate_voltage_angles (boolean):
                Take into account absolute voltage angles and phase shifts in transformers, if init is 'slack'.
                Default is True
            chi2_prob_false (float): probability of error / false alarms (standard value: 0.05)

        Returns: True if bad data has been detected
        """
        # perform SE
        self.estimate(v_in_out, delta_in_out, calculate_voltage_angles)

        # Performance index J(hx)
        J = np.dot(self.solver.r.T, np.dot(self.solver.R_inv, self.solver.r))

        # Number of measurements
        m = len(self.net.measurement)

        # Number of state variables (the -1 is due to the reference bus)
        n = len(self.solver.eppci.v) + len(self.solver.eppci.delta) - 1

        # Chi^2 test threshold
        test_thresh = chi2.ppf(1 - chi2_prob_false, m - n)

        # Print results
        self.logger.debug("Result of Chi^2 test:")
        self.logger.debug("Number of measurements: %d" % m)
        self.logger.debug("Number of state variables: %d" % n)
        self.logger.debug("Performance index: %.2f" % J.item())
        self.logger.debug("Chi^2 test threshold: %.2f" % test_thresh)

        if J <= test_thresh:
            self.bad_data_present = False
            self.logger.debug("Chi^2 test passed. No bad data or topology error detected.")
        else:
            self.bad_data_present = True
            self.logger.debug("Chi^2 test failed. Bad data or topology error detected.")
        return self.bad_data_present if self.solver.successful else None

    def perform_rn_max_test(
            self,
            v_in_out=None,
            delta_in_out=None,
            calculate_voltage_angles=True,
            rn_max_threshold=3.0
    ) -> bool:
        """
        The function perform_rn_max_test performs a largest normalized residual test for bad data identification and
        removal.

        It takes two input arguments: v_in_out and delta_in_out. These are the initial state variables for the combined
        estimation and bad data identification and removal process. They can be initialized as described above, e.g.,
        using a "flat" start. In an iterative process, the function performs a state estimation, identifies a bad data
        measurement, removes it from the set of measurements (only if the rn_max threshold is violated by the largest
        residual of all measurements, which can be modified), performs the state estimation again, and so on and so
        forth until no further bad data measurements are detected.

        Parameters:
            v_in_out (np.array, shape=(1,), optional):
                Vector with initial values for all voltage magnitudes in p.u. (sorted by bus index)
            delta_in_out (np.array, shape=(1,), optional):
                Vector with initial values for all voltage angles in degrees (sorted by bus index)
            calculate_voltage_angles (boolean):
                Take into account absolute voltage angles and phase shifts in transformers, if init is 'slack'. Default
                is True
            rn_max_threshold (float):
                Identification threshold to determine if the largest normalized residual reflects a bad measurement
                (standard value of 3.0)

        Returns: True if all bad data could be removed.
        """
        num_iterations = 0

        while num_iterations <= 10:
            # Estimate the state with bad data identified in previous iteration
            # removed from set of measurements:
            self.estimate(v_in_out, delta_in_out, calculate_voltage_angles)

            # Try to remove the bad data
            try:
                # Error covariance matrix:
                R = np.linalg.inv(self.solver.R_inv)

                # for future debugging: this line's results have changed with the ppc
                # overhaul in April 2017 after commit 9ae5b8f42f69ae39f8c8cf (which still works)
                # there are differences of < 1e-10 for the Omega entries which cause
                # the function to work far worse. As of now it is unclear if it's just numerical
                # accuracy to blame or an error in the code. a sort in the ppc creation function
                # was removed which caused this issue
                # Covariance matrix of the residuals: \Omega = S*R = R - H*G^(-1)*H^T
                # (S is the sensitivity matrix: r = S*e):
                Omega = R - np.dot(self.solver.H, np.dot(np.linalg.inv(self.solver.Gm), self.solver.H.T))

                # Diagonalize \Omega:
                Omega = np.diag(np.diag(Omega))

                # Compute square root (|.| since some -0.0 produced nans):
                Omega = np.sqrt(np.absolute(Omega))

                OmegaInv = np.linalg.inv(Omega)

                # Compute normalized residuals (r^N_i = |r_i|/sqrt{Omega_ii}):
                rN = np.dot(OmegaInv, np.absolute(self.solver.r))

                if float(max(rN)) <= rn_max_threshold:
                    self.logger.debug("Largest normalized residual test passed. No bad data detected.")
                    return True
                else:
                    self.logger.debug(
                        f"Largest normalized residual test failed ({max(rN).item():.1f} > {rn_max_threshold:.1f})."
                    )

                    # Identify bad data: Determine index corresponding to max(rN):
                    idx_rN = np.argsort(rN, axis=0)[-1]

                    # Determine pandapower index of measurement to be removed:
                    meas_idx = self.solver.pp_meas_indices[idx_rN]

                    # Remove bad measurement:
                    self.logger.debug(f"Removing measurement: {self.net.measurement.loc[meas_idx].values[0]}")
                    self.net.measurement = self.net.measurement.drop(meas_idx)
                    self.logger.debug("Bad data removed from the set of measurements.")

            except np.linalg.LinAlgError:
                self.logger.error(
                    "A problem appeared while using the linear algebra methods. Check and change the measurement set."
                )
                return False

            self.logger.debug(f"rN_max identification threshold: {rn_max_threshold:.2f}")
            num_iterations += 1
        return False
