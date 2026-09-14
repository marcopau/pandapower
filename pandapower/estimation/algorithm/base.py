# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

__all__ = ["WLSAlgorithm", "WLSZeroInjectionConstraintsAlgorithm", "IRWLSAlgorithm", "ESTIMATOR_SE_TYPES"]

import numpy as np
import pandas as pd
from typing import Literal

from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve, norm, inv


from pandapower.estimation.algorithm.estimator import BaseEstimatorIRWLS, get_estimator
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra, BaseAlgebraZeroInjConstraints
from pandapower.estimation.idx_bus import ZERO_INJ_FLAG, P, P_STD, Q, Q_STD
from pandapower.estimation.ppc_conversion import ExtendedPPCI
from pandapower.pypower.idx_bus import bus_cols

import logging
std_logger = logging.getLogger(__name__)
std_logger.setLevel(logging.DEBUG)

ESTIMATOR_SE_TYPES = Literal["wls", "smgm"]

class BaseAlgorithm:
    def __init__(self, tolerance: float, maximum_iterations: int, logger: logging.Logger = std_logger) -> None:
        """
        Initialize the base algorithm.

        Parameters:
            tolerance: Convergence threshold value.
            maximum_iterations: maximum number of iterations allowed for converge.
            logging.Logger logger: Logger instance

        """
        self.tolerance = tolerance
        self.max_iterations = maximum_iterations
        self.logger = logger
        self.successful = False  # state estimation converges, true if current_error <= self.tolerance -> end
        self.iterations = None  # number of iterations after estimation

        # Parameters for estimate
        self.eppci = None  # central data container: net, measurements, z-vector, etc.
        self.pp_meas_indices = None  # mapping between pandapower measurements and z-vector

    def check_observability(self) -> None:
        """
        Check the basic measurement count criterion for observability.

        This verifies whether the number of measurements is at least as large as the number of state variables, i.e.
        approximately 2N - number_of_slacks.
        """

        if self.eppci is None:
            raise ValueError("eppci must be initialized before checking observability.")

        num_slacks = sum(~self.eppci.non_slack_bus_mask)  # ~ convert false (slack) -> true (slack) and sums true values
        measurements_required = 2 * self.eppci["bus"].shape[0] - num_slacks
        measurements_available = len(self.eppci.z)

        if measurements_available < measurements_required:
            self.logger.error("System is not observable (cancelling)")
            self.logger.error(
                f"Measurements available: {measurements_available}. "
                f"Measurements required: {measurements_required}"
            )
            raise UserWarning(
                f"Measurements available: {measurements_available}. "
                f"Measurements required: {measurements_required}"
            )

    def check_result(self, current_error: float, cur_it: int) -> None:
        """
        Checks termination condition.

        Check current error with the threshold value (tolerance). If the termination condition is met, then
        self.successful is set to True and state estimation is over.

        Parameters:
            current_error: Current error value.
            cur_it: Current iteration number.
        """
        # print output for results
        if current_error <= self.tolerance:
            self.successful = True
            self.logger.debug(
                f"State Estimation successful ({cur_it:d} iterations)"
            )
        else:
            self.successful = False
            self.logger.debug(
                f"State Estimation not successful ({cur_it:d}/{self.max_iterations:d} iterations)"
            )

    def initialize(self, eppci: ExtendedPPCI) -> None:
        """
        Add eppci data container to class parameter and check if the power-gird is observable.

        Parameters:
            eppci: central data container with net, measurements, z-vector, etc.

        """
        # Check observability
        self.eppci = eppci
        self.pp_meas_indices = eppci.pp_meas_indices
        self.check_observability()

    def estimate(self, eppci: ExtendedPPCI, **kwargs):
        raise NotImplementedError("This method must be implemented in the subclass!")


class WLSAlgorithm(BaseAlgorithm):
    def __init__(self, tolerance: float, maximum_iterations: int, logger: logging.Logger = std_logger) -> None:
        """
        Initialize the wls algorithm for state estimation.

        Parameters:
            tolerance: Convergence threshold value.
            maximum_iterations: maximum number of iterations allowed for converge.
            logger: Logger instance
        """
        # Initialize base algorithm
        super(WLSAlgorithm, self).__init__(tolerance, maximum_iterations, logger)

        # Parameters for Bad data detection and removing in state_estimation.py -> remove_bad_data()
        self.R_inv = None  # weighting matrix R^{-1}
        self.Gm = None  # gain matrix G
        self.r = None  # residual z-h(x)
        self.H = None  # Jacobian matrix
        self.hx = None  # calculated measurements h(x)
        self.obj_func = None  # objective function J(x)
        self.af: pd.DataFrame | None = None  # calculation results from allocation factor

    def estimate(self, eppci: ExtendedPPCI, debug_mode=False, **kwargs) -> ExtendedPPCI | bool:
        r"""
        Perform augmented weighted least squares state estimation.

        The AF-WLS algorithm estimates both the classical electrical state variables and additional allocation-factor or
        cluster variables. Therefore, the state vector is assumed to have the form
        :math:`y = [\theta, V, \alpha_{1}, …, \alpha_{k}]^\top` , where :math:`\theta` and :math:`V` are the usual
        voltage angle and magnitude state variables, while :math:`\alpha_{i}` are cluster/allocation-factor variables.

        In each iteration, the nonlinear measurement model is linearized around the current state estimate:

        .. math::
            r = z - h(E)
        .. math::
           r_{\text{new}} \approx r - H\,\Delta E

        The weighted least squares update is then computed by solving

        .. math::
            (H^\mathsf{T} R^{-1} H)\,\Delta E = H^\mathsf{T} R^{-1} r

        without explicitly inverting the gain matrix. At the end of the estimation, the augmented state vector is split
        into:

            - the electrical state vector, which is written back via :func:`eppci.update_E`
            - the cluster/allocation-factor vector, which is stored in :func:`eppci.clusters`

        Parameters:
            eppci:
                Central data container containing the network model, measurements, measurement covariance, current
                augmented state vector ``E``, and cluster/allocation-factor information.
            debug_mode:
                If ``True``, additional diagnostic information is logged, including the current state update norm,
                objective function value, and possible ill-conditioning of the gain matrix.

        Keyword Arguments:
            **kwargs: Currently unused. Present for API compatibility and possible future extensions.

        Returns:
            The updated data container with estimated electrical state variables and cluster/allocation factors if the
            estimation succeeds. Returns ``False`` if a linear algebra error occurs.
        """
        # initialize eppci and check the observability
        self.initialize(eppci)

        # matrix calculation object for the state estimation parameter
        sem = BaseAlgebra(eppci)

        current_error, cur_it = 100., 0
        # invert covariance matrix
        # Very small standard deviations are capped to prevent the weighting from becoming infinitely large.
        eppci.r_cov[eppci.r_cov<(10**(-5))] = 10**(-5)
        r_weight = 1 / eppci.r_cov ** 2  # individual weights
        len_r = np.arange(len(r_weight))
        r_inv = csr_matrix((r_weight, (len_r, len_r)))  # diagonal matrix
        E = eppci.E  # current state vector E=x=[theta_2, ..., V_1, ...]^{T}
        obj_func = None  # J(x) objective function

        while current_error > self.tolerance and cur_it < self.max_iterations:
            # self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = csr_matrix(sem.create_rx(E)).T  # csr_matrix stores only the non-zero values

                # jacobian matrix H
                H = csr_matrix(sem.create_hx_jacobian(E))

                # remove current magnitude measurements at the first iteration
                # because with flat start they have null derivative
                if cur_it == 0 and eppci.any_i_meas:
                    idx = eppci.idx_non_imeas
                    r_inv = r_inv[idx,:][:,idx]
                    r = r[idx,:]
                    H = H[idx,:]

                # gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)
                if debug_mode:
                    norm_G = norm(G_m, np.inf)
                    norm_invG = norm(inv(G_m), np.inf)
                    cond = norm_G*norm_invG
                    if cond > 10**18:
                        self.logger.warning(f"WARNING: Gain matrix is ill-conditioned: {cond:.2e}")

                # state vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = spsolve(G_m, H.T * (r_inv * r))  # It does not explicitly compute G_m^{-1}.
                # It solves the system of equations directly.

                # Scaling of Delta_X to avoid divergence due o ill-conditioning and
                # operating conditions far from starting state variables
                current_error = float(np.max(np.abs(d_E)))
                if current_error > 0.35:
                    d_E = d_E*0.35/current_error  # type: ignore[assignment]

                # Update E with d_E
                E += d_E.ravel()  # type: ignore[union-attr] # ravel() convert multidimensional array to 1D-array
                eppci.update_E(E)

                if debug_mode:
                    obj_func = (r.T*r_inv*r)[0,0]  # J(x) = r^{T}R^{-1}r
                    self.logger.debug("Current delta_x: {:.7f}".format(current_error))
                    self.logger.debug("Current objective function value: {:.1f}".format(obj_func))

                # Restore full weighting matrix with current measurements
                if cur_it == 0 and eppci.any_i_meas:
                    r_inv = csr_matrix(np.diagflat(1 / eppci.r_cov ** 2))

                # prepare next iteration
                cur_it += 1

            except np.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successful
        self.check_result(current_error, cur_it)  # set self.successfull
        self.iterations = cur_it
        if debug_mode:
            self.obj_func = obj_func
        if self.successful:
            # store variables required for chi^2 and r_N_max test:
            self.R_inv = r_inv.toarray()
            self.Gm = G_m.toarray()
            self.r = r.toarray()
            self.H = H.toarray()
            # create h(x) for the current iteration
            self.hx = sem.create_hx(eppci.E)  # ToDo: check if eppci.E or E is correct Attention side effects possible

        # split voltage and allocation factor variables
        # clusters are set in ppc_conversion.py/_add_rated_power_information_af_wls() only if algorithm == "af-wls"
        if "clusters" in self.eppci:
            num_clusters = len(self.eppci["clusters"])
            E1 = E[:-num_clusters]
            E2 = E[-num_clusters:]
            self.af = pd.DataFrame([E2], columns=eppci["clusters"])
            eppci.update_E(E1)
            eppci.clusters = E2
        return eppci


class WLSZeroInjectionConstraintsAlgorithm(BaseAlgorithm):
    def estimate(self, eppci: ExtendedPPCI, **kwargs):
        # state vector built from delta, |V| and zero injections
        # Find pq bus with zero p,q and shunt admittance
        if not np.any(eppci["bus"][:, bus_cols + ZERO_INJ_FLAG]):
            raise UserWarning("Network has no bus with zero injections! Please use WLS instead!")
        zero_injection_bus = np.argwhere(eppci["bus"][:, bus_cols + ZERO_INJ_FLAG]).ravel()
        eppci["bus"][np.ix_(zero_injection_bus, [bus_cols + P, bus_cols + P_STD, bus_cols + Q, bus_cols + Q_STD])] = np.nan
        # Withn pq buses with zero injection identify those who have also no p or q measurement
        p_zero_injections = zero_injection_bus
        q_zero_injections = zero_injection_bus
        new_states = np.zeros(len(p_zero_injections) + len(q_zero_injections))

        num_bus = eppci["bus"].shape[0]

        # matrix calculation object
        sem = BaseAlgebraZeroInjConstraints(eppci)

        current_error, cur_it = 100., 0
        r_weight = 1 / eppci.r_cov ** 2
        len_r = np.arange(len(r_weight))
        r_inv = csr_matrix((r_weight, (len_r, len_r)))
        E = eppci.E
        # update the E matrix
        E_ext = np.r_[eppci.E, new_states]

        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                c_x = sem.create_cx(E, p_zero_injections, q_zero_injections)

                # residual r
                r = csr_matrix(sem.create_rx(E)).T
                c_rxh = csr_matrix(c_x).T

                # jacobian matrix H
                H_temp = sem.create_hx_jacobian(E)
                C_temp = sem.create_cx_jacobian(E, p_zero_injections, q_zero_injections)
                H, C = csr_matrix(H_temp), csr_matrix(C_temp)

                # gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)

                # building a new gain matrix for new constraints.
                A_1 = vstack([G_m, C])
                c_ax = hstack([C, np.zeros((C.shape[0], C.shape[0]))])  # type: ignore[arg-type, var-annotated]
                c_xT = c_ax.T
                M_tx = csr_matrix(hstack((A_1, c_xT)))  # again adding to the new gain matrix
                rhs = H.T * (r_inv * r)  # original right hand side
                C_rhs = vstack((rhs, -c_rxh))  # creating the righ hand side with new constraints

                # state vector difference d_E and update E
                d_E_ext = spsolve(M_tx, C_rhs)
                E_ext += d_E_ext.ravel()
                E = E_ext[:E.shape[0]]
                eppci.update_E(E)

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E_ext[:len(eppci.non_slack_buses) + num_bus]))
                self.logger.debug("Current error: {:.7f}".format(current_error))
            except np.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        return eppci


class IRWLSAlgorithm(BaseAlgorithm):
    def estimate(self, eppci: ExtendedPPCI, estimator: ESTIMATOR_SE_TYPES = "wls", **kwargs) -> ExtendedPPCI | bool:
        """
        Perform state estimation using an Iteratively Reweighted Least Squares (IRWLS) algorithm.

        The method iteratively updates the state vector E by solving a sequence of weighted least-squares problems. In
        each iteration, measurement residuals, the Jacobian matrix, and a (possibly robust) weighting matrix are
        computed via the selected estimator (e.g. WLS or SHGM). The iterations stop when the maximum state update falls
        below the configured tolerance or when the maximum number of iterations is reached.

        Arguments:
            eppci: Extended power system case input (ExtendedPPCI) containing the current state vector, measurement
                data, and model information.
            estimator: Type of estimator to use for building the IRWLS matrices. Typically one of:
                - "WLS": Weighted Least Squares
                - "SHGM": Robust estimator based on SHGM
            **kwargs: Additional keyword arguments passed to the underlying estimator implementation returned by
                ``get_estimator`` (e.g. tuning parameters for robust estimation).

        Returns:
            Updated ``ExtendedPPCI`` instance with the estimated state if the algorithm finishes without numerical
            errors. ``False`` if a numerical linear algebra error occurs during the solution of the linear system
            (e.g. singular gain matrix).

        """

        self.initialize(eppci)

        # matrix calculation object
        sem = get_estimator(BaseEstimatorIRWLS, estimator)(eppci, **kwargs)

        current_error, cur_it = 100., 0
        E = eppci.E
        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = csr_matrix(sem.create_rx(E)).T

                # jacobian matrix H
                H = csr_matrix(sem.create_hx_jacobian(E))

                # gain matrix G_m
                # G_m = H^t * Phi * H
                phi = csr_matrix(sem.create_phi(E))
                G_m = H.T * (phi * H)

                # state vector difference d_E and update E
                d_E = spsolve(G_m, H.T * (phi * r))
                E += d_E.ravel()  # type: ignore[attr-defined]
                eppci.update_E(E)

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E))
                self.logger.debug("Current error: {:.7f}".format(current_error))
            except np.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfully
        self.check_result(current_error, cur_it)
        # update V/delta
        return eppci
