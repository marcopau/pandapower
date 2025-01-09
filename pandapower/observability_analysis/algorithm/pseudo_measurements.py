# -*- coding: utf-8 -*-

# This code was written by Matsiushonak Siarhei and Zografos Dimitrios.
# Contributions made on 2025.

import logging

import numpy as np
from scipy.linalg import lu

from pandapower.estimation.algorithm.matrix_base import BaseAlgebra
from pandapower.estimation.idx_brch import P_FROM, P_TO
from pandapower.estimation.idx_bus import P, P_STD
from pandapower.observability_analysis.algorithm.network_analysis_core import NetworkAnalysisCore
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols

logger = logging.getLogger(__name__)


class PseudoMeasurementsHandler(NetworkAnalysisCore):

    def _add_p_measurement(self, bus_position: int, value: float):
        ppci = self.eppci.data
        ppci["bus"][bus_position, bus_cols + P] = value
        ppci["bus"][bus_position, bus_cols + P_STD] = 1.0
        p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
        p_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_FROM])
        p_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_TO])
        meas_mask = np.concatenate([
            p_bus_not_nan,
            p_line_f_not_nan,
            p_line_t_not_nan,
        ])
        self.eppci.non_nan_meas_selector = np.flatnonzero(meas_mask)

    def _get_candidates_for_power_injection(self, tolerance: float, branch_power_flow: np.ndarray):
        ppci = self.eppci.data

        branch_mask_without_power_flow = [True if abs(i) > tolerance else False for i in branch_power_flow]
        branches_idx_without_power_flow = np.flatnonzero(branch_mask_without_power_flow)

        branch_without_power_flow = self.eppci.data['branch'][branches_idx_without_power_flow]
        logger.info(f"Number of branches without power flow {len(branch_without_power_flow)}. Branches: {branches_idx_without_power_flow}")
        candidates_for_power_injection = np.unique(np.concatenate((branch_without_power_flow[:, 0], branch_without_power_flow[:, 1])))
        p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
        p_bus_not_nan_idx = np.flatnonzero(p_bus_not_nan)
        candidates_for_power_injection = np.setdiff1d(candidates_for_power_injection, p_bus_not_nan_idx)
        return candidates_for_power_injection

    def handle(self, max_iter=150, tolerance=1e-10):
        # Step 1
        buses_with_pseudo_power_injection = []
        N = int(self.eppci.data['bus'].shape[0])
        self.eppci.E[:N - 1] = 0
        self.eppci.E[N:] = 1
        self._clean_not_p_measurements()
        self._set_delta_v_bus_selector()
        self._reset_network_values()

        # form gain matrix
        sem = BaseAlgebra(self.eppci)
        jacobian = sem.create_hx_jacobian(self.eppci.E)
        gain_matrix = np.dot(jacobian.T, jacobian)

        # Step 2
        # LU decomposition with pivoting
        P, L, U = lu(gain_matrix)
        zero_pivots = np.where(np.abs(np.diag(U)) < tolerance)[0]
        stop_iterations = self._validate_zero_pivots(zero_pivots, N)
        if stop_iterations is True:
            return []

        current_iteration = 1
        jacobian_with_pseudo_meas = self._add_pseudo_meas_to_jacobian(jacobian, zero_pivots)

        while current_iteration <= max_iter:
            logger.info(f"Iteration: {current_iteration}")

            # Step 3
            solution = self._solve_dc_estimator_equation(jacobian_with_pseudo_meas, zero_pivots)

            # Step 4
            branch_power_flow = self._calculate_branch_power_flow(solution)
            candidates_for_power_injection = self._get_candidates_for_power_injection(tolerance, branch_power_flow)
            if not candidates_for_power_injection.any():
                logger.info("No candidates for power injection. Stop iterations.")
                break

            power_injection_candidate_index = int(candidates_for_power_injection[-1])
            buses_with_pseudo_power_injection.append(power_injection_candidate_index)

            self._add_p_measurement(power_injection_candidate_index, 0.0)

            # Step 5
            sem = BaseAlgebra(self.eppci)
            jacobian2 = sem.create_hx_jacobian(self.eppci.E)
            jacobian_with_pseudo_meas2 = self._add_pseudo_meas_to_jacobian(jacobian2, zero_pivots)

            # Step 6
            solution = self._solve_dc_estimator_equation(jacobian_with_pseudo_meas2, zero_pivots)
            # calculate residuals
            pseudo_theta = np.arange(len(zero_pivots))
            residuals = pseudo_theta - solution[zero_pivots]
            non_zero_indices = np.where(np.abs(residuals) > tolerance)[0]
            pivots_with_non_zero_residuals = zero_pivots[non_zero_indices]
            if not any(pivots_with_non_zero_residuals):
                logger.info("not any(pivots_with_non_zero_residuals)")
                continue
            theta_candidate_to_drop = pivots_with_non_zero_residuals[0]

            # drop redundant pseudo measurement
            zero_pivots = zero_pivots[zero_pivots != theta_candidate_to_drop]
            jacobian_with_pseudo_meas = self._add_pseudo_meas_to_jacobian(jacobian2, zero_pivots)

            # Step 7 - increase counter and go to step 3
            current_iteration += 1

        if current_iteration == max_iter:
            raise Exception("Maximum number of iterations reached. Algorithm did not converge.")

        return buses_with_pseudo_power_injection
