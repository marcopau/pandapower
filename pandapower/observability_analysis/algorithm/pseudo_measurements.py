# -*- coding: utf-8 -*-

# This code was written by Matsiushonak Siarhei and Zografos Dimitrios.
# Contributions made on 2025.

import logging

import numpy as np
from scipy.linalg import lu
from scipy.sparse import csr_matrix

from pandapower.estimation.algorithm.matrix_base import BaseAlgebra
from pandapower.estimation.idx_brch import P_FROM, P_TO
from pandapower.estimation.idx_bus import P, P_STD
from pandapower.observability_analysis.algorithm.network_analysis_core import NetworkAnalysisCore
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols

logger = logging.getLogger(__name__)


class PseudoMeasurementsHandler(NetworkAnalysisCore):

    def _add_p_measurement(self, bus_position: int, value: float):
        """
           Adds a power (P) measurement to the internal data structure for a specified bus position.

           This method updates the measurement data for active power (P) and its standard deviation (P_STD)
           at the specified bus position. It also recalculates and updates the selector for non-NaN
           measurements across buses and branches.

           Args:
               bus_position (int): The index of the bus where the measurement is added.
               value (float): The value of the active power measurement (P) to add.

           Modifies:
               - Updates `ppci["bus"]` to include the new measurement value and standard deviation.
               - Recomputes the non-NaN measurement selector (`self.non_nan_meas_selector`) to reflect the change.

           Example:
               If a new power measurement is added to bus 3, this method will:
               1. Update the bus data at the specified position.
               2. Recalculate the selector for all non-NaN measurements across buses and branches.

           Attributes Updated:
               - `self.eppci.data["bus"]` (adds measurement and its standard deviation).
               - `self.non_nan_meas_selector` (updates list of indices for non-NaN measurements).

        """

        ppci = self.eppci.data

        # Update bus measurements
        ppci["bus"][bus_position, bus_cols + P] = value
        ppci["bus"][bus_position, bus_cols + P_STD] = 1.0

        # Create masks for non-NaN measurements
        p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
        p_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_FROM])
        p_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_TO])

        # Combine all non-NaN masks
        meas_mask = np.hstack((p_bus_not_nan, p_line_f_not_nan, p_line_t_not_nan))

        # Find indices of new non-NaN measurements
        new_non_nan_meas_selector = np.flatnonzero(meas_mask)

        # Update the selector for non-NaN measurements
        existing_non_nan_tail = self.non_nan_meas_selector[len(new_non_nan_meas_selector) - 1:]
        self.non_nan_meas_selector = np.hstack((new_non_nan_meas_selector, existing_non_nan_tail))

    def _get_candidates_for_power_injection(self, tolerance: float, branch_power_flow: np.ndarray) -> np.ndarray:
        """
        Identifies candidate buses for power injection based on branches with negligible power flow.

        This method filters out branches with power flow below a specified tolerance, identifies their
        connected buses, and excludes buses that already have a power measurement.

        Args:
            tolerance (float): The minimum power flow magnitude to consider a branch as active.
            branch_power_flow (np.ndarray): An array representing the power flow through each branch.

        Returns:
            np.ndarray: Array of bus indices that are candidates for power injection.
        """

        ppci = self.eppci.data

        # Identify branches with negligible power flow
        branch_mask_without_power_flow = np.abs(branch_power_flow) > tolerance
        branches_idx_without_power_flow = np.flatnonzero(branch_mask_without_power_flow)

        # Extract branch data for these branches
        branch_without_power_flow = ppci["branch"][branches_idx_without_power_flow]

        # Get unique buses connected to these branches
        connected_buses = np.unique(np.concatenate((branch_without_power_flow[:, 0], branch_without_power_flow[:, 1])))

        # Exclude buses that already have power measurements
        p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
        p_bus_not_nan_idx = np.flatnonzero(p_bus_not_nan)
        candidates_for_power_injection = np.setdiff1d(connected_buses, p_bus_not_nan_idx)

        return candidates_for_power_injection

    def _create_jacobian(self, sem: BaseAlgebra) -> np.ndarray:
        """
        Creates the Jacobian matrix.

        This method initializes measurement and voltage selectors for the given algebra object
        (`sem`) and computes the Jacobian matrix based on the current system state.

        Args:
            sem (BaseAlgebra): An instance of the algebra class responsible for handling the
                Jacobian matrix computation and selector management.

        Returns:
            np.ndarray: The computed Jacobian matrix.
        """

        # Total number of measurements (buses + branches)
        num_buses = len(self.eppci.data['bus'])
        num_branches = len(self.eppci.data['branch'])
        all_meas_number = num_buses + 2 * num_branches

        # Initialize selectors in the algebra object
        sem.non_nan_meas_selector = np.arange(all_meas_number)  # Indices for all measurements
        sem.delta_v_bus_selector = np.arange(num_buses)  # Indices for voltage measurements at buses

        # Compute the Jacobian matrix using the algebra object
        jacobian = sem.create_hx_jacobian(self.eppci.E)

        return jacobian

    def _init_non_nan_meas_selector(self):
        self.non_nan_meas_selector = self.eppci.non_nan_meas_selector

    def handle(self, max_iter=150, tolerance=1e-10):
        # Step 1: Initialization
        buses_with_pseudo_power_injection = []
        N = int(self.eppci.data['bus'].shape[0])
        self.eppci.E[:N - 1] = 0
        self.eppci.E[N:] = 1
        self._clean_not_p_measurements()
        self._reset_network_values()
        self._init_non_nan_meas_selector()

        # Form gain matrix
        sem = BaseAlgebra(self.eppci)
        original_jacobian = self._create_jacobian(sem)
        jacobian = original_jacobian[self.non_nan_meas_selector, :]
        gain_matrix = jacobian.T @ jacobian

        # Step 2: LU decomposition with pivoting
        P, L, U = lu(gain_matrix)
        zero_pivots = np.where(np.abs(np.diag(U)) < tolerance)[0]
        stop_iterations = self._validate_zero_pivots(zero_pivots, N)
        if stop_iterations is True:
            return []

        # Add pseudo-measurements to Jacobian
        original_jacobian_with_pseudo_meas = self._add_pseudo_meas_to_jacobian(original_jacobian, zero_pivots)
        original_jacobian_with_pseudo_meas = csr_matrix(original_jacobian_with_pseudo_meas)
        self.non_nan_meas_selector = np.append(self.non_nan_meas_selector, original_jacobian.shape[0] + np.arange(len(zero_pivots)))
        jacobian_with_pseudo_meas = original_jacobian_with_pseudo_meas[self.non_nan_meas_selector, :]

        # Iterative algorithm
        current_iteration = 1
        while current_iteration <= max_iter:
            logger.info(f"Iteration: {current_iteration}")

            # Step 3: Solve DC estimator equation
            solution = self._solve_dc_estimator_equation(jacobian_with_pseudo_meas, zero_pivots)

            # Step 4: Evaluate branch power flow and find candidates
            branch_power_flow = self._calculate_branch_power_flow(solution)
            candidates_for_power_injection = self._get_candidates_for_power_injection(tolerance, branch_power_flow)
            if not candidates_for_power_injection.any():
                logger.info("No candidates for power injection. Stop iterations.")
                break

            power_injection_candidate_index = int(candidates_for_power_injection[-1])
            buses_with_pseudo_power_injection.append(power_injection_candidate_index)

            # Update Jacobian with pseudo-measurements
            self._add_p_measurement(power_injection_candidate_index, 0.0)

            # Step 5: Form gain matrix
            jacobian_with_pseudo_meas = original_jacobian_with_pseudo_meas[self.non_nan_meas_selector, :]

            # Step 6: Solve DC estimator equation
            solution = self._solve_dc_estimator_equation(jacobian_with_pseudo_meas, zero_pivots)

            # Calculate residuals
            pseudo_theta = np.arange(len(zero_pivots))
            residuals = pseudo_theta - solution[zero_pivots]
            non_zero_indices = np.where(np.abs(residuals) > tolerance)[0]
            pivots_with_non_zero_residuals = zero_pivots[non_zero_indices]
            if not any(pivots_with_non_zero_residuals):
                logger.info("not any(pivots_with_non_zero_residuals)")
                continue

            # Drop redundant pseudo-measurement
            theta_candidate_to_drop = pivots_with_non_zero_residuals[0]
            theta_candidate_to_drop_indices = len(zero_pivots) - np.where(zero_pivots == theta_candidate_to_drop)[0]
            self.non_nan_meas_selector = np.delete(self.non_nan_meas_selector, -theta_candidate_to_drop_indices)
            jacobian_with_pseudo_meas = original_jacobian_with_pseudo_meas[self.non_nan_meas_selector, :]
            zero_pivots = zero_pivots[zero_pivots != theta_candidate_to_drop]

            current_iteration += 1

        if current_iteration == max_iter:
            raise Exception("Maximum number of iterations reached. Algorithm did not converge.")

        return buses_with_pseudo_power_injection
