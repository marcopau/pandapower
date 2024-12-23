# -*- coding: utf-8 -*-
import logging
from copy import deepcopy

import numpy as np
from scipy.linalg import lu
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import ldl
import pandapower as pp
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra
from pandapower.estimation.idx_brch import P_FROM, P_TO, Q_FROM, Q_TO, Q_FROM_STD, Q_TO_STD, IA_FROM, IA_FROM_STD, IA_TO, IA_TO_STD, \
    IM_FROM, IM_FROM_STD, IM_TO, IM_TO_STD
from pandapower.estimation.idx_bus import P, P_STD, Q, Q_STD, VM, VM_STD, VA, VA_STD
from pandapower.estimation.ppc_conversion import ExtendedPPCI
from pandapower.estimation.ppc_conversion import pp2eppci
from pandapower.pypower.idx_brch import BR_R, BR_X, BR_B, BR_G, SHIFT, TAP
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols
from pandapower.pypower.idx_bus import bus_cols, GS, BS

logger = logging.getLogger(__name__)


class PseudoMeasurementsHandler:
    def __init__(self, eppci: ExtendedPPCI, net=None):
        self.eppci = deepcopy(eppci)
        self.net = net

    @classmethod
    def from_ppnet(
            cls,
            net: pp.pandapowerNet, v_start='flat',
            delta_start='flat',
            algorithm='wls',
            calculate_voltage_angles=True,
            zero_injection='auto',
    ) -> "PseudoMeasurementsHandler":
        net, _, eppci = pp2eppci(net, v_start=v_start, delta_start=delta_start,
                                 calculate_voltage_angles=calculate_voltage_angles,
                                 zero_injection=zero_injection, algorithm=algorithm,
                                 )

        return cls(eppci, net)

    def convert_res_to_ppnet(self, ppci_result):
        res = {}
        print("Result")
        for bs in ppci_result:
            tmp = [i for i, j in enumerate(self.net._pd2ppc_lookups['bus']) if j == bs]
            res[bs] = tmp
            print(f"{bs}: {tmp}")
        return res

    def clean_not_p_measurements(self):
        ppci = self.eppci.data

        ppci["bus"][:, bus_cols + VM] = np.NaN
        ppci["bus"][:, bus_cols + VM_STD] = np.NaN

        ppci["bus"][:, bus_cols + Q] = np.NaN
        ppci["bus"][:, bus_cols + Q_STD] = np.NaN

        ppci["bus"][:, bus_cols + VA] = np.NaN
        ppci["bus"][:, bus_cols + VA_STD] = np.NaN

        ppci["branch"][:, branch_cols + Q_FROM] = np.NaN
        ppci["branch"][:, branch_cols + Q_FROM_STD] = np.NaN

        ppci["branch"][:, branch_cols + Q_TO] = np.NaN
        ppci["branch"][:, branch_cols + Q_TO_STD] = np.NaN

        ppci["branch"][:, branch_cols + IA_FROM] = np.NaN
        ppci["branch"][:, branch_cols + IA_FROM_STD] = np.NaN

        ppci["branch"][:, branch_cols + IA_TO] = np.NaN
        ppci["branch"][:, branch_cols + IA_TO_STD] = np.NaN

        ppci["branch"][:, branch_cols + IM_FROM] = np.NaN
        ppci["branch"][:, branch_cols + IM_FROM_STD] = np.NaN

        ppci["branch"][:, branch_cols + IM_TO] = np.NaN
        ppci["branch"][:, branch_cols + IM_TO_STD] = np.NaN

        self.eppci._initialize_meas()

    def reset_network_values(self):
        self.eppci.data['branch'][:, BR_R] = np.zeros(len(self.eppci.data['branch'][:, BR_R]))
        self.eppci.data['branch'][:, BR_X] = np.ones(len(self.eppci.data['branch'][:, BR_X]))

        self.eppci.data['branch'][:, BR_B] = np.zeros(len(self.eppci.data['branch'][:, BR_B]))
        self.eppci.data['branch'][:, BR_G] = np.zeros(len(self.eppci.data['branch'][:, BR_G]))

        self.eppci.data['bus'][:, GS] = np.zeros(len(self.eppci.data['bus'][:, GS]))
        self.eppci.data['bus'][:, BS] = np.zeros(len(self.eppci.data['bus'][:, BS]))

        self.eppci.data['branch'][:, TAP] = np.ones(len(self.eppci.data['branch'][:, TAP]))
        self.eppci.data['branch'][:, SHIFT] = np.zeros(len(self.eppci.data['branch'][:, SHIFT]))

    def set_delta_v_bus_selector(self):
        self.eppci.delta_v_bus_selector = np.arange(len(self.eppci.data['bus']))

    def delete_branch(self, lines: np.ndarray):
        ppci = self.eppci.data
        N = ppci["branch"].shape[0]
        ppci["branch"] = np.delete(ppci["branch"], lines, axis=0)
        p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
        p_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_FROM])
        p_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_TO])
        meas_mask = np.concatenate([
            p_bus_not_nan,
            p_line_f_not_nan,
            p_line_t_not_nan,
        ])

        self.eppci.non_nan_meas_selector = np.flatnonzero(meas_mask)

        if "Ybus" in ppci["internal"] and ppci["internal"]["Ybus"].size:
            rows_to_keep = list(set(list(range(N))) - set(lines))
            Yf = ppci["internal"]["Yf"]
            ppci["internal"]["Yf"] = Yf[rows_to_keep, :]
            Yt = ppci["internal"]["Yt"]
            ppci["internal"]["Yt"] = Yt[rows_to_keep, :]

    def add_p_measurement(self, bus_position: int, value: float):
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

    def calculate_branch_power_flow(self, theta_vector: np.ndarray):
        from_buses = self.eppci.data['branch'][:, 0].real.astype(np.int64)
        to_buses = self.eppci.data['branch'][:, 1].real.astype(np.int64)

        # Calculate the difference in theta values for 'from' and 'to' buses
        theta_diff = theta_vector[from_buses] - theta_vector[to_buses]
        return theta_diff

    def get_candidates_for_power_injection(self, tolerance: float, branch_power_flow: np.ndarray):
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

    def solve_dc_estimator_equation(self, jacobian_with_pseudo_meas: np.ndarray, zero_pivots: np.ndarray):
        size = jacobian_with_pseudo_meas.shape[0]

        z = np.zeros(size)
        z[-len(zero_pivots):] = np.arange(len(zero_pivots))
        h_w_z = np.dot(jacobian_with_pseudo_meas.T, z)  # W is the identity matrix, so multiplication has no effect and is skipped

        gain_matrix_with_pseudo_meas = np.dot(jacobian_with_pseudo_meas.T, jacobian_with_pseudo_meas)  # check
        cond = np.linalg.cond(gain_matrix_with_pseudo_meas)
        print(f"Condition number: {cond}")

        sparse_gain_matrix = csr_matrix(gain_matrix_with_pseudo_meas)
        solution = spsolve(sparse_gain_matrix, h_w_z)
        self._validate_solution(gain_matrix_with_pseudo_meas, solution, h_w_z)
        return solution

    def _validate_solution(self, A: np.ndarray, x: np.ndarray, b: np.ndarray) -> None:
        """
           Checks for NaN values in the solution vector x, and if valid, computes and prints the squared residual.

           Parameters:
               A (np.ndarray): The coefficient matrix.
               x (np.ndarray): The solution vector.
               b (np.ndarray): The right-hand side vector.

           Raises:
               ValueError: If x contains NaN values.
           """
        if np.any(np.isnan(x)):
            raise Exception("Equation solving failed")

        # Compute the residual
        residual = np.dot(A, x) - b
        squared_residual = np.sum(residual ** 2)
        # print(f"Residual for theta vector at step 5: {squared_residual}")

    def validate_zero_pivots(self, zero_pivots: np.ndarray, N: int):
        stop_iteration = False
        if zero_pivots.size <= 1:
            logger.info("Only one zero pivot. Stop iterations.")
            stop_iteration = True
        else:
            logger.info(f"Zero_pivots {zero_pivots}")
        return stop_iteration

    def add_pseudo_meas_to_jacobian(self, jacobian: np.ndarray, zero_pivots: np.ndarray):

        # Create new rows with pseudo measurements
        new_jacobian_rows = np.zeros((len(zero_pivots), jacobian.shape[1]))
        new_jacobian_rows[np.arange(len(zero_pivots)), zero_pivots] = 1

        # Stack the new rows with the original matrix
        jacobian_with_pseudo_meas = np.vstack((jacobian, new_jacobian_rows))
        logger.info(f"Introduced {len(zero_pivots)} pseudo measurements")
        return jacobian_with_pseudo_meas

    def handle(self, max_iter=150, tolerance=1e-10):
        # Step 1
        buses_with_pseudo_power_injection = []
        N = int(self.eppci.data['bus'].shape[0])
        self.eppci.E[:N - 1] = 0
        self.eppci.E[N:] = 1
        self.clean_not_p_measurements()
        self.set_delta_v_bus_selector()
        self.reset_network_values()

        # form gain matrix
        sem = BaseAlgebra(self.eppci)
        jacobian = sem.create_hx_jacobian(self.eppci.E)
        gain_matrix = np.dot(jacobian.T, jacobian)

        # Step 2
        # LU decomposition with pivoting
        P, L, U = lu(gain_matrix)
        zero_pivots = np.where(np.abs(np.diag(U)) < tolerance)[0]
        stop_iterations = self.validate_zero_pivots(zero_pivots, N)
        if stop_iterations is True:
            return []

        current_iteration = 1
        jacobian_with_pseudo_meas = self.add_pseudo_meas_to_jacobian(jacobian, zero_pivots)

        while current_iteration <= max_iter:
            print(f"Iteration: {current_iteration}")

            # Step 3
            solution = self.solve_dc_estimator_equation(jacobian_with_pseudo_meas, zero_pivots)

            # Step 4
            branch_power_flow = self.calculate_branch_power_flow(solution)
            candidates_for_power_injection = self.get_candidates_for_power_injection(tolerance, branch_power_flow)
            if not candidates_for_power_injection.any():
                logger.info("No candidates for power injection. Stop iterations.")
                break

            power_injection_candidate_index = int(candidates_for_power_injection[-1])
            buses_with_pseudo_power_injection.append(power_injection_candidate_index)

            self.add_p_measurement(power_injection_candidate_index, 0.0)

            # Step 5
            sem = BaseAlgebra(self.eppci)
            jacobian2 = sem.create_hx_jacobian(self.eppci.E)
            jacobian_with_pseudo_meas2 = self.add_pseudo_meas_to_jacobian(jacobian2, zero_pivots)

            # Step 6
            solution = self.solve_dc_estimator_equation(jacobian_with_pseudo_meas2, zero_pivots)
            # calculate residuals
            pseudo_theta = np.arange(len(zero_pivots))
            residuals = pseudo_theta - solution[zero_pivots]
            non_zero_indices = np.where(np.abs(residuals) > tolerance)[0]
            pivots_with_non_zero_residuals = zero_pivots[non_zero_indices]
            if not any(pivots_with_non_zero_residuals):
                print("not any(pivots_with_non_zero_residuals)")
                continue
            theta_candidate_to_drop = pivots_with_non_zero_residuals[0]

            # drop redundant pseudo measurement
            zero_pivots = zero_pivots[zero_pivots != theta_candidate_to_drop]
            jacobian_with_pseudo_meas = self.add_pseudo_meas_to_jacobian(jacobian2, zero_pivots)

            # Step 7 - increase counter and go to step 3
            current_iteration += 1

        if current_iteration == max_iter:
            raise Exception("Maximum number of iterations reached. Algorithm did not converge.")

        return buses_with_pseudo_power_injection
