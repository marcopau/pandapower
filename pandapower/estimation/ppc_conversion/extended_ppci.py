# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from collections import UserDict

import numpy as np

from pandapower.estimation.idx_brch import (P_FROM, P_FROM_IDX, P_FROM_STD,
                                            Q_FROM, Q_FROM_IDX, Q_FROM_STD,
                                            IM_FROM, IM_FROM_IDX, IM_FROM_STD,
                                            P_TO, P_TO_IDX, P_TO_STD,
                                            Q_TO, Q_TO_IDX, Q_TO_STD,
                                            IM_TO, IM_TO_IDX, IM_TO_STD)
from pandapower.estimation.idx_bus import (VM, VM_IDX, VM_STD,
                                           VA, VA_IDX, VA_STD,
                                           P, P_IDX, P_STD,
                                           Q, Q_IDX, Q_STD)
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import BUS_TYPE as pypower_BUS_TYPE, VM as pypower_VM, VA as pypower_VA
from pandapower.pypower.idx_bus import bus_cols
from pandapower.pypower.makeYbus import makeYbus


class ExtendedPPCI(UserDict):
    def __init__(self, ppci, algorithm):
        """Initialize ppci object with measurements."""
        self.data = ppci
        self.algorithm = algorithm

        # Measurement relevant parameters
        self.z = None
        self.r_cov = None
        self.pp_meas_indices = None
        self.non_nan_meas_mask = None
        self.non_nan_meas_selector = None
        self.any_i_meas = False
        self.any_degree_meas = False

        # check slack bus
        self.non_slack_buses = np.argwhere(ppci["bus"][:, pypower_BUS_TYPE] != 3).ravel()
        self.non_slack_bus_mask = (ppci['bus'][:, pypower_BUS_TYPE] != 3).ravel()
        self.num_non_slack_bus = np.sum(self.non_slack_bus_mask)
        self.delta_v_bus_mask = np.r_[self.non_slack_bus_mask,
        np.ones(self.non_slack_bus_mask.shape[0], dtype=bool)].ravel()
        self.delta_v_bus_selector = np.flatnonzero(self.delta_v_bus_mask)

        # Iniialize measurements
        self._initialize_meas()

        # Initialize state variable
        self.v_init = ppci["bus"][:, pypower_VM]
        self.delta_init = np.radians(ppci["bus"][:, pypower_VA])
        self.E_init = np.r_[self.delta_init[self.non_slack_bus_mask], self.v_init]
        self.v = self.v_init.copy()
        self.delta = self.delta_init.copy()
        self.E = self.E_init.copy()
        if algorithm == "af-wls":
            self.E = np.concatenate((self.E, np.full(ppci["clusters"].shape, 0.5)))

    def _initialize_meas(self):
        # calculate relevant vectors from ppci measurements
        self.z, self.pp_meas_indices, self.r_cov, self.non_nan_meas_mask, \
            self.idx_non_imeas = \
            _build_measurement_vectors(self, update_meas_only=False)
        # self.non_nan_meas_selector = np.flatnonzero(self.non_nan_meas_mask)

    def update_meas(self):
        self.z = _build_measurement_vectors(self, update_meas_only=True)

    @property
    def V(self):
        return self.v * np.exp(1j * self.delta)

    def reset(self):
        self.v, self.delta, self.E = \
            self.v_init.copy(), self.delta_init.copy(), self.E_init.copy()

    def update_E(self, E):
        self.E = E
        self.v = E[self.num_non_slack_bus:]
        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]

    def E2V(self, E):
        self.update_E(E)
        return self.V

    def get_Y(self):
        # Using recycled version if available
        if "Ybus" in self["internal"] and self["internal"]["Ybus"].size:
            Ybus, Yf, Yt = self["internal"]['Ybus'], self["internal"]['Yf'], self["internal"]['Yt']
        else:
            # build admittance matrices
            Ybus, Yf, Yt = makeYbus(self['baseMVA'], self['bus'], self['branch'])
            self["internal"]['Ybus'], self["internal"]['Yf'], self["internal"]['Yt'] = Ybus, Yf, Yt
        return Ybus, Yf, Yt


def _build_measurement_vectors(ppci, update_meas_only=False):
    """
    Building measurement vector z, pandapower to ppci measurement mapping and covariance matrix R
    :param ppci: generated ppci which contains the measurement columns
    :param branch_cols: number of columns in original ppci["branch"] without measurements
    :param bus_cols: number of columns in original ppci["bus"] without measurements
    :return: both created vectors
    """
    p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
    p_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_FROM])
    p_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_TO])
    q_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + Q])
    q_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + Q_FROM])
    q_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + Q_TO])
    v_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + VM])
    v_degree_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + VA])
    i_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IM_FROM])
    i_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IM_TO])
    # i_degree_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IA_FROM])
    # i_degree_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IA_TO])

    # piece together our measurement vector z
    z = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P],
                        ppci["bus"][q_bus_not_nan, bus_cols + Q],
                        ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM],
                        ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM],
                        ppci["branch"][p_line_t_not_nan, branch_cols + P_TO],
                        ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO],
                        ppci["bus"][v_bus_not_nan, bus_cols + VM],
                        ppci["bus"][v_degree_bus_not_nan, bus_cols + VA],
                        ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM],
                        ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO]
                        )).real.astype(np.float64)
    imag_meas = np.concatenate((np.zeros(sum(p_bus_not_nan)),
                                np.zeros(sum(q_bus_not_nan)),
                                np.zeros(sum(p_line_f_not_nan)),
                                np.zeros(sum(q_line_f_not_nan)),
                                np.zeros(sum(p_line_t_not_nan)),
                                np.zeros(sum(q_line_t_not_nan)),
                                np.zeros(sum(v_bus_not_nan)),
                                np.zeros(sum(v_degree_bus_not_nan)),
                                np.ones(sum(i_line_f_not_nan)),
                                np.ones(sum(i_line_t_not_nan))
                                )).astype(bool)
    if ppci.algorithm == "af-wls":
        balance_eq_meas = np.zeros(ppci["rated_power_clusters"].shape[0]).astype(np.float64)
        af_vmeas = 0.4 * np.ones(len(ppci["clusters"]))
        z = np.concatenate(
            (z, balance_eq_meas[ppci.non_slack_bus_mask], balance_eq_meas[ppci.non_slack_bus_mask], af_vmeas))
        imag_meas = np.concatenate((imag_meas,
                                    np.zeros(2 * balance_eq_meas[ppci.non_slack_bus_mask].shape[0]),
                                    np.zeros(af_vmeas.shape[0]))).astype(bool)
    idx_non_imeas = np.flatnonzero(~imag_meas)

    if not update_meas_only:
        # conserve the pandapower indices of measurements in the ppci order
        pp_meas_indices = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P_IDX],
                                          ppci["bus"][q_bus_not_nan, bus_cols + Q_IDX],
                                          ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM_IDX],
                                          ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM_IDX],
                                          ppci["branch"][p_line_t_not_nan, branch_cols + P_TO_IDX],
                                          ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO_IDX],
                                          ppci["bus"][v_bus_not_nan, bus_cols + VM_IDX],
                                          ppci["bus"][v_degree_bus_not_nan, bus_cols + VA_IDX],
                                          ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM_IDX],
                                          ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO_IDX],
                                          )).real.astype(np.int64)
        # Covariance matrix R
        r_cov = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P_STD],
                                ppci["bus"][q_bus_not_nan, bus_cols + Q_STD],
                                ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM_STD],
                                ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM_STD],
                                ppci["branch"][p_line_t_not_nan, branch_cols + P_TO_STD],
                                ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO_STD],
                                ppci["bus"][v_bus_not_nan, bus_cols + VM_STD],
                                ppci["bus"][v_degree_bus_not_nan, bus_cols + VA_STD],
                                ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM_STD],
                                ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO_STD],
                                )).real.astype(np.float64)
        meas_mask = {"pbus": np.flatnonzero(p_bus_not_nan),
                     "qbus": np.flatnonzero(q_bus_not_nan),
                     "pfrom": np.flatnonzero(p_line_f_not_nan),
                     "qfrom": np.flatnonzero(q_line_f_not_nan),
                     "pto": np.flatnonzero(p_line_t_not_nan),
                     "qto": np.flatnonzero(q_line_t_not_nan),
                     "vm": np.flatnonzero(v_bus_not_nan),
                     "va": np.flatnonzero(v_degree_bus_not_nan),
                     "ifrom": np.flatnonzero(i_line_f_not_nan),
                     "ito": np.flatnonzero(i_line_t_not_nan)}

        if ppci.algorithm == "af-wls":
            num_clusters = len(ppci["clusters"])
            P_balance_dev_std = np.sqrt(
                np.sum(np.square(ppci["rated_power_clusters"][:, 2 * num_clusters:3 * num_clusters]), axis=1))
            Q_balance_dev_std = np.sqrt(
                np.sum(np.square(ppci["rated_power_clusters"][:, 3 * num_clusters:4 * num_clusters]), axis=1))
            af_vmeas_dev_std = 0.15 * np.ones(len(ppci["clusters"]))
            r_cov = np.concatenate(
                (r_cov, P_balance_dev_std[ppci.non_slack_bus_mask], Q_balance_dev_std[ppci.non_slack_bus_mask],
                 af_vmeas_dev_std))
            meas_mask["pbalance"] = np.flatnonzero(ppci.non_slack_bus_mask)
            meas_mask["qbalance"] = np.flatnonzero(ppci.non_slack_bus_mask)
            meas_mask["afactor"] = np.arange(num_clusters)

        return z, pp_meas_indices, r_cov, meas_mask, idx_non_imeas
    else:
        return z
