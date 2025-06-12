# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np

from pandapower.auxiliary import _init_runse_options
from pandapower.estimation.idx_brch import (branch_cols_se)
from pandapower.estimation.idx_bus import (bus_cols_se)
from pandapower.estimation.ppc_conversion.branch import _add_measurements_to_line, _add_measurements_to_trafo, \
    _add_measurements_to_trafo3w
from pandapower.estimation.ppc_conversion.bus import _add_measurements_to_bus, _add_zero_injection
from pandapower.estimation.ppc_conversion.extended_ppci import ExtendedPPCI
from pandapower.estimation.ppc_conversion.utils import map_measurement_to_bus
from pandapower.pd2ppc import _pd2ppc
from pandapower.pf.run_newton_raphson_pf import _run_dc_pf
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

std_logger = logging.getLogger(__name__)


def _add_rated_power_information_af_wls(net, ppci):
    cluster_list_loads = net.load["type"].unique()
    cluster_list_gen = net.sgen["type"].unique()
    cluster_list_tot = np.concatenate((cluster_list_loads, cluster_list_gen), axis=0)
    ppci["clusters"] = cluster_list_tot
    num_clusters = len(cluster_list_tot)
    num_buses = ppci["bus"].shape[0]
    ppci["rated_power_clusters"] = np.zeros([num_buses, 4 * num_clusters])
    for var in ["load", "sgen"]:
        in_service = net[var]["in_service"]
        active_elements = net[var][in_service]
        bus = net._pd2ppc_lookups["bus"][active_elements.bus].astype(int)
        P = active_elements.p_mw.values / ppci["baseMVA"]
        Q = active_elements.q_mvar.values / ppci["baseMVA"]
        if var == 'load':
            P *= -1
            Q *= -1
        cluster = active_elements.type.values
        if (bus >= ppci["bus"].shape[0]).any():
            std_logger.warning("Loads or sgen defined in pp-grid do not exist in ppci, will be deleted!")
            P = P[bus < ppci["bus"].shape[0]]
            Q = Q[bus < ppci["bus"].shape[0]]
            cluster = cluster[bus < ppci["bus"].shape[0]]
            bus = bus[bus < ppci["bus"].shape[0]]
        for k in range(num_clusters):
            cluster[cluster == cluster_list_tot[k]] = k
        cluster = cluster.astype(int)
        for i in range(len(P)):
            bus_i, cluster_i, P_i, Q_i = bus[i], cluster[i], P[i], Q[i]
            ppci["rated_power_clusters"][bus_i, cluster_i] += P_i
            ppci["rated_power_clusters"][bus_i, cluster_i + num_clusters] += Q_i
            ppci["rated_power_clusters"][bus_i, cluster_i + 2 * num_clusters] += abs(
                0.03 * P_i)  # std dev cluster variability hardcoded, think how to change it
            ppci["rated_power_clusters"][bus_i, cluster_i + 3 * num_clusters] += abs(
                0.03 * Q_i)  # std dev cluster variability hardcoded, think how to change it


def _init_ppc(net, v_start, delta_start, calculate_voltage_angles):
    # select elements in service and convert pandapower ppc to ppc
    _init_runse_options(net, v_start=v_start, delta_start=delta_start,
                        calculate_voltage_angles=calculate_voltage_angles)
    ppc, ppci = _pd2ppc(net)

    # do dc power flow for phase shifting transformers
    if np.any(net.trafo.shift_degree):
        vm_backup = ppci["bus"][:, 7].copy()
        pq_backup = ppci["bus"][:, [2, 3]].copy()
        # ppci["bus"][:, [2, 3]] = 0.
        ppci = _run_dc_pf(ppci)
        ppci["bus"][:, 7] = vm_backup
        ppci["bus"][:, [2, 3]] = pq_backup

    return ppc, ppci


def _add_measurements_to_ppci(net, ppci, zero_injection, algorithm):
    """
       Add measurement data from pandapower to the internal ppci structure.
       Extends ppci with additional columns for measurement values, standard deviations, and indices.

       Parameters:
       - net: pandapower net object
       - ppci: internal ppci dictionary (result of _pd2ppc)
       - zero_injection: string with desired option for zero injection measurement creation
       - algorithm: estimator algorithm name

       Returns:
       - ppci: updated ppci dictionary with additional measurement data columns
    """
    meas = net.measurement.copy(deep=True)
    if meas.empty:
        raise Exception("No measurements are available in pandapower Network! Abort estimation!")

    # Convert power measurements (p, q) to per unit (p.u.)
    meas.loc[meas.measurement_type == "p", ["value", "std_dev"]] /= ppci["baseMVA"]
    meas.loc[meas.measurement_type == "q", ["value", "std_dev"]] /= ppci["baseMVA"]

    # Convert current (i) measurements to p.u.
    i_meas = meas.query("measurement_type=='i'")
    if not i_meas.empty:
        # Convert side from string to bus id
        i_meas["side"] = map_measurement_to_bus(net, i_meas)
        base_i_ka = ppci["baseMVA"] / i_meas.side.map(net.bus.vn_kv)
        meas.loc[i_meas.index, "value"] /= base_i_ka / np.sqrt(3)
        meas.loc[i_meas.index, "std_dev"] /= base_i_ka / np.sqrt(3)

    # Convert angle measurements (va) from degrees to radians
    meas_dg_mask = (meas.measurement_type == 'va')
    if not meas[meas_dg_mask].empty:
        meas.loc[meas_dg_mask, "value"] = np.deg2rad(meas.loc[meas_dg_mask, "value"])
        meas.loc[meas_dg_mask, "std_dev"] = np.deg2rad(meas.loc[meas_dg_mask, "std_dev"])

    # Get bus mapping from pandapower to ppc index
    map_bus = net["_pd2ppc_lookups"]["bus"]
    meas_bus = meas[(meas['element_type'] == 'bus')]

    # Drop invalid bus measurements (those that map outside the ppci bus array)
    if (map_bus[meas_bus['element'].values.astype(np.int64)] >= ppci["bus"].shape[0]).any():
        std_logger.warning("Measurement defined in pp-grid does not exist in ppci, will be deleted!")
        meas_bus = meas_bus[map_bus[meas_bus['element'].values.astype(np.int64)] < ppci["bus"].shape[0]]

    # Create empty append array for bus measurements
    bus_append = np.full((ppci["bus"].shape[0], bus_cols_se), np.nan, dtype=ppci["bus"].dtype)

    # Add bus measurements (v, va, p, q)
    _add_measurements_to_bus(net, meas_bus, bus_append, map_bus)

    # Add zero injection measurements if specified
    bus_append = _add_zero_injection(net, ppci, bus_append, zero_injection)

    # Add virtual measurements for artificial buses (from open line switches)
    new_in_line_buses = np.setdiff1d(np.arange(ppci["bus"].shape[0]), map_bus[map_bus >= 0])
    bus_append[new_in_line_buses, 2] = 0.
    bus_append[new_in_line_buses, 3] = 1e-6
    bus_append[new_in_line_buses, 4] = 0.
    bus_append[new_in_line_buses, 5] = 1e-6

    # Create empty append array for branch measurements
    branch_append = np.full((ppci["branch"].shape[0], branch_cols_se), np.nan, dtype=ppci["branch"].dtype)
    br_is_mask = ppci['internal']['branch_is']

    # Add line, trafo, and trafo3w measurements
    _add_measurements_to_line(net, branch_append, meas, br_is_mask)
    _add_measurements_to_trafo(net, branch_append, meas, br_is_mask)
    _add_measurements_to_trafo3w(net, branch_append, meas, br_is_mask)

    # Integrate new measurement columns into ppci bus matrix
    if ppci["bus"].shape[1] == bus_cols:
        ppci["bus"] = np.hstack((ppci["bus"], bus_append))
    else:
        ppci["bus"][:, bus_cols: bus_cols + bus_cols_se] = bus_append

    # Integrate new measurement columns into ppci branch matrix
    if ppci["branch"].shape[1] == branch_cols:
        ppci["branch"] = np.hstack((ppci["branch"], branch_append))
    else:
        ppci["branch"][:, branch_cols: branch_cols + branch_cols_se] = branch_append

    # Add rated power information needed for AF-WLS estimator
    if algorithm == 'af-wls':
        _add_rated_power_information_af_wls(net, ppci)

    return ppci


def pp2eppci(net, v_start=None, delta_start=None,
             calculate_voltage_angles=True, zero_injection="aux_bus",
             algorithm='wls', ppc=None, eppci=None):
    if isinstance(eppci, ExtendedPPCI):
        eppci.algorithm = algorithm
        eppci.data = _add_measurements_to_ppci(net, eppci.data, zero_injection, algorithm)
        eppci.update_meas()
        return net, ppc, eppci
    else:
        # initialize ppc
        ppc, ppci = _init_ppc(net, v_start, delta_start, calculate_voltage_angles)

        # add measurements to ppci structure
        # Finished converting pandapower network to ppci
        ppci = _add_measurements_to_ppci(net, ppci, zero_injection, algorithm)
        return net, ppc, ExtendedPPCI(ppci, algorithm)
