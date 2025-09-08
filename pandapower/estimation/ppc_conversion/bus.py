# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from pandapower.estimation.idx_bus import (VM, VM_IDX, VM_STD,
                                           VA, VA_IDX, VA_STD,
                                           P, P_IDX, P_STD,
                                           Q, Q_IDX, Q_STD,
                                           ZERO_INJ_FLAG)
from pandapower.estimation.ppc_conversion.utils import _calculate_weighted_measurements

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

std_logger = logging.getLogger(__name__)

ZERO_INJECTION_STD_DEV = 0.001

BUS_MEAS_PPCI_IX = {"v": {"VALUE": VM, "IDX": VM_IDX, "STD": VM_STD},
                    "va": {"VALUE": VA, "IDX": VA_IDX, "STD": VA_STD},
                    "p": {"VALUE": P, "IDX": P_IDX, "STD": P_STD},
                    "q": {"VALUE": Q, "IDX": Q_IDX, "STD": Q_STD}}


def _add_measurements_to_bus(meas_bus, bus_append, map_bus, no_inj_buses):
    """
        Aggregate measurements by bus index and append results to bus_append array.

        Parameters:
        - meas_bus: subset of measurement DataFrame containing only measurements at buses
        - bus_append: NumPy array to store measurements, std devs, and indices to add to ppci
        - map_bus: dict mapping bus IDs to PPCI bus indices
        - no_inj_buses = set of buses without any connection of power injection elements
        """

    # Process voltage (v) and voltage angle (va) measurements
    for meas_type in ("v", "va"):
        this_meas = meas_bus[(meas_bus.measurement_type == meas_type)]

        if this_meas.empty:
            continue

        this_meas["ppci_index"] = this_meas.element.map(lambda x: map_bus[int(x)])

        ind_map = this_meas.drop_duplicates(subset=["ppci_index"], keep="first")
        ind_map = ind_map.reset_index().set_index("ppci_index")

        # voltage measurements at the same ppci bus will be merged via a weighted average
        meas_merged = _calculate_weighted_measurements(this_meas, "ppci_index")
        meas_merged["index"] = ind_map["index"]

        bus_append[meas_merged.index, BUS_MEAS_PPCI_IX[meas_type]["VALUE"]] = meas_merged.weighted_measurement
        bus_append[meas_merged.index, BUS_MEAS_PPCI_IX[meas_type]["STD"]] = meas_merged.merged_weight
        bus_append[meas_merged.index, BUS_MEAS_PPCI_IX[meas_type]["IDX"]] = meas_merged["index"]

    # Process active (p) and reactive (q) power injections
    for meas_type in ("p", "q"):
        this_meas = meas_bus[(meas_bus.measurement_type == meas_type)]
        if this_meas.empty:
            continue

        this_meas.value *= -1

        this_meas["ppci_index"] = this_meas.element.map(lambda x: map_bus[int(x)])
        ind_map = this_meas.drop_duplicates(subset=["ppci_index"], keep="first")
        ind_map = ind_map.reset_index().set_index("ppci_index")

        # power measurements at the same pp bus will be merged via a weighted average
        this_meas = _calculate_weighted_measurements(this_meas, "element")
        this_meas["ppci_index"] = this_meas.index.map(lambda x: map_bus[int(x)])

        # power measurements at different pp buses but same ppci bus will be aggregated
        # if all measurements of the ppci bus exist or if the missing ones are zero inj
        sum_values = sum_std_dev = np.array([])
        idx = np.array([], dtype=int)
        for k in ind_map.index:
            meas_subset = this_meas[this_meas["ppci_index"]==k]
            if meas_subset.empty:
                continue
            measured_buses = set(meas_subset.index)
            buses_merged_in_ppci_index = set(np.where(map_bus==k)[0])
            no_inj_buses_in_subset = no_inj_buses & buses_merged_in_ppci_index
            if measured_buses.union(no_inj_buses_in_subset) == buses_merged_in_ppci_index:
                idx = np.append(idx, k)
                sum_values = np.append(sum_values, meas_subset["weighted_measurement"].sum())
                meas_subset["merged_weight"] = np.square(meas_subset["merged_weight"])
                sum_variance = meas_subset["merged_weight"].sum()
                sum_std_dev = np.append(sum_std_dev, np.sqrt(sum_variance))

        # sum_values = this_meas.groupby("ppci_index")["weighted_measurement"].sum()
        # this_meas["merged_weight"] = np.square(this_meas["merged_weight"])
        # sum_variance = this_meas.groupby("ppci_index")["merged_weight"].sum()
        # sum_std_dev = np.sqrt(sum_variance)

        # merged_value = sum_values.to_frame(name="sum_values")
        # merged_value["sum_std_dev"] = sum_std_dev
        # merged_value["index"] = ind_map["index"]

        bus_append[idx, BUS_MEAS_PPCI_IX[meas_type]["VALUE"]] = sum_values
        bus_append[idx, BUS_MEAS_PPCI_IX[meas_type]["STD"]] = sum_std_dev
        bus_append[idx, BUS_MEAS_PPCI_IX[meas_type]["IDX"]] = ind_map["index"][idx]


def _add_zero_injection(net, ppci, bus_append, zero_injection):
    """
    Add zero injection labels to the ppci structure and add virtual measurements to those buses
    :param net: pandapower net
    :param ppci: generated ppci
    :param bus_append: added columns to the ppci bus with zero injection label
    :param zero_injection: parameter to control which bus to be identified as zero injection
        - None: no zero injection buses added
        - "aux_bus": only auxiliary buses created in ppc 
        - "no_inj_bus": aux buses + buses without load, gen, sgen, etc.
        - "zero_pwr_bus": aux buses + all buses with a zero power (also if there is load, sgen, etc.)
    :return bus_append: added columns
    """
    bus_append[:, ZERO_INJ_FLAG] = False
    if zero_injection is not None:
        # identify aux bus as zero injection
        if net._pd2ppc_lookups['aux']:
            aux_bus_lookup = np.concatenate([v for k, v in net._pd2ppc_lookups['aux'].items() if k != 'xward'])
            aux_bus = net._pd2ppc_lookups['bus'][aux_bus_lookup]
            aux_bus = aux_bus[aux_bus < ppci["bus"].shape[0]]
            bus_append[aux_bus, ZERO_INJ_FLAG] = True

        if isinstance(zero_injection, str):
            if zero_injection in ['zero_pwr_bus', 'no_inj_bus']:
                # identify all buses with zero power and no pq measurements as zero injection
                zero_inj_bus_mask = (ppci["bus"][:, 1] == 1) & (ppci["bus"][:, 2:4] == 0).all(axis=1) & \
                                    np.isnan(bus_append[:, P:(Q_STD + 1)]).all(axis=1)
                bus_append[zero_inj_bus_mask, ZERO_INJ_FLAG] = True
                if zero_injection == 'no_inj_bus':
                    b = np.array([], dtype=np.int64)
                    pq_elements = ["load", "motor", "sgen", "storage", "ward", "xward", 
                                   "asymmetric_load", "asymmetric_sgen"]
                    bus_lookup = net["_pd2ppc_lookups"]["bus"]
                    for element in pq_elements:
                        tab = net[element]
                        if len(tab) == 0:
                            continue
                        in_service = tab["in_service"] & (net.bus["in_service"][tab["bus"].values])
                        b = np.hstack([b, tab["bus"][in_service]])
                    active_buses = np.unique(b)
                    active_buses = bus_lookup[active_buses]
                    bus_append[active_buses, ZERO_INJ_FLAG] = False
            elif zero_injection != "aux_bus":
                raise UserWarning("zero injection parameter is not correctly initialized")
        elif hasattr(zero_injection, '__iter__'):
            zero_inj_bus = net._pd2ppc_lookups['bus'][zero_injection]
            bus_append[zero_inj_bus, ZERO_INJ_FLAG] = True

        zero_inj_bus = np.argwhere(bus_append[:, ZERO_INJ_FLAG]).ravel()
        bus_append[zero_inj_bus, P] = 0
        bus_append[zero_inj_bus, P_STD] = ZERO_INJECTION_STD_DEV
        bus_append[zero_inj_bus, P_IDX] = -1
        bus_append[zero_inj_bus, Q] = 0
        bus_append[zero_inj_bus, Q_STD] = ZERO_INJECTION_STD_DEV
        bus_append[zero_inj_bus, Q_IDX] = -1
    return bus_append
   


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
