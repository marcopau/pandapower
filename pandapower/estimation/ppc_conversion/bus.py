# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from collections import defaultdict
from itertools import chain

import numpy as np

from pandapower.auxiliary import pandapowerNet
from pandapower.estimation.idx_bus import (VM, VM_IDX, VM_STD,
                                           VA, VA_IDX, VA_STD,
                                           P, P_IDX, P_STD,
                                           Q, Q_IDX, Q_STD,
                                           ZERO_INJ_FLAG)
from pandapower.estimation.ppc_conversion.utils import _calculate_weighted_measurements

ZERO_INJECTION_STD_DEV = 0.001

BUS_MEAS_PPCI_IX = {"v": {"VALUE": VM, "IDX": VM_IDX, "STD": VM_STD},
                    "va": {"VALUE": VA, "IDX": VA_IDX, "STD": VA_STD},
                    "p": {"VALUE": P, "IDX": P_IDX, "STD": P_STD},
                    "q": {"VALUE": Q, "IDX": Q_IDX, "STD": Q_STD}}


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
                        in_service = tab["in_service"]
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


def get_non_zero_inj_bus_to_ppnet_map(net, isolated_buses):
    eppci_bus_to_ppnet_map = defaultdict(set)
    for i, v in enumerate(net._pd2ppc_lookups["bus"]):
        if v != -1 and i not in isolated_buses:
            eppci_bus_to_ppnet_map[v].add(i)
    return eppci_bus_to_ppnet_map


def find_buses_missing_power_measurements(
        net,
        isolated_buses: set[int],
        measurement_type: str,
) -> list[int]:
    """
    Identify buses with non-zero setpoint injections that lack a full set of
    power measurements (active or reactive) within their PPCI node.

    """
    # 1. Map PPCI bus indices to pandapower bus indices for all non-zero injections
    ppci_to_ppnet = get_non_zero_inj_bus_to_ppnet_map(
        net, isolated_buses
    )

    # 2. Filter only the specified power measurements on buses
    meas = net.measurement
    bus_meas = meas[
        (meas.measurement_type == measurement_type) & (meas.element_type == 'bus')
        ].copy()

    # 3. Associate each measurement with its PPCI index
    bus_meas['ppci_index'] = bus_meas['element'].astype(int).map(
        lambda idx: net._pd2ppc_lookups['bus'][idx]  # pylint: disable=W0212
    )

    # 4. Group measurements by PPCI index and build set of measured bus indices
    grouped = bus_meas.groupby('ppci_index')['element']
    measured_map = {
        ppci: set(values) for ppci, values in grouped
    }

    # 5. Identify PPCI nodes with missing measurements
    missing_map = {}
    for ppci, expected_buses in ppci_to_ppnet.items():
        actual_buses = measured_map.get(ppci, set())
        if len(actual_buses) != len(expected_buses):
            # Determine which buses lack the measurement
            missing_map[ppci] = expected_buses

    # 6. Flatten the sets of missing bus indices into a list
    missing_buses = list(chain.from_iterable(missing_map.values()))
    return missing_buses


def get_pp_zero_injection_buses(net):
    """
    Identifies buses that have zero power injection (i.e., not connected to loads, generators, or external grids).

    Args:
        net (pandapowerNet): The power network model.

    Returns:
        set: A set of isolated buses with zero injection.
    """
    all_buses = set(net.bus.index)

    # Collect all connected buses from various components
    connected_buses = set().union(
        *[net[element].bus.dropna().unique() for element in ["load", "sgen", "gen", "ext_grid", "ward"] if
          element in net]
    )

    # Return buses that are in all_buses but not in connected_buses
    return all_buses.difference(connected_buses)


def _add_measurements_to_bus(net, meas_bus, bus_append, map_bus):
    """
        Aggregate measurements by bus index and append results to bus_append array.

        Parameters:
        - meas_bus: subset of measurement DataFrame containing only measurements at buses
        - bus_append: NumPy array to store measurements, std devs, and indices to add to ppci
        - map_bus: dict mapping bus IDs to PPCI bus indices
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
    isolated_buses = get_pp_zero_injection_buses(net)
    for meas_type in ("p", "q"):
        buses_missing_power_measurements = find_buses_missing_power_measurements(net, isolated_buses, meas_type)
        this_meas = meas_bus[
            (meas_bus.measurement_type == meas_type)
            & (~meas_bus.element.isin(buses_missing_power_measurements))
            ]

        this_meas.value *= -1

        if this_meas.empty:
            continue

        this_meas["ppci_index"] = this_meas.element.map(lambda x: map_bus[int(x)])
        ind_map = this_meas.drop_duplicates(subset=["ppci_index"], keep="first")
        ind_map = ind_map.reset_index().set_index("ppci_index")

        # power measurements at the same pp bus will be merged via a weighted average
        this_meas = _calculate_weighted_measurements(this_meas, "element")
        this_meas["ppci_index"] = this_meas.index.map(lambda x: map_bus[int(x)])

        # power measurements at different pp buses but same ppci bus will be aggregated
        sum_values = this_meas.groupby("ppci_index")["weighted_measurement"].sum()
        this_meas["merged_weight"] = np.square(this_meas["merged_weight"])
        sum_variance = this_meas.groupby("ppci_index")["merged_weight"].sum()
        sum_std_dev = np.sqrt(sum_variance)

        merged_value = sum_values.to_frame(name="sum_values")
        merged_value["sum_std_dev"] = sum_std_dev
        merged_value["index"] = ind_map["index"]

        bus_append[merged_value.index, BUS_MEAS_PPCI_IX[meas_type]["VALUE"]] = merged_value.sum_values
        bus_append[merged_value.index, BUS_MEAS_PPCI_IX[meas_type]["STD"]] = merged_value.sum_std_dev
        bus_append[merged_value.index, BUS_MEAS_PPCI_IX[meas_type]["IDX"]] = merged_value["index"]
