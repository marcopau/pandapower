# -*- coding: utf-8 -*-

# This code was written by Matsiushonak Siarhei and Zografos Dimitrios.
# Contributions made on 2025.

import time
from collections import defaultdict
from itertools import chain

import pandas as pd

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)


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
        *[net[element].bus.dropna().unique() for element in ["load", "sgen", "gen", "ext_grid", "ward"] if element in net]
    )

    # Return buses that are in all_buses but not in connected_buses
    return all_buses.difference(connected_buses)


def get_non_zero_inj_bus_to_ppnet_map(net, isolated_buses):
    eppci_bus_to_ppnet_map = defaultdict(set)
    for i, v in enumerate(net._pd2ppc_lookups["bus"]):
        if v != -1 and i not in isolated_buses:
            eppci_bus_to_ppnet_map[v].add(i)
    return eppci_bus_to_ppnet_map


def _create_measurements_df(elements: pd.Series, values: pd.Series, std_dev: float, measurement_type: str):
    return pd.DataFrame({
        'name': f'Setpoint_{measurement_type}_' + elements.astype(str),
        'side': None,
        'measurement_type': measurement_type,
        'element': elements,
        'element_type': 'bus',
        'value': values,
        'std_dev': std_dev
    })


def _compute_power_injections(net):
    """Computes total active power injections per bus."""
    load_grouped = net.load.groupby('bus')['p_mw'].sum()
    sgen_grouped = net.sgen.groupby('bus')['p_mw'].sum()
    gen_grouped = net.gen.groupby('bus')['p_mw'].sum()

    return pd.DataFrame({
        'load_p_mw': load_grouped,
        'sgen_p_mw': sgen_grouped,
        'gens_p_mw': gen_grouped
    }).fillna(0).assign(total_p_mw=lambda df: df['load_p_mw'] - df['sgen_p_mw'] - df['gens_p_mw'])


def create_active_power_measurements_from_setpoints(net, isolated_buses, setpoint_std_dev: float, drop_measurements=False):
    """
    Creates active power (P) measurements from setpoints for mismatched buses in the power grid model.

    Args:
        net (pandapowerNet): The power network model.
        isolated_buses (set): Set of isolated buses.
        setpoint_std_dev (float): Standard deviation for measurement noise.
        drop_measurements (bool, optional): If True, drops mismatched measurements instead of adding new ones.
    """
    # Get bus mapping from non-zero injection buses
    eppci_bus_to_ppnet_map = get_non_zero_inj_bus_to_ppnet_map(net, isolated_buses)

    # Filter P measurements of buses
    meas = net.measurement
    p_measurements = meas[(meas.measurement_type == 'p') & (meas.element_type == 'bus')].copy()

    # Map measurement elements to PPCI indices
    p_measurements["ppci_index"] = p_measurements["element"].map(
        lambda x: net._pd2ppc_lookups["bus"][int(x)]  # pylint: disable=W0212
    )

    # Group by PPCI index and get unique element counts
    p_grouped = p_measurements.groupby("ppci_index")
    series_dict = p_grouped["element"].nunique()

    # Create measurement mapping
    p_measurements_map = {ppci_bus_id: set(group.element) for ppci_bus_id, group in p_grouped}

    # Find mismatched buses where the number of measurements differs from expected
    mismatched_ppci_map = {
        ppci_bus_id: eppci_bus_to_ppnet_map[ppci_bus_id] - p_measurements_map.get(ppci_bus_id, set())
        for ppci_bus_id, num_buses in series_dict.items()
        if num_buses != len(eppci_bus_to_ppnet_map[ppci_bus_id])
    }

    candidates_buses = list(chain.from_iterable(mismatched_ppci_map.values()))

    if drop_measurements:
        # Drop mismatched measurements
        bus_measurements_to_drop = p_measurements[p_measurements.ppci_index.isin(mismatched_ppci_map.keys())].index
        net.measurement = meas[~meas.index.isin(bus_measurements_to_drop)]
        return

    # Aggregate active power injections by bus
    injections = _compute_power_injections(net)

    # Filter injections for mismatched buses
    filtered_injections = injections.loc[candidates_buses, 'total_p_mw'].reset_index()

    # Create new power measurements
    new_p_measurements = _create_measurements_df(
        filtered_injections['bus'],
        filtered_injections['total_p_mw'],
        setpoint_std_dev,
        "p",
    )

    # Append new measurements to net
    net.measurement = pd.concat([meas, new_p_measurements], ignore_index=True)


def create_reactive_power_measurements_from_setpoints(net, isolated_buses, setpoint_std_dev: float, drop_measurements=False):
    """
    Creates reactive power (Q) measurements from setpoints for mismatched buses in the power grid model.

    Args:
        net (pandapowerNet): The power network model.
        isolated_buses (set): Set of isolated buses.
        setpoint_std_dev (float): Standard deviation for measurement noise.
        drop_measurements (bool, optional): If True, drops mismatched measurements instead of adding new ones.
    """
    # Get bus mapping from non-zero injection buses
    eppci_bus_to_ppnet_map = get_non_zero_inj_bus_to_ppnet_map(net, isolated_buses)

    # Filter Q measurements of buses
    meas = net.measurement
    q_measurements = meas[(meas.measurement_type == 'q') & (meas.element_type == 'bus')].copy()

    # Map measurement elements to PPCI indices
    q_measurements["ppci_index"] = q_measurements["element"].map(
        lambda x: net._pd2ppc_lookups["bus"][int(x)]
    )

    # Group by PPCI index and get unique element counts
    q_grouped = q_measurements.groupby("ppci_index")
    series_dict = q_grouped["element"].nunique()

    # Create measurement mapping
    q_measurements_map = {ppci_bus_id: set(group.element) for ppci_bus_id, group in q_grouped}

    # Find mismatched buses where the number of measurements differs from expected
    mismatched_ppci_map = {
        ppci_bus_id: eppci_bus_to_ppnet_map[ppci_bus_id] - q_measurements_map.get(ppci_bus_id, set())
        for ppci_bus_id, num_buses in series_dict.items()
        if num_buses != len(eppci_bus_to_ppnet_map[ppci_bus_id])
    }

    candidates_buses = list(chain.from_iterable(mismatched_ppci_map.values()))

    if drop_measurements:
        # Drop mismatched measurements
        net.measurement = _drop_mismatched_q_measurements(meas, q_measurements, mismatched_ppci_map)
        return

    # Remove reactive power measurements at generator buses
    meas = _remove_q_measurements_at_gen_buses(net, meas, q_measurements, candidates_buses)

    # Aggregate reactive power injections by bus
    injections = _compute_reactive_power_injections(net)

    # Filter injections for mismatched buses
    filtered_injections = injections.loc[candidates_buses, 'total_q_mvar'].reset_index()

    # Create new Q power measurements
    new_q_measurements = _create_measurements_df(
        filtered_injections['bus'],
        filtered_injections['total_q_mvar'],
        setpoint_std_dev,
        "q",
    )

    # Append new measurements to net
    net.measurement = pd.concat([meas, new_q_measurements], ignore_index=True)


def _drop_mismatched_q_measurements(meas, q_measurements, mismatched_ppci_map):
    """Drops mismatched reactive power measurements from the measurement DataFrame."""
    bus_measurements_to_drop = q_measurements[q_measurements.ppci_index.isin(mismatched_ppci_map.keys())].index
    return meas[~meas.index.isin(bus_measurements_to_drop)]


def _remove_q_measurements_at_gen_buses(net, meas, q_measurements, candidates_buses):
    """Removes reactive power measurements at buses where generators are present."""
    buses_with_gen = net.gen[net.gen['bus'].isin(candidates_buses)]
    if not buses_with_gen.empty:
        ppci_node_buses_with_gen = [net._pd2ppc_lookups["bus"][int(i)] for i in buses_with_gen.bus.unique()]
        all_net_buses_with_gen = q_measurements[q_measurements.ppci_index.isin(ppci_node_buses_with_gen)].element
        return meas[~(
                (meas['element'].isin(all_net_buses_with_gen)) &
                (meas["measurement_type"] == 'q') &
                (meas["element_type"] == 'bus')
        )]
    return meas


def _compute_reactive_power_injections(net):
    """Computes total reactive power injections per bus."""
    load_grouped = net.load.groupby('bus')['q_mvar'].sum()
    sgen_grouped = net.sgen.groupby('bus')['q_mvar'].sum()

    return pd.DataFrame({
        'load_q_mvar': load_grouped,
        'sgen_q_mvar': sgen_grouped,
    }).fillna(0).assign(total_q_mvar=lambda df: df['load_q_mvar'] - df['sgen_q_mvar'])


def _calculate_weighted_measurements(measurements):
    """Calculate weighted measurements for active power (p)."""
    measurements["weight"] = 1 / measurements["std_dev"]
    measurements["weighted_value"] = measurements["weight"] * measurements["value"]

    merged_weight = 1 / measurements.groupby("element")["weight"].sum()
    merged_value = measurements.groupby("element")["weighted_value"].sum()

    return merged_weight * merged_value


def _get_remaining_buses(net, p_mw_meas, zero_inj_buses):
    """Get buses that are not in measurement elements or zero-injection buses."""
    return net.bus[~((net.bus.index.isin(p_mw_meas.element)) | (net.bus.index.isin(zero_inj_buses)))]


def _calculate_power_injections(net, rest_buses):
    """Compute power injections (load, sgen, gen) for remaining buses."""
    load_grouped = net.load[net.load.bus.isin(rest_buses.index)].groupby('bus')['p_mw'].sum()
    sgen_grouped = net.sgen[net.sgen.bus.isin(rest_buses.index)].groupby('bus')['p_mw'].sum()
    gen_grouped = net.gen[net.gen.bus.isin(rest_buses.index)].groupby('bus')['p_mw'].sum()

    injections = pd.DataFrame({
        'load_p_mw': load_grouped,
        'sgen_p_mw': sgen_grouped,
        'gens_p_mw': gen_grouped
    }).fillna(0)

    injections['total_p_mw'] = injections['load_p_mw'] - injections['sgen_p_mw'] - injections['gens_p_mw']
    return injections['total_p_mw']


def assign_active_power(net):
    """Assign active power (p_mw) to buses based on weighted measurements and injections."""
    st_main = time.perf_counter()

    res_bus_est = net.res_bus_est
    origin_columns = res_bus_est.columns.tolist()

    # Get active power (p) measurements
    p_mw_meas = net.measurement.query("element_type == 'bus' and measurement_type == 'p'")

    # Compute weighted values
    result = _calculate_weighted_measurements(p_mw_meas)

    # Get zero-injection buses and remaining buses
    zero_inj_buses = get_pp_zero_injection_buses(net)
    rest_buses = _get_remaining_buses(net, p_mw_meas, zero_inj_buses)

    # Compute power injections
    injections = _calculate_power_injections(net, rest_buses)

    # Merge results
    result = pd.concat([result, injections], ignore_index=False)

    # Update res_bus_est
    res_bus_est["ppci_index"] = res_bus_est.index.map(lambda x: net._pd2ppc_lookups["bus"][int(x)])
    res_bus_est["measurement_value"] = result.fillna(0)

    # Compute sum of measurements per ppci_index
    sum_meas = res_bus_est.groupby("ppci_index")["measurement_value"].sum()
    res_bus_est["sum_meas"] = res_bus_est["ppci_index"].map(sum_meas)

    # Normalize and assign active power
    res_bus_est["p_mw"] = res_bus_est["measurement_value"] / res_bus_est["sum_meas"] * res_bus_est["p_mw"]

    # Restore original columns and update net
    net.res_bus_est = res_bus_est[origin_columns]

    print(f"p assign {time.perf_counter() - st_main}")


def _calculate_reactive_power_injections(net, rest_buses):
    """Compute reactive power injections (load, sgen) for remaining buses."""
    load_grouped = net.load[net.load.bus.isin(rest_buses.index)].groupby('bus')['q_mvar'].sum()
    sgen_grouped = net.sgen[net.sgen.bus.isin(rest_buses.index)].groupby('bus')['q_mvar'].sum()

    injections = pd.DataFrame({
        'load_q_mvar': load_grouped,
        'sgen_q_mvar': sgen_grouped,
    }).fillna(0)

    injections['total_q_mvar'] = injections['load_q_mvar'] - injections['sgen_q_mvar']
    return injections['total_q_mvar']


def assign_reactive_power(net):
    """Assign reactive power (q_mvar) to buses based on weighted measurements and injections."""
    st_main = time.perf_counter()

    res_bus_est = net.res_bus_est
    origin_columns = res_bus_est.columns.tolist()

    # Get reactive power (q) measurements
    q_mvar_meas = net.measurement.query("element_type == 'bus' and measurement_type == 'q'")

    # Compute weighted values
    result = _calculate_weighted_measurements(q_mvar_meas)

    # Get zero-injection buses and remaining buses
    zero_inj_buses = get_pp_zero_injection_buses(net)
    rest_buses = net.bus[~(
            (net.bus.index.isin(q_mvar_meas.element)) |
            (net.bus.index.isin(zero_inj_buses)) |
            (net.bus.index.isin(net.gen.bus))
    )]

    # Compute power injections
    injections = _calculate_reactive_power_injections(net, rest_buses)

    # Merge results
    result = pd.concat([result, injections], ignore_index=False)

    # Update res_bus_est
    res_bus_est["ppci_index"] = res_bus_est.index.map(lambda x: net._pd2ppc_lookups["bus"][int(x)])
    res_bus_est["measurement_value"] = result.fillna(0)

    # Compute sum of measurements per ppci_index
    sum_meas = res_bus_est.groupby("ppci_index")["measurement_value"].sum()
    res_bus_est["sum_meas"] = res_bus_est["ppci_index"].map(sum_meas)

    # Normalize and assign reactive power
    res_bus_est["q_mvar"] = res_bus_est["measurement_value"] / res_bus_est["sum_meas"] * res_bus_est["q_mvar"]

    # Restore original columns and update net
    net.res_bus_est = res_bus_est[origin_columns]

    print(f"q assign {time.perf_counter() - st_main}")
