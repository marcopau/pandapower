import numpy as np
from pandapower.estimation.util import estimate_voltage_vector


def _initialize_voltage(net, init):
    v_start, delta_start = None, None
    if init == 'results':
        v_start, delta_start = 'results', 'results'
    elif init == 'slack':
        res_bus = estimate_voltage_vector(net)
        v_start = res_bus.vm_pu.values
        delta_start = res_bus.va_degree.values
    elif init != 'flat':
        raise UserWarning("Unsupported init value. Using flat initialization.")
    return v_start, delta_start


def _calculate_weighted_measurements(measurements, grouped_column: str):
    """Calculate weighted measurements."""
    measurements["weight"] = 1 / (measurements["std_dev"] ** 2)
    measurements["weighted_value"] = measurements["weight"] * measurements["value"]

    merged_weight = 1 / measurements.groupby(grouped_column)["weight"].sum()
    merged_value = measurements.groupby(grouped_column)["weighted_value"].sum()

    merged_value = merged_value.to_frame(name="weighted_measurement")
    merged_value["merged_weight"] = merged_weight
    merged_value["weighted_measurement"] *= merged_value["merged_weight"]
    merged_value["merged_weight"] = np.sqrt(merged_weight)

    return merged_value


def _get_no_inj_buses(net):
    """
    Identifies buses that have no power injection (i.e., not connected to loads, generators, external grids, etc.).
    Args:
        net (pandapowerNet): The power network model.
    Returns:
        set: A set of buses without any connected injection.
    """

    all_buses = set(net.bus.index)
    # Collect all connected buses from various components
    connected_buses = set().union(
        *[net[element].bus.dropna().unique() for element in ["load", "motor", "sgen", "gen", "ext_grid", "ward", "xward",
                                                            "storage", "asymmetric_load", "asymmetric_sgen"] if
                                                            element in net])

    # Return buses that are in all_buses but not in connected_buses
    return all_buses.difference(connected_buses)
