import numpy as np

from pandapower.estimation.util import estimate_voltage_vector


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


def map_measurement_to_bus(net, measurements):
    """
        Maps each measurement to its corresponding bus in the network.
    """

    def _resolve_bus(row):
        side = row["side"]
        element = row["element"]
        element_type = row["element_type"]

        if element_type == "line":
            return net[element_type].at[element, f"{side}_bus"]
        elif element_type == "switch":
            switch_side_map = {"from": "element", "to": "bus"}
            return net[element_type].at[element, f"{switch_side_map[side]}"]
        elif element_type in ("trafo", "trafo3w"):
            return net[element_type].at[element, f"{side}_bus"]
        elif element_type == "shunt":
            return net[element_type].at[element, "bus"]
        else:
            return side  # Fallback, possibly not a valid bus

    return measurements.apply(_resolve_bus, axis=1)


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
