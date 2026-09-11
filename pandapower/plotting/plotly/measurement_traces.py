from collections import defaultdict

import pandas as pd

from pandapower.auxiliary import pandapowerNet
from pandapower.plotting.plotly.traces import create_bus_trace
from typing import Any


_MEASUREMENT_UNITS: dict[str, str] = {
    "v": "p.u.",
    "va": "°",
    "p": "MW",
    "q": "MVAr",
    "i": "kA",
    "ia": "°",
}

def _side_matches(side: str | int | float | None, side_name: str, bus: int) -> bool:
    r"""
    Check whether a measurement side refers to a given terminal or bus.

    A measurement side can be specified either by its terminal name, such as ``"from"``, ``"to"``, ``"hv"``, ``"mv"``,
    or ``"lv"``, or by the index of the corresponding bus.

    Parameters:
        side:
            Side specification stored in the measurement table. This can be a terminal name, a bus index, or ``None``.
        side_name:
            Expected terminal name, for example ``"from"``, ``"to"``, ``"hv"``, ``"mv"``, or ``"lv"``.
        bus:
            Index of the bus connected to the terminal.

    Returns:
        ``True`` if ``side`` refers to the specified terminal or bus; otherwise ``False``.
    """

    if side is None or pd.isna(side):
        return False

    side = str(side).strip().lower()
    bus = int(bus)

    valid_values = {side_name, f"{side_name}_bus", str(bus), f"{bus}.0"}

    return side in valid_values


def _get_measurement_buses(net: pandapowerNet, measurement: pd.Series) -> list[int]:
    r"""
    Determine the buses associated with a measurement.

    Bus measurements are assigned directly to their referenced bus. Branch measurements are assigned to the terminal bus
    specified by ``side``. If no valid side is available for a branch measurement, all terminal buses of the
    corresponding element are returned.

    The following element types are supported:

        * ``"bus"``
        * ``"line"``
        * ``"trafo"``
        * ``"trafo3w"``

    Parameters:
        net: pandapower network containing the measured elements.
        measurement:
            Row from ``net.measurement`` containing at least ``element_type``, ``element``, and optionally ``side``.

    Returns:
        List of bus indices associated with the measurement. An empty list is returned if the element type is
        unsupported or the referenced element does not exist.
    """

    element_type = str(measurement["element_type"]).strip().lower()
    element = int(measurement["element"])
    side = measurement.get("side")

    if element_type == "bus":
        return [element] if element in net.bus.index else []

    if element_type == "line":
        if element not in net.line.index:
            return []

        line = net.line.loc[element]

        if _side_matches(side, "from", line.from_bus):
            return [int(line.from_bus)]

        if _side_matches(side, "to", line.to_bus):
            return [int(line.to_bus)]

        return [int(line.from_bus), int(line.to_bus)]

    if element_type == "trafo":
        if element not in net.trafo.index:
            return []

        trafo = net.trafo.loc[element]

        if _side_matches(side, "hv", trafo.hv_bus):
            return [int(trafo.hv_bus)]

        if _side_matches(side, "lv", trafo.lv_bus):
            return [int(trafo.lv_bus)]

        return [int(trafo.hv_bus), int(trafo.lv_bus)]

    if element_type == "trafo3w":
        if element not in net.trafo3w.index:
            return []

        trafo = net.trafo3w.loc[element]

        if _side_matches(side, "hv", trafo.hv_bus):
            return [int(trafo.hv_bus)]

        if _side_matches(side, "mv", trafo.mv_bus):
            return [int(trafo.mv_bus)]

        if _side_matches(side, "lv", trafo.lv_bus):
            return [int(trafo.lv_bus)]

        return [
            int(trafo.hv_bus),
            int(trafo.mv_bus),
            int(trafo.lv_bus),
        ]
    return []


def _format_number(value: float | None) -> str:
    r"""
    Format a numeric measurement value for hover information.

    Parameters:
        value: Numeric value to format. ``None`` and missing values are represented by an en dash.

    Returns:
        String representation with five significant digits, or ``"–"`` if the value is missing.
    """

    if value is None or pd.isna(value):
        return "–"

    return f"{value:.5g}"


def _create_measurement_hovertext(net: pandapowerNet, bus: int, measurements: list[tuple[int, pd.Series]]) -> str:
    r"""
    Create Plotly hover information for measurements assigned to a bus.

    Parameters:
        net: pandapower network containing the bus and measurement data.
        bus: Index of the bus for which the hover information is generated.
        measurements: Sequence of ``(measurement_index, measurement)`` tuples assigned to the bus.

    Returns:
        HTML-formatted string containing the bus and measurement information.
    """

    bus_name = net.bus.at[bus, "name"]

    lines = [f"<b>Bus {bus} Measurements</b>", f"bus name: {bus_name}", f"number of measurements: {len(measurements)}"]

    for measurement_index, measurement in measurements:
        measurement_type = str(measurement["measurement_type"]).strip().lower()

        element_type = measurement["element_type"]
        element = int(measurement["element"])
        side = measurement.get("side")
        measurement_name = measurement.get("name")

        unit = _MEASUREMENT_UNITS.get(measurement_type, "")

        side_text = str(side) if side is not None and pd.notna(side) else "–"
        element_text = (
            f"{element_type}-{element}" if side_text in ("-", "–") else f"{element_type}-{element}-{side_text}"
        )

        measurement_name_text = (
            str(measurement_name) if measurement_name is not None and pd.notna(measurement_name) else "–"
        )

        lines.extend([
            f"<br><b>index: {measurement_index}</b>",
            f"name: {measurement_name_text}",
            f"type: {measurement_type}",
            f"element: {element_text}",
            f"measurement value: {_format_number(measurement.get('value'))} {unit}",
            f"standard deviation: {_format_number(measurement.get('std_dev'))}"
        ])

    return "<br>".join(lines)


def create_measurement_trace(
        net: pandapowerNet, size: float =18, color: str = "red", symbol: str = "diamond"
) -> dict[str, Any] | None:
    r"""
    Create a Plotly trace highlighting measurement locations.

    Measurements from ``net.measurement`` are assigned to their corresponding buses. Bus measurements are placed
    directly at the measured bus. Line and transformer measurements are placed at the terminal bus indicated by their
    ``side`` value.

    If no valid side is specified for a branch measurement, all terminal buses of the element are highlighted. Multiple
    measurements assigned to the same bus are combined into a single marker and displayed together in its hover
    information.
    Unsupported measurement element types are ignored. A branch measurement without a valid ``side`` is assigned to all
    terminal buses of the corresponding branch.

    Parameters:
        net: pandapower network containing ``bus``, ``measurement``, and the referenced branch element tables.
        size: Size of the Plotly measurement markers.
        color: Plotly-compatible color of the measurement markers.
        symbol: Plotly marker symbol, for example ``"diamond"``, ``"circle"``, or ``"square"``.

    Returns:
        Plotly trace containing the measurement markers and hover information. Returns ``None`` if the measurement table
        is missing or empty, or if no measurement can be assigned to a valid bus.

    Example:
        >>> import simbench as sb
        >>> import os
        >>> from pandapower.topology.create_graph import create_nxgraph
        >>> from pandapower.plotting.generic_geodata import create_generic_coordinates
        >>> from pandapower.plotting.plotly.measurement_traces import create_measurement_trace
        >>> from pandapower.plotting.plotly import simple_plotly

        >>> net_sb = sb.get_simbench_net("1-MV-comm--0-sw")
        >>> graph = create_nxgraph(net_sb)
        >>> create_generic_coordinates(net_sb, graph, overwrite=True)
        >>> meas_traces = create_measurement_trace(net_sb)
        >>> fig = simple_plotly(net_sb, filename=f"path/to/file.html", additional_traces=meas_traces)
    """

    if "measurement" not in net or net.measurement.empty:
        return None

    measurements_by_bus = defaultdict(list)

    for measurement_index, measurement in net.measurement.iterrows():
        buses = _get_measurement_buses(net, measurement)

        for bus in buses:
            if bus in net.bus.index:
                measurements_by_bus[bus].append(
                    (measurement_index, measurement)
                )

    if not measurements_by_bus:
        return None

    measurement_buses = list(measurements_by_bus)

    hoverinfo = pd.Series({
        bus: _create_measurement_hovertext(
            net,
            bus,
            measurements_by_bus[bus],
        )
        for bus in measurement_buses
    })

    return create_bus_trace(
        net,
        buses=measurement_buses,
        size=size,
        patch_type=symbol,
        color=color,
        infofunc=hoverinfo,
        trace_name="measuring point",
        legendgroup="measurements",
    )