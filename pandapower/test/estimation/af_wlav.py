# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import numpy as np
import pandas as pd
import os
import simbench as sb
import time

from datetime import timedelta
from tqdm import tqdm
from dotenv import load_dotenv

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt

# imports from pandapower
import pandapower.networks as pn
from pandapower import to_pickle, from_pickle
from pandapower.run import runpp
from pandapower.estimation import estimate
from pandapower.create import (create_measurement, create_empty_network, create_bus, create_ext_grid,
                               create_line_from_parameters, create_load, create_sgen)
from pandapower.auxiliary import pandapowerNet
from pandapower.test.estimation.test_lav_estimation import _r
# from pandapower.plotting import to_html as pp_to_html
from pandapower.plotting.plotly import simple_plotly  # , vlevel_plotly
from pandapower.topology.create_graph import create_nxgraph
from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting.plotly.measurement_traces import create_measurement_trace


# begin functions
def get_non_empty_table_names(net: pandapowerNet) -> list[str]:
    """
    Return the names of all non-empty DataFrame tables in a pandapower network.
    """
    table_names: list[str] = []

    for name, value in net.items():
        if isinstance(value, pd.DataFrame) and not value.empty:
            table_names.append(name)

    return table_names


def deactivate_sgen_by_type(
        net: pandapowerNet,
        sgen_type: str = "Biomass_MV"
) -> None:
    r"""
    Deactivate all static generators of a specified type.

    The selected static generators are excluded from power-flow calculations and state estimation by setting their
    ``in_service`` status to ``False``. The network is modified in-place.

    Parameters:
        net: Pandapower network that is modified in-place.
        sgen_type: Static-generator type to deactivate.

    Raises: KeyError: If the ``type`` column does not exist in ``net.sgen``.

    Returns: None
    """
    if "type" not in net.sgen.columns:
        raise KeyError("The column 'type' does not exist in net.sgen.")

    # Select static generators with the specified type
    sgen_type_mask = net.sgen["type"].eq(sgen_type)

    # Exclude the selected static generators from calculations
    net.sgen.loc[sgen_type_mask, "in_service"] = False


def _plot_bus_voltage(net: pandapowerNet, close_b: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Ergebnisse des Lastflusses
    bus_voltage_pu = net.res_bus["vm_pu"]
    ax.plot(bus_voltage_pu.index, bus_voltage_pu, marker="o", linestyle="-", label="Lastfluss")

    # Ergebnisse der State Estimation, falls vorhanden
    if "res_bus_est" in net and not net.res_bus_est.empty:
        estimated_voltage_pu = net.res_bus_est["vm_pu"].dropna()

        ax.plot(estimated_voltage_pu.index, estimated_voltage_pu, marker="x", linestyle="--", label="State Estimation")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Nennspannung")

    ax.set_xlabel("bus index")
    ax.set_ylabel("voltage [p.u.]")
    ax.set_title("bus voltages")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()

    if close_b:
        plt.close(fig)


def _check_net_limits(net: pandapowerNet) -> list[dict[str, str]]:
    """
    Check whether voltage and loading limits are violated in the network.

    The function checks bus and transformer voltages against the permissible voltage range from 0.90 p.u. to 1.10 p.u.
    It also checks line and transformer loading against the maximum permissible loading of 100 %.

    Arguments:
        net: Pandapower network containing the power flow results in ``res_bus``, ``res_line`` and ``res_trafo``.

    Returns:
        A list containing one dictionary if at least one limit is violated. The dictionary contains the key
        ``"violation"`` and a comma-separated string describing all detected violations. An empty list is returned
        if no limit is violated.
    """
    violation_records: list[dict[str, str]] = []
    # check limits of voltage and loads +-10% of p.u and 100% of loads.
    voltage_violation = ((net.res_bus["vm_pu"] < 0.90) | (net.res_bus["vm_pu"] > 1.10)).any()
    line_loading_violation = (net.res_line["loading_percent"] > 100.0).any()

    trafo_voltage_violation = (
            (net.res_trafo["vm_lv_pu"] < 0.90) | (net.res_trafo["vm_lv_pu"] > 1.10)
    ).any()
    trafo_loading_violation = (net.res_trafo["loading_percent"] > 100.0).any()

    reasons: list[str] = []  # reasons for not respected limits/boundaries/tolerances

    if voltage_violation:
        reasons.append("voltage tolerance")
    if line_loading_violation:
        reasons.append("line utilization")

    if trafo_voltage_violation:
        reasons.append("voltage tolerance transformer")
    if trafo_loading_violation:
        reasons.append("line utilization transformer")

    has_violation = bool(reasons)
    if has_violation:
        violation_records.append(
            {"violation": ", ".join(reasons)}
        )
    return violation_records


def _check_plot_net(net: pandapowerNet) -> list[dict[str, str]]:
    runpp(net)
    violation_records = _check_net_limits(net)
    _plot_bus_voltage(net)
    return violation_records


def apply_case(
        net: pandapowerNet,
        case_values: dict[tuple[str, str], pd.DataFrame],
        case: str = "lPV"
) -> None:
    r"""
        Apply a SimBench study case to a pandapower network.

        The values of the selected study case are assigned to the corresponding pandapower elements and parameters.
        Empty element tables are skipped.

        Parameters:
            net: pandapower network that is modified in-place.
            case_values: Absolute SimBench study case values as returned by :func:`simbench.get_absolute_values`.
            case: Name of the study case to apply, e.g. ``bc``, ``"hL"``, ``n1``, ``hW``, ``hPV``, ``lW``, ``"lPV"``.

        Returns:
            None.
        """
    for (element, parameter), values in case_values.items():

        # take only existing elements
        if values.shape[1] == 0:
            continue
        # set value
        net[element].loc[values.columns, parameter] = values.loc[case]


def _add_measurements_af(
        net_base: pandapowerNet,
        seed_m: int | None = None,
        measurement_interval: int = 1,
        rv: float = .01,
        rp: float = .03,
        rq: float = .03
) -> None:
    """
    Add measurements to test gird for state estimation with allocation factors.

    With the parameter ``measurement_interval`` various properties can be set. Fully observable (1),
    non-observable (>1). The measurement uncertainty can be set with ``rv, rp, rq`` for voltage, active and reactive
    power. To the net will add measurements which comes from powerflow calculation results with statistic uncertainties.

    Arguments:
        net_base: net with or without load flow calculations can be imported.
        seed_m: Attention if None, no seed will be used different to normal use case.
        measurement_interval:
            Integer greater than or equal to 1. Measurements are added to every `measurement_interval`-th bus (according
            to the bus order in `net_base.res_bus`). Higher values result in fewer measurements. The allocation is
            systematic and not random.
        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage
        rp: standard deviation to apply a multiplicative perturbation to quantities for active power
        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power

    Returns:
        None
    """
    if seed_m is not None:
        np.random.seed(seed_m)

    if not net_base.converged:
        runpp(net_base)

    # 2. create measurements ONCE on the base network
    for i, (bus, row) in enumerate(net_base.res_bus.iterrows()):
        if i % measurement_interval != 0:
            continue

        create_measurement(
            net_base,
            meas_type="v",
            element_type="bus",
            value=row.vm_pu * _r(rv),
            std_dev=max(.001, abs(rv * row.vm_pu)),
            element=int(bus)
        )
        create_measurement(
            net=net_base,
            meas_type="p",
            element_type="bus",
            value=row.p_mw * _r(rp),
            std_dev=max(.001, abs(rp * row.p_mw)),
            element=int(bus)
        )
        create_measurement(
            net=net_base,
            meas_type="q",
            element_type="bus",
            value=row.q_mvar * _r(rq),
            std_dev=max(.001, abs(rq * row.q_mvar)),
            element=int(bus)
        )


def _fill_measurement_values_from_powerflow(
        net: pandapowerNet,
        seed_m: int | None = None,
        rv: float = 0.01,
        ri: float = 0.01,
        rp: float = 0.03,
        rq: float = 0.03
) -> None:
    """
    Fill existing empty pandapower measurements from simbench net with values from power flow results.

    Existing entries in net.measurement are not recreated. Only `value` and `std_dev` are updated. The measurement
    values are taken from the corresponding result tables and multiplied by `_r(...)` to add measurement uncertainty.

    Parameters:
        net: net with or without load flow calculations can be imported.
        seed_m: Attention if None, no seed will be used different to normal use case
        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage
        ri: standard deviation to apply a multiplicative perturbation to quantities for current
        rp: standard deviation to apply a multiplicative perturbation to quantities for active power
        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power

    Returns: None
    """

    if seed_m is not None:
        np.random.seed(seed_m)

    if not net.converged or net.res_bus.empty:
        runpp(net)

    for idx, meas in net.measurement.iterrows():
        meas_type = meas["measurement_type"]
        element_type = meas["element_type"]
        element = int(meas["element"])
        side = meas.get("side", None)

        value: float | None = None
        std_dev: float | None = None

        # --- bus measurements ---
        if element_type == "bus":
            if meas_type == "v":
                base_value = float(net.res_bus.at[element, "vm_pu"])
                value = base_value * _r(rv)
                std_dev = max(0.001, abs(rv * base_value))

            elif meas_type == "p":
                base_value = float(net.res_bus.at[element, "p_mw"])
                value = base_value * _r(rp)
                std_dev = max(0.001, abs(rp * base_value))

            elif meas_type == "q":
                base_value = float(net.res_bus.at[element, "q_mvar"])
                value = base_value * _r(rq)
                std_dev = max(0.001, abs(rq * base_value))

        # --- line measurements ---
        elif element_type == "line":
            if side == "from":
                prefix = "from"
            elif side == "to":
                prefix = "to"
            else:
                raise ValueError(f"Invalid or missing side for line measurement at index {idx}: {side}")

            if meas_type == "p":
                base_value = float(net.res_line.at[element, f"p_{prefix}_mw"])
                value = base_value * _r(rp)
                std_dev = max(0.001, abs(rp * base_value))

            elif meas_type == "q":
                base_value = float(net.res_line.at[element, f"q_{prefix}_mvar"])
                value = base_value * _r(rq)
                std_dev = max(0.001, abs(rq * base_value))

            elif meas_type == "i":
                base_value = float(net.res_line.at[element, f"i_{prefix}_ka"])
                value = base_value * _r(ri)
                std_dev = max(0.001, abs(ri * base_value))

        # --- transformer measurements ---
        elif element_type == "trafo":
            if side not in ["hv", "lv"]:
                raise ValueError(f"Invalid or missing side for trafo measurement at index {idx}: {side}")

            if meas_type == "p":
                base_value = float(net.res_trafo.at[element, f"p_{side}_mw"])
                value = base_value * _r(rp)
                std_dev = max(0.001, abs(rp * base_value))

            elif meas_type == "q":
                base_value = float(net.res_trafo.at[element, f"q_{side}_mvar"])
                value = base_value * _r(rq)
                std_dev = max(0.001, abs(rq * base_value))

            elif meas_type == "i":
                base_value = float(net.res_trafo.at[element, f"i_{side}_ka"])
                value = base_value * _r(ri)
                std_dev = max(0.001, abs(ri * base_value))

        if value is not None:
            net.measurement.at[idx, "value"] = value
            net.measurement.at[idx, "std_dev"] = std_dev

    # ToDo: remove after testing
    # rng = np.random.default_rng(42)
    # # copy clean measurement set
    # meas = net.measurement.copy()
    #
    # # choose 5 % as bad data
    # n_bad = max(1, int(0.05 * len(meas)))
    # bad_idx = rng.choice(meas.index, size=n_bad, replace=False)
    #
    # # create gross errors
    # meas.loc[bad_idx, "value"] *= 1.5
    # net.measurement = meas


def _create_simbench_mc_case(
        net,
        seed_pf: int | None = None,
        scaling_ranges: dict[str, tuple[float, float]] | None = None
) -> dict[str, dict]:
    """
    Generate a random operating point for a SimBench network and perform a power flow calculation.

    A deep copy of the input network is created. Loads, static generators (``sgen``), and generators (``gen``) are
    scaled individually according to their cluster specified in the ``type`` column.

    For each element, an independent scaling factor is sampled from a uniform distribution. The lower and upper bounds
    of the distribution are obtained from ``scaling_ranges`` using the element's ``type`` as the dictionary key.

    All clusters occurring in the ``load``, ``sgen``, and ``gen`` tables must have a corresponding entry in
    ``scaling_ranges``. Additional entries in ``scaling_ranges`` that are not used by the network are allowed.

    Active power ``p_mw`` is multiplied by the sampled scaling factor. Reactive power ``q_mvar`` is scaled by the same
    factor if the respective element table contains a ``q_mvar`` column. Thus, the original power factor of the element
    is preserved.

    After scaling all elements, a power flow is performed on the copied network. The resulting ``res_*`` tables are then
    copied back to the original network. Consequently, the original network retains its nominal load and generation
    values while its result tables represent the randomly generated operating point.

    Parameters:
        net:
            network containing the nominal load and generation values. The element tables ``load``, ``sgen``, and
            ``gen`` are expected to contain a ``type`` column defining the corresponding cluster.
        seed_pf: Random seed used for sampling the scaling factors. If ``None``, the current NumPy random state is used.
        scaling_ranges :
            Mapping from cluster names to the lower and upper bounds of their uniformly distributed scaling factors,
            i.e. ``{"cluster": (lower_bound, upper_bound)}``. Every cluster occurring in ``net.load``, ``net.sgen``, or
            ``net.gen`` must be present in this dictionary. If ``None``, the predefined default scaling ranges are used.

    Returns:
        Dictionary containing the sampled scaling factor for each element, grouped by element type
        (``load``, ``sgen``, and ``gen``). The element indices are used as keys.
    """

    # ToDo: Check what the scaling factor in load, sgen, gen does
    if seed_pf is not None:
        np.random.seed(seed_pf)

    if scaling_ranges is None:
        scaling_ranges = {
            "Biomass_MV": (0.8, 1.0),
            "Hydro_MV": (0.6, 1.0),
            "PV_MV": (0.2, 1.0),
            "Wind_MV": (0.3, 1.0),
            "commercial": (0.5, 0.9),
            "lv_RES": (0.3, 0.8),
            "residential": (0.3, 0.9),
        }

    for cluster, (low, high) in scaling_ranges.items():
        if low > high:
            raise ValueError(f"Invalid scaling range for '{cluster}': lower bound {low} > upper bound {high}")

    net_pf = copy.deepcopy(net)

    k = {
        "load": {},
        "sgen": {},
        "gen": {}
    }

    # check that all used clusters have a scaling range
    used_clusters = set()

    for element in ("load", "sgen", "gen"):
        table = net_pf[element]
        if table.empty:
            continue
        if "type" not in table.columns:
            raise ValueError(f"Element table '{element}' has no 'type' column.")
        if table["type"].isna().any():
            missing_type_indices = table.index[table["type"].isna()].tolist()
            raise ValueError(f"Missing cluster/type for {element} indices: {missing_type_indices}")
        used_clusters.update(table["type"].unique())

    missing = used_clusters - set(scaling_ranges)
    if missing:
        raise ValueError(
            f"Missing scaling ranges for clusters: {sorted(missing)}"
        )

    # scale elements depending on their cluster/type
    for element in ("load", "sgen", "gen"):
        table = net_pf[element]

        if table.empty:
            continue

        for idx in table.index:
            cluster = table.at[idx, "type"]

            low, high = scaling_ranges[cluster]
            factor = np.random.uniform(low, high)

            k[element][idx] = factor

            table.at[idx, "p_mw"] *= factor

            if "q_mvar" in table.columns:
                table.at[idx, "q_mvar"] *= factor

    # run power flow with scaled operating point
    runpp(net_pf)

    # copy results from powerflow to net for state estimation
    for key in net_pf.keys():
        if key.startswith("res_"):
            net[key] = net_pf[key].copy(deep=True)
    return k


def _create_measurement_18_bus_grid(
        net: pandapowerNet,
        rv: float = .01,
        rp: float = .03,
        rq: float = .03
) -> None:
    """
    Add measurements to the 18 bus test gird for state estimation.

    This function adds a predefined set of measurement points at fixed locations. The measurement values are derived
    from the power flow results of the grid and are perturbed with Gaussian noise, whose magnitude is controlled by the
    given standard deviations.

    Parameters:
        net: power grid
        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage
        rp: standard deviation to apply a multiplicative perturbation to quantities for active power
        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power
    """
    # if seed is not None:
    #    np.random.seed(seed)  # Set deterministic random numbers for reproducible simulations
    # =========================================================================
    # Bus 1: v, p, q (index: 0)
    # =========================================================================
    create_measurement(
        net,
        meas_type="v",
        element_type="bus",
        value=net.res_bus.vm_pu[0] * _r(rv),
        std_dev=max(.001, abs(rv * net.res_bus.vm_pu[0])),
        element=0
    )
    create_measurement(
        net,
        meas_type="p",
        element_type="bus",
        value=net.res_bus.p_mw[0] * _r(rp),
        std_dev=max(.001, abs(rp * net.res_bus.p_mw[0])),
        element=0)
    create_measurement(
        net,
        meas_type="q",
        element_type="bus",
        value=net.res_bus.q_mvar[0] * _r(rq),
        std_dev=max(.001, abs(rq * net.res_bus.q_mvar[0])),
        element=0
    )
    # =========================================================================
    # Bus 4: voltage measurement (index: 3)
    # =========================================================================
    create_measurement(
        net,
        meas_type="v",
        element_type="bus",
        value=net.res_bus.vm_pu[3] * _r(rv),
        std_dev=max(.001, abs(rv * net.res_bus.vm_pu[3])),
        element=3
    )
    # =========================================================================
    # Line 4 -> 5 (Bus): active/reactive power flow measurement
    # Line index = 3 ( 3 -> 4 Busindex)
    # =========================================================================
    create_measurement(
        net,
        meas_type="p",
        element_type="line",
        value=net.res_line.p_from_mw[3] * _r(rp),
        std_dev=max(.001, abs(rp * net.res_line.p_from_mw[3])),
        element=3,
        side="from"
    )
    create_measurement(
        net,
        meas_type="q",
        element_type="line",
        value=net.res_line.q_from_mvar[3] * _r(rq),
        std_dev=max(.001, abs(rq * net.res_line.q_from_mvar[3])),
        element=3,
        side="from"
    )
    # =========================================================================
    # Line 4 -> 15 (Bus): active/reactive power flow measurement
    # Line index = 13 (3 -> 14 Busindex)
    # =========================================================================
    create_measurement(
        net,
        meas_type="p",
        element_type="line",
        value=net.res_line.p_from_mw[13] * _r(rp),
        std_dev=max(.001, abs(rp * net.res_line.p_from_mw[13])),
        element=13,
        side="from"
    )
    create_measurement(
        net,
        meas_type="q",
        element_type="line",
        value=net.res_line.q_from_mvar[13] * _r(rq),
        std_dev=max(.001, abs(rq * net.res_line.q_from_mvar[13])),
        element=13,
        side="from"
    )
    # =========================================================================
    # Bus 10: voltage measurement (index: 9)
    # =========================================================================
    create_measurement(
        net,
        meas_type="v",
        element_type="bus",
        value=net.res_bus.vm_pu[9] * _r(rv),
        std_dev=max(.001, abs(rv * net.res_bus.vm_pu[9])),
        element=9
    )
    # =========================================================================
    # Line 10 -> 11 (Bus): active/reactive power flow measurement
    # Line index = 9 (9 -> 10 Busindex)
    # =========================================================================
    create_measurement(
        net,
        meas_type="p",
        element_type="line",
        value=net.res_line.p_from_mw[9] * _r(rp),
        std_dev=max(.001, abs(rp * net.res_line.p_from_mw[9])),
        element=9,
        side="from"
    )
    create_measurement(
        net,
        meas_type="q",
        element_type="line",
        value=net.res_line.q_from_mvar[9] * _r(rq),
        std_dev=max(.001, abs(rq * net.res_line.q_from_mvar[9])),
        element=9,
        side="from"
    )


def _create_18_bus_grid(
        base_mva: float = 10.0,
        v_b: float = 11.0,
        slack_v: float = 1.0,
        slack_va_degree: float = 0.0,
        load_range: tuple[float, float] = (.5, .8),
        com_range: tuple[float, float] = (.3, .6),
        pv_range: tuple[float, float] = (.3, .4),
        wind_range: tuple[float, float] = (.2, .4)
) -> tuple[pandapowerNet, dict[str, np.ndarray]]:
    """
    Create the 18-bus radial distribution network from the original MATLAB implementation and run a power flow
    calculation using pandapower.

    The network is modeled as an 11 kV radial distribution grid with residential loads, commercial loads, photovoltaic
    (PV) generation, and wind generation connected to different buses.

    The original MATLAB implementation uses per-unit (p.u.) values. Since pandapower expects physical units, all line
    impedance and power values are converted to engineering units before creating the network elements.

    A random operating point is generated for each simulation by scaling residential loads, commercial loads,
    PV generation and wind generation with uniformly distributed random factors.

    For the **power flow calculation**, these randomly scaled (statistically perturbed) values are used.
    For the **state estimation network**, the corresponding **nominal** (unscaled) load and generation values are
    stored, while the power flow results (voltages, line flows, etc.) from the randomly scaled case ar copied into the
    result tables of the state estimation network.

    Parameters:
        base_mva: Base apparent power of the system in MVA. Corresponds to ``Sb`` in the MATLAB implementation.
        v_b: in kV. Base voltage of the distribution grid
        slack_v: Voltage magnitude of the slack bus in per-unit.
        slack_va_degree: Voltage angle of the slack bus in degrees.
        load_range: Range of scaling factor (uniformly distributed) for load (residential load)
        com_range: Range of scaling factor (uniformly distributed) for load (commercial load)
        pv_range: Range of scaling factor (uniformly distributed) for sgne (PV generation)
        wind_range: Range of scaling factor (uniformly distributed) for wind (wind generation)

    Returns:
        Return a pandapower net with the results from the powerflow calculation and the nominal loads and powers for
        state estimation. The dict includes the random scaling factors:

            - **net**: The pandapower network including buses, lines, loads, generators, and power flow results.
            - **K**: Dictionary containing the random scaling coefficients:

                - ``KL_res``: residential load scaling
                - ``KL_com``: commercial load scaling
                - ``KG_pv``: PV generation scaling
                - ``KG_wind``: wind generation scaling
    """

    # if seed is not None:
    #     np.random.seed(seed)  # Set deterministic random numbers for reproducible simulations

    # =========================================================================
    # Base values
    # =========================================================================
    # The original MATLAB implementation uses per-unit (p.u.) values.
    #
    # pandapower expects physical units:
    #   - voltage in kV
    #   - power in MW / MVAR
    #   - impedance in Ohm
    #
    # Therefore, the per-unit values must be converted.

    # Base impedance:
    #
    #     Z_base = V_base² / S_base
    #
    # With:
    #     V_base in kV
    #     S_base in MVA
    #
    # This gives:
    #     Z_base = 11² / 10 = 12.1 Ohm
    #
    # Used to convert line impedance:
    #
    #     Z_ohm = Z_pu * Z_base
    z_base = (v_b ** 2) / base_mva  # Ohm

    # =========================================================================
    # Create empty power network for power flow calculation
    # =========================================================================
    net_pf: pandapowerNet = create_empty_network(sn_mva=base_mva, f_hz=50)

    # =========================================================================
    # 1) Buses
    # =========================================================================
    # Create 18 buses corresponding to the original MATLAB network.
    #
    # Bus 1:
    #     Slack / reference bus
    #
    # Bus 2-18:
    #     PQ buses with loads and distributed generation
    #
    buses = []
    for i in range(1, 19):
        bus = create_bus(net_pf, vn_kv=v_b, name=f"Bus {i}")
        buses.append(bus)
    # Create slack bus / external grid connection
    # vm_pu: voltage magnitude in per-unit
    # degree: voltage angle in degrees
    #
    create_ext_grid(net_pf, bus=buses[0], vm_pu=slack_v, degree=slack_va_degree, name="Slack")

    # =========================================================================
    # 2) Lines
    # =========================================================================
    start = np.array([1, 2, 3, 4, 5, 6, 6, 8, 9, 10, 11, 11, 13, 4, 15, 16, 16])
    end = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    # Per-unit line resistances
    r_pu = np.array([0.00001, 0.0174, 0.0001, 0.0052, 0.0003, 0.0010, 0.0017, 0.0022, 0.0001, 0.0016, 0.0007, 0.0299,
                     0.0010, 0.0025, 0.0041, 0.0034, 0.0013])  # change zeros to small value for pf-calc
    # Per-unit line reactances
    x_pu = np.array([0.1000, 0.0085, 0.0001, 0.0028, 0.0002, 0.0010, 0.0008, 0.0011, 0.00002, 0.0008, 0.0003, 0.0081,
                     0.0010, 0.0007, 0.0013, 0.0009, 0.0004])  # change zeros to small value for pf-calc
    # Create lines
    #
    # The MATLAB data provides total line impedance in per-unit.
    #
    # pandapower requires:
    #     r_ohm_per_km
    #     x_ohm_per_km
    #
    # Since no physical line lengths are available,
    # each line is modeled with:
    #
    #     length_km = 1.0
    #
    # Therefore:
    #
    #     impedance_per_km == total_impedance
    #
    for idx, (s, e, r, x) in enumerate(zip(start, end, r_pu, x_pu)):
        create_line_from_parameters(
            net_pf,
            from_bus=buses[s - 1],
            to_bus=buses[e - 1],
            length_km=1.0,
            r_ohm_per_km=r * z_base,
            x_ohm_per_km=x * z_base,
            c_nf_per_km=0.0,
            max_i_ka=1.0,
            name=f"Line {idx + 1}"
        )
    net_se = copy.deepcopy(net_pf)  # create second net for state estimation
    # =========================================================================
    # 3) Loads and distributed generation
    # =========================================================================
    # The original MATLAB model contains:
    #
    #   - residential loads
    #   - commercial loads
    #   - photovoltaic generation
    #   - wind generation
    #
    # All values are defined in per-unit on the system base power.
    #
    # Positive values:
    #     generation
    #
    # Negative values in MATLAB:
    #     loads
    #
    # In pandapower:
    #     loads are modeled as positive consumption

    # Nominal values these values are used for state estimation and adapted for powerflow calculation
    p_l_res = 2 * np.array([0.05, 0.08, 0, 0.05, 0, 0.06, 0, 0.02, 0.04, 0, 0.09, 0, 0.08, 0, 0, 0.05, 0.07])
    q_l_res = 2 * np.array([0.01, 0.02, 0, 0.01, 0, 0.01, 0, 0.01, 0.01, 0, 0.01, 0, 0.01, 0, 0, 0.01, 0.01])

    p_l_com = np.array([0.03, 0.08, 0, 0.05, 0, 0.05, 0, 0.07, 0.03, 0, 0.01, 0, 0.03, 0, 0, 0.01, 0.02])
    q_l_com = np.array([0.02, 0.02, 0, 0.01, 0, 0.01, 0, 0.01, 0.01, 0, 0.01, 0, 0.01, 0, 0, 0.01, 0.01])

    p_g_pv = np.array([0.04, 0.05, 0, 0.02, 0, 0.08, 0, 0.05, 0.03, 0, 0.04, 0, 0.05, 0, 0, 0.07, 0.02])
    q_g_pv = -np.array([0.00, 0.00, 0, 0.00, 0, 0.01, 0, 0.00, 0.00, 0, 0.00, 0, 0.01, 0, 0, 0.00, 0.00])

    p_g_wind = np.array([0.00, 0.00, 0, 0.07, 0, 0.00, 0, 0.00, 0.03, 0, 0.00, 0, 0.08, 0, 0, 0.04, 0.04])
    q_g_wind = -np.array([0.00, 0.00, 0, 0.00, 0, 0.00, 0, 0.00, 0.01, 0, 0.00, 0, 0.01, 0, 0, 0.00, 0.00])

    # Dictionary storing the random scaling factors
    k_dc = {
        "KL_res": np.zeros(17),
        "KL_com": np.zeros(17),
        "KG_pv": np.zeros(17),
        "KG_wind": np.zeros(17)
    }
    # =========================================================================
    # Create loads and generators at each bus
    # =========================================================================
    for idx in range(17):
        bus = buses[idx + 1]  # buses 2-18
        # Random operating-point scaling factors
        #
        # Residential load:
        #     50% - 80%
        # Commercial load:
        #     30% - 60%
        # PV:
        #     30% - 40%
        # Wind:
        #     20% - 40%
        #
        var_l_res = np.random.uniform(*load_range)  # ToDo: create other scaling values to get better behavior
        var_l_com = np.random.uniform(*com_range)
        var_g_pv = np.random.uniform(*pv_range)
        var_g_wind = np.random.uniform(*wind_range)

        k_dc["KL_res"][idx] = var_l_res
        k_dc["KL_com"][idx] = var_l_com
        k_dc["KG_pv"][idx] = var_g_pv
        k_dc["KG_wind"][idx] = var_g_wind

        # =====================================================================
        # Apply scaling factors to nominal per-unit values for powerflow
        # =====================================================================
        p_res_pu = float(p_l_res[idx] * var_l_res)
        q_res_pu = float(q_l_res[idx] * var_l_res)

        p_com_pu = float(p_l_com[idx] * var_l_com)
        q_com_pu = float(q_l_com[idx] * var_l_com)

        p_pv_pu = float(p_g_pv[idx] * var_g_pv)
        q_pv_pu = float(q_g_pv[idx] * var_g_pv)

        p_wind_pu = float(p_g_wind[idx] * var_g_wind)
        q_wind_pu = float(q_g_wind[idx] * var_g_wind)

        # =====================================================================
        # Convert per-unit values to MW / MVAR
        # =====================================================================
        #
        # Conversion:
        #
        #     P_MW = P_pu * S_base
        #
        #     Q_MVAR = Q_pu * S_base
        #
        # Loads are created as separate elements:
        #     - residential
        #     - commercial
        #
        # Generators are created as:
        #     - pv
        #     - wind
        # For state estimation the nominal values are used

        # Residential load
        if p_res_pu != 0 or q_res_pu != 0:
            create_load(
                net_pf,
                bus=bus,
                p_mw=p_res_pu * base_mva,
                q_mvar=q_res_pu * base_mva,
                name=f"Residential Load Bus {idx + 2}",
                type="residential"
            )
            create_load(
                net_se,
                bus=bus,
                p_mw=p_l_res[idx] * base_mva,  # nominal value
                q_mvar=q_l_res[idx] * base_mva,  # nominal value
                name=f"Residential Load Bus {idx + 2}",
                type="residential"
            )
        # Commercial load
        if p_com_pu != 0 or q_com_pu != 0:
            create_load(
                net_pf,
                bus=bus,
                p_mw=p_com_pu * base_mva,
                q_mvar=q_com_pu * base_mva,
                name=f"Commercial Load Bus {idx + 2}",
                type="commercial"
            )
            create_load(
                net_se,
                bus=bus,
                p_mw=p_l_com[idx] * base_mva,  # nominal value
                q_mvar=q_l_com[idx] * base_mva,  # nominal value
                name=f"Commercial Load Bus {idx + 2}",
                type="commercial"
            )
        # Photovoltaic generation
        if p_pv_pu != 0 or q_pv_pu != 0:
            create_sgen(
                net_pf,
                bus=bus,
                p_mw=p_pv_pu * base_mva,
                q_mvar=q_pv_pu * base_mva,
                name=f"PV Bus {idx + 2}",
                type="pv"
            )
            create_sgen(
                net_se,
                bus=bus,
                p_mw=p_g_pv[idx] * base_mva,  # nominal value
                q_mvar=q_g_pv[idx] * base_mva,  # nominal value
                name=f"PV Bus {idx + 2}",
                type="pv"
            )
        # Wind generation
        if p_wind_pu != 0 or q_wind_pu != 0:
            create_sgen(
                net_pf,
                bus=bus,
                p_mw=p_wind_pu * base_mva,
                q_mvar=q_wind_pu * base_mva,
                name=f"Wind Bus {idx + 2}",
                type="wind"
            )
            create_sgen(
                net_se,
                bus=bus,
                p_mw=p_g_wind[idx] * base_mva,  # nominal value
                q_mvar=q_g_wind[idx] * base_mva,  # nominal value
                name=f"Wind Bus {idx + 2}",
                type="wind"
            )

    # =========================================================================
    # Run power flow calculation
    # =========================================================================
    # after run powerflow the results will save in the net tables for state estimation
    # ToDo: for future general use net_pf and net_se should separately return
    runpp(net_pf)
    for key in net_pf.keys():
        if key.startswith("res_"):
            net_se[key] = net_pf[key].copy(deep=True)
    return net_se, k_dc


def _calc_different_se(
        net_base: pandapowerNet,
        failures: list,
        af_vc: list,
        num_it: str,
        with_ortools: bool = True,
        with_af_constraints: bool = True,
        with_wls: bool = True,
        data_path: str = ".") -> None:
    """
    un three different allocation-factor-based state estimation methods (AF-WLS, AF-WLAV, AF-LAV) for a specific power
    grid and return bus voltages, angles, deviations and the allocation factors. The power grid is unobservable.

    For each estimator:
        1. A deep copy of ``net_base`` is created.
        2. State estimation is run with the specified algorithm.
        3. The resulting pandapower network, including state estimation results, is saved as a pickle file.
        4. The estimated allocation factors are collected, and a method-specific index (AF-WLS, AF-LAV, AF-WLAV) is
           assigned.
        5. If the estimator does not converge (``success`` is False), a failure message is appended to ``failures``.

    After all three estimations have been performed, the allocation factors of all methods are concatenated into a
    single DataFrame and saved as a CSV file.

    Parameters:
        net_base: Base pandapower grid on which all three state estimation methods are applied.
        failures: List that is extended by a text entry for each estimation that fails to converge.
        af_vc: List where information about af which violate constraints.
        num_it: Identifier for this run (e.g. iteration counter) used in all output filenames.
        with_ortools: OR-Tools solver for linear solver ("lp" algorithm). False take scipy solver.
        with_af_constraints: Constraints for allocation factors between 0 and 1 for (W)LAV algorithm.
        with_wls: WLS include by true
        data_path: Directory where the PICKLE and CSV result files are stored.

    Returns:
        None
    """
    if with_wls:
        af_wls = None
    af_lav = None
    af_wlav = None

    if with_wls:
        # run AF-WLS estimation
        af_wls_path = os.path.join(data_path, "af_wls")
        os.makedirs(af_wls_path, exist_ok=True)
        af_wls_file = os.path.join(af_wls_path, f"af_wls_{num_it}.p")
        if os.path.exists(af_wls_file):
            print(f"file {af_wls_file} already exists")
        else:
            try:
                net_af_wls = copy.deepcopy(net_base)
                af_wls = estimate(net_af_wls, algorithm="af-wls",maximum_iterations=100)
                # ToDo: add TypeDict for state estimation
                af_wls["allocation_factors"].index = ["AF-WLS"]  # type: ignore[attr-defined]
                if not af_wls["success"]:
                    failures.append(f"AF-WLS, {num_it}, se failed")  # add failures information to a list
                to_pickle(net_af_wls, af_wls_file)  # save grid to pickle
            except Exception as e:
                failures.append(f"AF-WLS, {num_it}, exception {type(e).__name__}: {e}")
                print(f"AF-WLS iteration {num_it} crashed: {type(e).__name__}: {e}")

    # run LAV estimation
    af_lav_path = os.path.join(data_path, "af_lav")
    os.makedirs(af_lav_path, exist_ok=True)
    af_lav_file = os.path.join(af_lav_path, f"af_lav_{num_it}.p")
    if os.path.exists(af_lav_file):
        print(f"file {af_lav_file} already exists")
    else:
        try:
            net_af_lav = copy.deepcopy(net_base)
            af_lav = estimate(
                net_af_lav,
                algorithm="af-lp",
                wlav=False,
                with_ortools=with_ortools,
                with_af_constraints=with_af_constraints,
                linprog_method="highs-ipm",
                maximum_iterations=100
            )
            af_lav["allocation_factors"].index = ["AF-LAV"]  # set index for saving data
            if not af_lav["success"]:
                failures.append(f"AF-LAV, {num_it}, se failed")
            to_pickle(net_af_lav, af_lav_file)
            if (af_lav["allocation_factors"].loc["AF-LAV"].min() < 0
                    or af_lav["allocation_factors"].loc["AF-LAV"].max() > 1.):
                # af_lav["allocation_factors"]["sum"] = af_lav["allocation_factors"].loc["AF-LAV"].sum()
                af_vc.append(f"AF-LAV, {num_it}, allocation factors violate constraints")
                print(f"AF (AF-LAV) should be between zero and one.")

        except Exception as e:
            failures.append(f"AF-LAV, {num_it}, exception {type(e).__name__}: {e}")
            print(f"AF-LAV iteration {num_it} crashed: {type(e).__name__}: {e}")

    # run AF-WLAV estimation
    af_wlav_path = os.path.join(data_path, "af_wlav")
    os.makedirs(af_wlav_path, exist_ok=True)
    af_wlav_file = os.path.join(af_wlav_path, f"af_wlav_{num_it}.p")
    if os.path.exists(af_wlav_file):
        print(f"file {af_wlav_file} already exists")
    else:
        try:
            # if int(num_it) == 48:  # Todo: remove after testing
            #     print(f"halt stop, jetzt programmiere ich!!")
            net_af_wlav = copy.deepcopy(net_base)
            af_wlav = estimate(
                net_af_wlav,
                algorithm="af-lp",
                wlav=True,
                with_ortools=with_ortools,
                with_af_constraints=with_af_constraints,
                linprog_method="highs-ipm",
                maximum_iterations=100
            )
            af_wlav["allocation_factors"].index = ["AF-WLAV"]  # set index for saving data
            if not af_wlav["success"]:
                failures.append(f"AF-WLAV, {num_it}, se failed")
            to_pickle(net_af_wlav, af_wlav_file)
            if (af_wlav["allocation_factors"].loc["AF-WLAV"].min() < 0
                    or af_wlav["allocation_factors"].loc["AF-WLAV"].max() > 1.):
                af_vc.append(f"AF-WLAV, {num_it}, allocation factors violate constraints")
                print(f"AF (AF-WLAV) should be between zero and one.")
        except Exception as e:
            failures.append(f"AF-WLAV, {num_it}, exception {type(e).__name__}: {e}")
            print(f"AF-WLAV iteration {num_it} crashed: {type(e).__name__}: {e}")

    # save allocation factors from all estimation solvers in one csv file
    af_path = os.path.join(data_path, "af")
    os.makedirs(af_path, exist_ok=True)
    res_af_file = os.path.join(af_path, f"af_df_{num_it}.csv")
    if os.path.exists(res_af_file):
        print(f"file {res_af_file} already exists")
    else:
        col = None
        # If one solver return allocation factors (af) their names will read out
        if with_wls:
            af_ls = [af_wls, af_wlav, af_lav]
        else:
            af_ls = [af_wlav, af_lav]
        for af in af_ls:
            if af is not None:
                col = af["allocation_factors"].columns
                break
        # If all allocation factors from the solvers are None no file will save.
        if col is None:
            print(f"No allocation factors available for iteration {num_it}")
            return
        # The solver which have None values get an DataFrame with None
        if with_wls:
            if af_wls is None:
                af_wls_df = pd.DataFrame([[None] * len(col)], columns=col, index=["AF-WLS"])
            else:
                af_wls_df = af_wls["allocation_factors"]
        if af_lav is None:
            af_lav_df = pd.DataFrame([[None] * len(col)], columns=col, index=["AF-LAV"])
        else:
            af_lav_df = af_lav["allocation_factors"]
        if af_wlav is None:
            af_wlav_df = pd.DataFrame([[None] * len(col)], columns=col, index=["AF-WLAV"])
        else:
            af_wlav_df = af_wlav["allocation_factors"]
        # all dataframes with af form all solvers will concat to one
        if with_wls:
            res_af_df = pd.concat([af_wls_df, af_wlav_df, af_lav_df])
        else:
            res_af_df = pd.concat([af_wlav_df, af_lav_df])
        res_af_df.to_csv(res_af_file, index=True)


def create_random_18_bus_grid_random_estimation(
        data_path: str = ".",
        itr: int = 1000,
        seed: int = 112,
        with_ortools: bool = True,
        with_wls: bool = True,
        rv: float = .01,
        rp: float = .03,
        rq: float = .03,
        load_range: tuple[float, float] = (.5, .8),
        com_range: tuple[float, float] = (.3, .6),
        pv_range: tuple[float, float] = (.3, .4),
        wind_range: tuple[float, float] = (.2, .4)
) -> None:
    """
    Every iteration applies random perturbations to the load and generation values used in the power flow calculation
    and, in addition, to the measurement values used for state estimation.

    Parameters:
        data_path: Place where grid and allocation factors will be saved.
        itr: Number of iterations.
        seed:
            Optional random seed for reproducible simulations. If ``None``, random values are generated for every call.
        with_ortools: OR-Tools solver for linear solver ("lp" algorithm). False take scipy solver.
        with_wls: for :func:`_calc_different_se` if true wls solver will be used.
        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage
        rp: standard deviation to apply a multiplicative perturbation to quantities for active power
        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power
        load_range:
            Range of scaling factor (uniformly distributed) for load (residential load) :func:`_create_18_bus_grid`
        com_range:
            Range of scaling factor (uniformly distributed) for load (commercial load) :func:`_create_18_bus_grid`
        pv_range:
            Range of scaling factor (uniformly distributed) for sgne (PV generation) :func:`_create_18_bus_grid`
        wind_range:
            Range of scaling factor (uniformly distributed) for wind (wind generation) :func:`_create_18_bus_grid`
    Returns: None
    """

    np.random.seed(seed)
    failures: list = []  # The list contains information about the final status of the state estimation
    violation_constraints: list = []
    for i in tqdm(range(itr)):
        name_str = f"{i:03d}"  # number 1 -> 001, 56 -> 056 etc.
        net18, k = _create_18_bus_grid(
            load_range=load_range, com_range=com_range, pv_range=pv_range, wind_range=wind_range
        )
        _create_measurement_18_bus_grid(net=net18, rv=rv, rp=rp, rq=rq)  #
        _calc_different_se(
            net18, failures, violation_constraints, name_str, with_ortools, True, with_wls, data_path
        )

    # If failures and negative af are not empty, the list will save as txt file.
    if not failures:
        print("failures is empty, very good")
    else:
        with open(os.path.join(data_path, "failures.txt"), "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(failure + "\n")
        print(f"List is not empty: {failures}")

    if not violation_constraints:
        print("violation_constraints is empty, also good")
    else:
        with open(os.path.join(data_path, "violation_constraints.txt"), "w", encoding="utf-8") as f:
            for n_vc in violation_constraints:
                f.write(n_vc + "\n")
        print(f"List is not empty: {violation_constraints}")


def create_random_estimations_simbench(
        net: pandapowerNet,
        path: str = ".",
        itr: int = 1000,
        seed: int = 112,
        seed_pf: int | None = None,
        seed_m: int | None = None,
        with_ortools: bool = True,
        with_af_constraints: bool = True,
        with_wls: bool = True,
        rv: float = .01,
        ri: float = .01,
        rp: float = .03,
        rq: float = .03,
        scaling_ranges: dict[str, tuple[float, float]] | None = None
) -> None:
    """
    Every iteration applies random perturbations to the load and generation values used in the power flow calculation
    and, in addition, to the measurement values used for state estimation.

    Parameters:
        net: Power grid with powerflow calculation results and without measurements.
        path: Place where grid and allocation factors will be saved.
        itr: Number of iterations.
        seed:
            Optional random seed for reproducible simulations. If ``None``, random values are generated for every call.
        seed_pf:
            Parameter for :func:`_create_simbench_mc_case`. Attention if None, no seed will be used different to normal use
            case.
        seed_m:
            Parameter for :func:`_add_measurements_af`. Attention if None, no seed will be used different to normal use
            case.
        with_ortools: OR-Tools solver for linear solver ("lp" algorithm). False take scipy solver.
        with_af_constraints: Constraints for allocation factors between 0 and 1 for (W)LAV algorithm.
        with_wls: for :func:`_calc_different_se` if true wls solver will be used.
        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage
        ri: standard deviation to apply a multiplicative perturbation to quantities for current
        rp: standard deviation to apply a multiplicative perturbation to quantities for active power
        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power
        scaling_ranges: scaling factor range for sgen and load
    Returns: None
    """
    if scaling_ranges is None:
        scaling_ranges = {
            "Biomass_MV": (0.8, 1.0),
            "Hydro_MV": (0.6, 1.0),
            "PV_MV": (0.2, 1.0),
            "Wind_MV": (0.3, 1.0),
            "commercial": (0.5, 0.9),
            "lv_RES": (0.3, 0.8),
            "residential": (0.3, 0.9),
        }
    np.random.seed(seed)
    failures: list = []  # The list contains information about the final status of the state estimation
    violation_constraints: list = []
    for i in range(itr):
        name_str = f"{i:03d}"  # number 1 -> 001, 56 -> 056 etc.
        k = _create_simbench_mc_case(net, seed_pf, scaling_ranges)
        _fill_measurement_values_from_powerflow(net, seed_m, rv, ri, rp, rq)
        _calc_different_se(
            net, failures, violation_constraints, name_str, with_ortools, with_af_constraints, with_wls, path
        )

    if not failures:
        print("List is empty, very good")
    else:
        with open(os.path.join(path, "failures.txt"), "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(failure + "\n")
        print(f"List is not empty: {failures}")
    if not violation_constraints:
        print("violation_constraints is empty, also good")
    else:
        with open(os.path.join(path, "violation_constraints.txt"), "w", encoding="utf-8") as f:
            for n_vc in violation_constraints:
                f.write(n_vc + "\n")
        print(f"List is not empty: {violation_constraints}")


def load_failures(data_path: str = ".", eval_path: str = ".") -> set[tuple[str, int]]:
    """
    Load previously recorded failed solver runs from a text file and return them as a set of ``(solver, iteration)``
    tuples.

    The function reads the file ``failures.txt`` located in ``path``. Each row is expected to contain a solver name,
    an iteration number, and a status field. The status column is ignored when constructing the return value.

    Parameters:
        data_path: Directory containing the ``failures.txt`` file. Defaults to the current working directory (``"."``).
        eval_path: Directory where the new csv will save

    Returns:
        Set of unique ``(solver, iteration)`` pairs representing failed solver runs.
    """
    failures = pd.read_csv(
        os.path.join(data_path, "failures.txt"),
        header=None,
        names=["solver", "iteration", "status"],
        skipinitialspace=True,
        dtype={"solver": str, "iteration": str, "status": str}
    )
    failures["iteration"] = failures["iteration"].astype(int)
    failures_csv = os.path.join(eval_path, "failures.csv")
    if os.path.exists(failures_csv):
        print(f"file {failures_csv} exists, ignoring")
    else:
        failures.to_csv(failures_csv, index=False)
    failure_set = set(zip(failures["solver"], failures["iteration"]))
    return failure_set


def evaluation_af(data_path: str = ".", eval_path: str = ".") -> None:
    """
    Evaluate and visualize the distribution of allocation factors across all simulation runs.

    This function aggregates allocation factor results from multiple CSV files (``af_df_*.csv``), excludes failed state
    estimation runs listed in ``failures.txt``, and creates an HTML file containing boxplots and histograms for each
    allocation factor and solver. Creates the file ``allocation_factor_plots.html`` in ``path_eval`` (subdirectory which
    will created if it does not exist), containing for each solver and allocation factor:
        * a boxplot of the allocation factor over all valid iterations, and
        * a histogram of the corresponding value distribution.

    Parameters:
        data_path:
            Expected files:
                * ``failures.txt``: text file with three columns (solver, iteration, status), used to identify and skip
                    failed runs.
                * ``af_df_*.csv``: CSV files containing allocation factor results for each iteration. Each file must
                    have solvers as index (e.g. ``AF-WLS``, ``AF-LAV``, ``AF-WLAV``) and allocation factors as columns.
        eval_path:
            Directory where the evaluation files are stored. Defaults to the current/working directory.

    Raises:
        FileNotFoundError: If no CSV files matching ``af_df_*.csv`` are found in the given directory.

    Returns: None
    """
    save_path = os.path.join(eval_path, "statistical")
    os.makedirs(save_path, exist_ok=True)
    html_file = os.path.join(save_path, "allocation_factor_plots.html")  # check if the file exists
    if os.path.exists(html_file):
        print(f"html file already exists: {html_file}")
        return
    # -------------------------------------------------------------------------
    # Read in failures
    # -------------------------------------------------------------------------
    failure_set = load_failures(data_path, eval_path)
    # -------------------------------------------------------------------------
    # Read in CSV file with allocation factors
    # -------------------------------------------------------------------------
    af_path = os.path.join(data_path, "af")
    csv_files = sorted(
        os.path.join(af_path, f) for f in os.listdir(af_path) if f.startswith("af_df_") and f.endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError("No af_df_*.csv files found")
    # load general data about solver and allocation factors
    solver_ls = pd.read_csv(csv_files[0], index_col=0).index.tolist()
    af_ls = pd.read_csv(csv_files[0], index_col=0).columns.tolist()
    # -------------------------------------------------------------------------
    # collect data
    # -------------------------------------------------------------------------
    af_total_dc = {solver: pd.DataFrame(columns=af_ls) for solver in solver_ls}
    for i in tqdm(range(len(csv_files))):
        if not os.path.exists(csv_files[i]):
            print(f"Missing: {csv_files[i]}")
            continue
        for solver in solver_ls:
            if (solver, i) in failure_set:  # only data are added, where the state estimation runs successfully
                continue
            df = pd.read_csv(csv_files[i], index_col=0)
            af_total_dc[solver].loc[i, af_ls] = df.loc[solver]
    # -------------------------------------------------------------------------
    # create plot
    # -------------------------------------------------------------------------
    rows_box = len(solver_ls)
    rows_hist = len(solver_ls)
    total_rows = rows_box + rows_hist

    # -------------------------------------------------------------------------
    # Subplot-Titel inklusive Mittelwert und Standardabweichung
    # -------------------------------------------------------------------------
    subplot_titles = []

    # Titel für Boxplots
    for solver in solver_ls:
        df_solver = af_total_dc[solver]
        for af in af_ls:
            values = pd.to_numeric(df_solver[af], errors="coerce").dropna()
            mean = values.mean()
            std = values.std()  # Stichproben-Standardabweichung

            subplot_titles.append(f"Boxplot<br>{solver}<br>{af}<br>μ = {mean:.4f}, σ = {std:.4f}")

    # Titel für Histogramme
    for solver in solver_ls:
        df_solver = af_total_dc[solver]

        for af in af_ls:
            values = pd.to_numeric(df_solver[af], errors="coerce").dropna()
            mean = values.mean()
            std = values.std()

            subplot_titles.append(f"Histogram<br>{solver}<br>{af}<br>μ = {mean:.4f}, σ = {std:.4f}")

    fig = make_subplots(rows=total_rows, cols=len(af_ls), subplot_titles=subplot_titles, vertical_spacing=0.07)
    # -------------------------------------------------------------------------
    # Boxplots
    # -------------------------------------------------------------------------
    for row, solver in enumerate(solver_ls, start=1):
        df_solver = af_total_dc[solver]

        for col, af in enumerate(af_ls, start=1):
            values = pd.to_numeric(df_solver[af], errors="coerce").dropna()

            fig.add_trace(
                go.Box(
                    y=values,
                    name=f"{solver}-{af}",
                    boxmean=True,
                    showlegend=False
                ),
                row=row,
                col=col
            )

    # -------------------------------------------------------------------------
    # Histogram in same HTML file like boxplot
    # -------------------------------------------------------------------------
    for row_offset, solver in enumerate(solver_ls, start=1):
        df_solver = af_total_dc[solver]
        row = rows_box + row_offset
        for col, af in enumerate(af_ls, start=1):
            values = pd.to_numeric(df_solver[af], errors="coerce").dropna()

            fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=30,  # number of bars -> value range
                    name=f"{solver}-{af}",
                    showlegend=False
                ),
                row=row,
                col=col
            )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    fig.update_layout(
        title="Allocation Factors – Boxplots and Histograms",
        height=max(2500, total_rows * 450),
        width=1600,
        margin=dict(t=200)
    )
    fig.write_html(html_file)
    print(f"saved html to: {html_file}")


def build_eval_dataframe(result_dict, variable="vm_pu") -> pd.DataFrame:
    """
    Construct an evaluation DataFrame from simulation result tables.

    The function extracts the specified variable from each result DataFrame in ``result_dict`` and combines the values
    into a single DataFrame. Each row corresponds to one simulation iteration, while each column corresponds to a
    network element (e.g. bus or line).

    Parameters:
        result_dict: Dictionary mapping iteration numbers to pandas DataFrames containing simulation results.
        variable: Name of the result column to extract from each DataFrame. Defaults to ``"vm_pu"``.

    Returns:
        DataFrame containing the selected variable for all iterations. Rows represent iterations and columns represent
        network elements. The index is sorted in ascending order.
    """
    return pd.DataFrame({iteration: df[variable].astype(float) for iteration, df in result_dict.items()}).T.sort_index()


def collect_pickle_files(folder: str, prefix: str) -> dict[int, str]:
    """
    Collect pickle files and map iteration numbers to file paths.

    Parameters:
        folder: Directory containing the pickle files.
        prefix: Filename prefix, e.g. "af_wls_".

    Returns: Dictionary mapping iteration numbers to the corresponding pickle file paths.
    """
    files: dict[int, str] = {}

    for filename in os.listdir(folder):
        if filename.startswith(prefix) and filename.endswith(".p"):
            iteration = int(
                filename.removeprefix(prefix).removesuffix(".p")
            )
            files[iteration] = os.path.join(folder, filename)

    return files


def evaluation_vp(data_path: str = ".", eval_path: str = ".", k: float = 3.0, with_wls: bool = True) -> None:
    """
    Evaluate voltage magnitude and branch active power estimation results for all state estimation methods and generate
    interactive HTML visualizations.

    This function loads the result networks of the three allocation-factor-based state estimation methods (AF-WLS,
    AF-WLAV, AF-LAV), excludes failed runs listed in ``failures.txt``, and compares estimated values against the true
    network results.

    For each estimator:
        1. Bus voltage magnitudes (``vm_pu``) and branch active powers (``p_from_mw``) are collected from all successful
            simulation runs.
        2. Mean values of the true and estimated quantities are calculated for each bus and branch.
        3. The root-mean-square error (RMSE) is determined and converted into an expanded uncertainty using a coverage
            factor of ``k = 3``.
        4. Interactive Plotly figures are created showing:
               * estimated and true voltage magnitudes,
               * estimated and true branch active powers,
               * expanded uncertainty of the voltage magnitude estimates.
        5. The figures are saved as separate HTML files.

    The following output files are created and saved to subdirectory estimation:
        * ``voltage_magnitude.html``: Mean estimated and true bus voltage magnitudes with uncertainty bands.
        * ``line_power.html``: Mean estimated and true branch active powers with uncertainty bands.
        * ``voltage_uncertainty.html``: Expanded uncertainty of the voltage magnitude estimates.

    Parameters:
        data_path:
            Directory containing the result files. Defaults to the current directory. Expected files:

                * ``failures.txt``: Text file with three columns (solver, iteration, status), used to identify and skip
                    failed state estimation runs.
                * ``af_wls_*.p``: Result networks generated with the AF-WLS estimator.
                * ``af_wlav_*.p``: Result networks generated with the AF-WLAV estimator.
                * ``af_lav_*.p``: Result networks generated with the AF-LAV estimator.

        eval_path: Where the generated HTML evaluation files will be written.
        k: coverage factor from paper (DOI: 10.1109/TIM.2024.3387498)
        with_wls: include wls solver if true.

    Raises:
        FileNotFoundError: If required PICKLE result files cannot be found.
    """
    save_path = os.path.join(eval_path, "statistical")
    os.makedirs(save_path, exist_ok=True)
    html_vm_file = os.path.join(save_path, "voltage_magnitude.html")
    html_lp_file = os.path.join(save_path, "line_power.html")
    html_ve_file = os.path.join(save_path, "voltage_uncertainty.html")
    if os.path.exists(html_vm_file) and os.path.exists(html_lp_file) and os.path.exists(html_ve_file):
        print(f"html files already exists: {html_vm_file, html_lp_file, html_ve_file}")
        return
    # -------------------------------------------------------------------------
    # Read in failures
    # -------------------------------------------------------------------------
    failure_set = load_failures(data_path, eval_path)
    # -------------------------------------------------------------------------
    # Read in bus and line data from pickle
    # -------------------------------------------------------------------------
    if with_wls:
        solver_ls = ["AF-WLS", "AF-WLAV", "AF-LAV"]
        af_wls_path = os.path.join(data_path, "af_wls")
        af_wls_files = collect_pickle_files(af_wls_path, "af_wls_")
    else:
        solver_ls = ["AF-WLAV", "AF-LAV"]

    af_wlav_path = os.path.join(data_path, "af_wlav")
    af_wlav_files = collect_pickle_files(af_wlav_path, "af_wlav_")
    af_lav_path = os.path.join(data_path, "af_lav")
    af_lav_files = collect_pickle_files(af_lav_path, "af_lav_")

    if with_wls:
        pkl_files_dc = {
            "AF-WLS": af_wls_files,
            "AF-WLAV": af_wlav_files,
            "AF-LAV": af_lav_files,
        }
    else:
        pkl_files_dc = {
            "AF-WLAV": af_wlav_files,
            "AF-LAV": af_lav_files,
        }

    res_bus_dc = {solver: {} for solver in solver_ls}
    res_bus_est_dc = {solver: {} for solver in solver_ls}

    res_line_dc = {solver: {} for solver in solver_ls}
    res_line_est_dc = {solver: {} for solver in solver_ls}

    all_iterations = sorted(set().union(*[files.keys() for files in pkl_files_dc.values()]))

    for i in tqdm(all_iterations):
        for solver in solver_ls:
            if (solver, i) in failure_set:
                print(f"skip failure: solver={solver}, iteration={i}")
                continue
            if i not in pkl_files_dc[solver]:
                print(f"missing pickle: solver={solver}, iteration={i}")
                continue

            net_ij = from_pickle(pkl_files_dc[solver][i])

            if not hasattr(net_ij, "res_bus"):
                print(f"missing res_bus: solver={solver}, iteration={i}")
                continue

            if not hasattr(net_ij, "res_bus_est"):
                print(f"missing res_bus_est: solver={solver}, iteration={i}")
                continue

            res_bus_dc[solver][i] = net_ij.res_bus.copy()
            res_bus_est_dc[solver][i] = net_ij.res_bus_est.copy()
            res_line_dc[solver][i] = net_ij.res_line.copy()
            res_line_est_dc[solver][i] = net_ij.res_line_est.copy()
        print(f"{i} finished")

    # -------------------------------------------------------------------------
    # Plot Figures separately
    # -------------------------------------------------------------------------
    fig_vm = go.Figure()
    fig_lp = go.Figure()
    fig_ve = go.Figure()

    for solver in solver_ls:
        # ---------------------------------------------------------------------
        # Voltage magnitude
        # ---------------------------------------------------------------------
        # DataFrame for bus
        vm_true = build_eval_dataframe(res_bus_dc[solver], "vm_pu").astype(float)
        vm_est = build_eval_dataframe(res_bus_est_dc[solver], "vm_pu").astype(float)

        # calc mean for every bus separate
        vm_true_mean = vm_true.mean(axis=0)
        vm_est_mean = vm_est.mean(axis=0)

        # clac RMSE pro bus
        vm_error = vm_est - vm_true
        vm_rmse = np.sqrt((vm_error ** 2).mean(axis=0))
        # clac expanded uncertainty
        vm_u = k * vm_rmse

        x_bus = np.arange(len(vm_est_mean))

        # plot uncertainty band
        fig_vm.add_trace(
            go.Scatter(
                x=np.concatenate([x_bus, x_bus[::-1]]),
                y=np.concatenate([
                    (vm_est_mean + vm_u).to_numpy(),
                    (vm_est_mean - vm_u).to_numpy()[::-1]
                ]),
                fill="toself",
                line=dict(width=0),
                opacity=0.2,
                name=f"{solver} uncertainty"
            )
        )
        # plot mean value from estimated voltage magnitude for every bus
        fig_vm.add_trace(
            go.Scatter(
                x=x_bus,
                y=vm_est_mean.to_numpy(),
                mode="lines",
                name=f"{solver} estimated V"
            )
        )
        # plot mean value from calculated voltage magnitude (powerflow) for every bus
        fig_vm.add_trace(
            go.Scatter(
                x=x_bus,
                y=vm_true_mean.to_numpy(),
                mode="lines",
                line=dict(dash="dash"),
                name=f"{solver} true V"
            )
        )

        # ---------------------------------------------------------------------
        # Branch active power same like voltage magnitude above
        # ---------------------------------------------------------------------
        p_true = build_eval_dataframe(res_line_dc[solver], "p_from_mw").astype(float)
        p_est = build_eval_dataframe(res_line_est_dc[solver], "p_from_mw").astype(float)

        p_true_mean = p_true.mean(axis=0)
        p_est_mean = p_est.mean(axis=0)

        p_error = p_est - p_true
        p_rmse = np.sqrt((p_error ** 2).mean(axis=0))
        p_u = k * p_rmse

        x_line = np.arange(len(p_est_mean))

        fig_lp.add_trace(
            go.Scatter(
                x=np.concatenate([x_line, x_line[::-1]]),
                y=np.concatenate([
                    (p_est_mean + p_u).to_numpy(),
                    (p_est_mean - p_u).to_numpy()[::-1]
                ]),
                fill="toself",
                line=dict(width=0),
                opacity=0.2,
                name=f"{solver} uncertainty"
            )
        )
        fig_lp.add_trace(
            go.Scatter(
                x=x_line,
                y=p_est_mean.to_numpy(),
                mode="lines",
                name=f"{solver} estimated P"
            )
        )
        fig_lp.add_trace(
            go.Scatter(
                x=x_line,
                y=p_true_mean.to_numpy(),
                mode="lines",
                line=dict(dash="dash"),
                name=f"{solver} true P"
            )
        )

        # ---------------------------------------------------------------------
        # Voltage uncertainty
        # ---------------------------------------------------------------------
        fig_ve.add_trace(
            go.Scatter(
                x=x_bus,
                y=(100 * vm_u).to_numpy(),
                mode="lines",
                name=f"{solver} V uncertainty [%]"
            )
        )

    fig_vm.update_layout(
        title="Voltage Magnitude Estimation",
        xaxis_title="Bus",
        yaxis_title="Voltage magnitude [p.u.]",
        height=700,
        width=1400
    )

    fig_lp.update_layout(
        title="Branch Active Power Estimation",
        xaxis_title="Branch",
        yaxis_title="Active power [MW]",
        height=700,
        width=1400
    )

    fig_ve.update_layout(
        title="Voltage Magnitude Expanded Uncertainty",
        xaxis_title="Bus",
        yaxis_title="Expanded uncertainty [%]",
        height=700,
        width=1400
    )

    fig_vm.write_html(html_vm_file)
    fig_lp.write_html(html_lp_file)
    fig_ve.write_html(html_ve_file)

    print(f"saved html to: {html_vm_file}")
    print(f"saved html to: {html_lp_file}")
    print(f"saved html to: {html_ve_file}")


def write_bus_voltage_multi_html(
        records: list[dict],
        eval_path: str,
        html_name: str,
        title: str,
        vc_af_bool: bool = False
) -> None:
    """
    Create an HTML file comparing bus-voltage results by iteration.

    Parameters:
        records: Bus-voltage records containing power-flow and estimated values.
        eval_path: Base directory for the generated HTML file.
        html_name: Name of the HTML file.
        title: Main title displayed in the HTML document.
        vc_af_bool: Whether to compare constrained and unconstrained results.

    Returns: None
    """
    # Create the output directory for bus-voltage evaluations.
    save_path = os.path.join(eval_path, "bus")
    os.makedirs(save_path, exist_ok=True)

    save_html = os.path.join(save_path, html_name)

    df = pd.DataFrame(records)

    if df.empty:
        print(f"No records for {save_html}")
        return
    # Constraint comparisons require a column identifying each case.
    if vc_af_bool and "case" not in df.columns:
        print(f"Missing column 'case' for vc_af plot: {save_html}")
        return

    # Sort the iterations to produce a deterministic plot order.
    iterations = sorted(df["iteration"].unique())

    html_parts = ["<html><head><meta charset='utf-8'></head><body>", f"<h1>{title}</h1>"]

    # Create one voltage plot for each simulation iteration.
    for m, iteration in enumerate(iterations):

        group = df[df["iteration"] == iteration]
        fig = go.Figure()

        if vc_af_bool:
            group = group.copy()
            # Sort bus identifiers numerically instead of alphabetically.
            group["bus_sort"] = group["bus"].astype(int)
            group = group.sort_values(["bus_sort", "case"])

            # Separate results with and without violated allocation-factor constraints.
            vc_group = group[group["case"] == "unconstrained"]
            c_group = group[group["case"] == "constrained"]

            colors = {
                "powerflow": "rgba(120, 120, 120, 0.75)",
                "unconstrained_af_estimation": "rgba(31, 119, 180, 0.95)",
                "constrained_af_estimation": "rgba(255, 127, 14, 0.95)",
            }
            # Add the power-flow reference and unconstrained estimation.
            if not vc_group.empty:
                fig.add_trace(go.Scatter(
                    x=vc_group["bus"],
                    y=vc_group["powerflow"],
                    name="Powerflow",
                    mode="lines+markers",
                    line=dict(color=colors["powerflow"], dash="dash"),
                    marker=dict(color=colors["powerflow"]),
                ))

                fig.add_trace(go.Scatter(
                    x=vc_group["bus"],
                    y=vc_group["estimated"],
                    name="SE unconstrained af",
                    mode="lines+markers",
                    line=dict(color=colors["unconstrained_af_estimation"]),
                    marker=dict(color=colors["unconstrained_af_estimation"]),
                ))
            # Add the estimation obtained with allocation-factor constraints.
            if not c_group.empty:
                fig.add_trace(go.Scatter(
                    x=c_group["bus"],
                    y=c_group["estimated"],
                    name="SE constrained af",
                    mode="lines+markers",
                    line=dict(color=colors["constrained_af_estimation"]),
                    marker=dict(color=colors["constrained_af_estimation"]),
                ))

        else:
            # Compare power-flow and state-estimation results directly.
            fig.add_trace(go.Scatter(x=group["bus"], y=group["powerflow"], name="Powerflow", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=group["bus"], y=group["estimated"], name="Estimated", mode="lines+markers"))

        # Configure the appearance of the current iteration's chart.
        fig.update_layout(
            title=f"Iteration {iteration}",
            xaxis_title="Bus",
            yaxis_title="Spannung [p.u.]",
            height=450,
            legend_title="Daten",
        )

        # Optionally use a fixed voltage range:
        # fig.update_yaxes(range=[0.9, 1.05])

        # Complete and save the HTML document.
        html_parts.append(f"<h2>Iteration {iteration}</h2>")
        html_parts.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if m == 0 else False  # include_plotlyjs=True (without internet possible)
            )
        )

    html_parts.append("</body></html>")

    with open(save_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"saved to {save_html}")


def write_bus_power_multi_html(
        records: list[dict],
        eval_path: str,
        html_name: str,
        title: str,
        hide_s_bus: bool = False,
        vc_af_bool: bool = False,
        slack_buses: list[int] | None = None,
) -> None:
    """
    Create an HTML file comparing bus active-power results by iteration.

    Parameters:
       records: Bus-power records containing power-flow and estimated values.
       eval_path: Base directory for the generated HTML file.
       html_name: Name of the HTML file.
       title: Main title displayed in the HTML document.
       hide_s_bus: Whether to exclude slack buses from the plots.
       vc_af_bool: Whether to compare constrained and unconstrained results.
       slack_buses: Bus indices that are treated as slack buses.

    Returns: None
    """
    # Create the output directory for bus-power evaluations.
    save_path = os.path.join(eval_path, "bus")
    os.makedirs(save_path, exist_ok=True)

    save_html = os.path.join(save_path, html_name)

    df = pd.DataFrame(records)

    if df.empty:
        print(f"No records for {save_html}")
        return
    # Constraint comparisons require a column identifying each case.
    if vc_af_bool and "case" not in df.columns:
        print(f"Missing column 'case' for vc_af plot: {save_html}")
        return
    # Sort the iterations to produce a deterministic plot order.
    iterations = sorted(df["iteration"].unique())
    # Initialize the HTML document.
    html_parts = ["<html><head><meta charset='utf-8'></head><body>", f"<h1>{title}</h1>"]

    # Create one grouped bar chart for each simulation iteration.
    for m, iteration in enumerate(iterations):
        group = df[df["iteration"] == iteration]
        # Exclude slack buses if requested.
        if hide_s_bus and slack_buses is not None:
            slack_buses_str = {str(bus) for bus in slack_buses}
            group = group[~group["bus"].isin(slack_buses_str)]

        fig = go.Figure()

        if vc_af_bool:
            group = group.copy()
            # Sort bus identifiers numerically instead of alphabetically.
            group["bus_sort"] = group["bus"].astype(int)
            group = group.sort_values(["bus_sort", "case"])
            # Separate results with and without violated allocation-factor constraints.
            vc_group = group[group["case"] == "unconstrained"]
            c_group = group[group["case"] == "constrained"]

            colors = {
                "powerflow": "rgba(120, 120, 120, 0.45)",
                "unconstrained_af_estimation": "rgba(31, 119, 180, 0.85)",
                "constrained_af_estimation": "rgba(255, 127, 14, 0.85)",
            }
            # Add the power-flow reference and unconstrained estimation.
            if not vc_group.empty:
                fig.add_trace(go.Bar(
                    x=vc_group["bus"],
                    y=vc_group["powerflow"],
                    name="Powerflow",
                    offsetgroup="powerflow",
                    marker_color=colors["powerflow"],
                ))

                fig.add_trace(go.Bar(
                    x=vc_group["bus"],
                    y=vc_group["estimated"],
                    name="SE unconstrained af",
                    offsetgroup="unconstrained_af_estimation",
                    marker_color=colors["unconstrained_af_estimation"],
                ))
            # Add the estimation obtained with allocation-factor constraints.
            if not c_group.empty:
                fig.add_trace(go.Bar(
                    x=c_group["bus"],
                    y=c_group["estimated"],
                    name="SE constrained af",
                    offsetgroup="constrained_af_estimation",
                    marker_color=colors["constrained_af_estimation"],
                ))

        else:
            # Compare power-flow and state-estimation results directly.
            fig.add_trace(go.Bar(x=group["bus"], y=group["powerflow"], name="Powerflow"))
            fig.add_trace(go.Bar(x=group["bus"], y=group["estimated"], name="Estimated"))

        # Configure the appearance of the current iteration's chart.
        fig.update_layout(
            title=f"Iteration {iteration}",
            xaxis_title="Bus",
            yaxis_title="Power [p.u.]",
            barmode="group",
            height=450,
            legend_title="Daten",
        )
        # Load Plotly from the CDN only for the first chart.
        html_parts.append(f"<h2>Iteration {iteration}</h2>")
        html_parts.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if m == 0 else False  # include_plotlyjs=True (without internet possible)
            )
        )

    # Complete and save the HTML document.
    html_parts.append("</body></html>")

    with open(save_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"saved to {save_html}")


def write_line_current_multi_html(
        records: list[dict],
        eval_path: str,
        html_name: str,
        title: str,
        vc_af_bool: bool = False
) -> None:
    """
    Create an HTML file comparing line-current results by iteration.

    Parameters:
        records: Line-current records containing power-flow and estimated values.
        eval_path: Base directory for the generated HTML file.
        html_name: Name of the HTML file.
        title: Main title displayed in the HTML document.
        vc_af_bool: Whether to compare constrained and unconstrained results.

    Returns: None
    """
    # Create the output directory for line-current evaluations.
    save_path = os.path.join(eval_path, "line")
    os.makedirs(save_path, exist_ok=True)

    save_html = os.path.join(save_path, html_name)

    df = pd.DataFrame(records)

    if df.empty:
        print(f"No records for {save_html}")
        return
    # Constraint comparisons require a column identifying each case.
    if vc_af_bool and "case" not in df.columns:
        print(f"Missing column 'case' for vc_af plot: {save_html}")
        return
    # Sort the iterations to produce a deterministic plot order.
    iterations = sorted(df["iteration"].unique())
    # Initialize the HTML document.
    html_parts = ["<html><head><meta charset='utf-8'></head><body>", f"<h1>{title}</h1>"]

    # Create one grouped bar chart for each simulation iteration.
    for m, iteration in enumerate(iterations):

        group = df[df["iteration"] == iteration]
        fig = go.Figure()

        if vc_af_bool:
            group = group.copy()
            # Sort line identifiers numerically instead of alphabetically.
            group["line_sort"] = group["line"].astype(int)
            group = group.sort_values(["line_sort", "case"])
            # Separate results with and without violated allocation-factor constraints.
            vc_group = group[group["case"] == "unconstrained"]
            c_group = group[group["case"] == "constrained"]

            colors = {
                "powerflow": "rgba(120, 120, 120, 0.45)",
                "unconstrained_af_estimation": "rgba(31, 119, 180, 0.85)",
                "constrained_af_estimation": "rgba(255, 127, 14, 0.85)",
            }
            # Use the power-flow result from the unconstrained case as the reference and add its state-estimation
            # result.
            if not vc_group.empty:
                fig.add_trace(go.Bar(
                    x=vc_group["line"],
                    y=vc_group["powerflow"],
                    name="Powerflow",
                    offsetgroup="powerflow",
                    marker_color=colors["powerflow"],
                ))

                fig.add_trace(go.Bar(
                    x=vc_group["line"],
                    y=vc_group["estimated"],
                    name="SE unconstrained af",
                    offsetgroup="unconstrained_af_estimation",
                    marker_color=colors["unconstrained_af_estimation"],
                ))
            # Add the state-estimation result obtained with constraints.
            if not c_group.empty:
                fig.add_trace(go.Bar(
                    x=c_group["line"],
                    y=c_group["estimated"],
                    name="SE constrained af",
                    offsetgroup="constrained_af_estimation",
                    marker_color=colors["constrained_af_estimation"],
                ))

        else:
            # Compare the power-flow and state-estimation results directly.
            fig.add_trace(go.Bar(x=group["line"], y=group["powerflow"], name="Powerflow"))
            fig.add_trace(go.Bar(x=group["line"], y=group["estimated"], name="Estimated"))

        # Configure the appearance of the current iteration's chart.
        fig.update_layout(
            title=f"Iteration {iteration}",
            xaxis_title="Line",
            yaxis_title="Current [p.u.]",
            barmode="group",
            height=450,
            legend_title="Daten",
        )
        # Load Plotly only once in the first chart.
        html_parts.append(f"<h2>Iteration {iteration}</h2>")
        html_parts.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if m == 0 else False  # include_plotlyjs=True (without internet possible)
            )
        )
    # Complete and save the HTML document.
    html_parts.append("</body></html>")

    with open(save_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"saved to {save_html}")


def evaluation_bus(data_path: str, eval_path: str, with_wls: bool = True) -> None:
    """
    Evaluate bus voltages, bus active powers, and line currents.

    Power-flow and state-estimation results are loaded for each solver and exported as interactive HTML visualizations.

    Parameters:
        data_path: Directory containing the simulation results.
        eval_path: Directory in which the HTML files are saved.
        with_wls: Whether to include the AF-WLS solver in the evaluation.

    Returns: None
    """
    # Load solver/iteration pairs for failed simulations.
    failure_set = load_failures(data_path, eval_path)

    # Select the solvers and collect their result files.
    if with_wls:
        solver_ls = ["AF-WLS", "AF-WLAV", "AF-LAV"]
        af_wls_path = os.path.join(data_path, "af_wls")
        af_wls_files = collect_pickle_files(af_wls_path, "af_wls_")
    else:
        solver_ls = ["AF-WLAV", "AF-LAV"]
    af_wlav_path = os.path.join(data_path, "af_wlav")
    af_wlav_files = collect_pickle_files(af_wlav_path, "af_wlav_")
    af_lav_path = os.path.join(data_path, "af_lav")
    af_lav_files = collect_pickle_files(af_lav_path, "af_lav_")
    # Map each solver to its available pickle files.
    if with_wls:
        pkl_files_dc = {
            "AF-WLS": af_wls_files,
            "AF-WLAV": af_wlav_files,
            "AF-LAV": af_lav_files
        }
    else:
        pkl_files_dc = {
            "AF-WLAV": af_wlav_files,
            "AF-LAV": af_lav_files
        }
    # Store the comparison records for each solver.
    bus_voltage_records = {solver: [] for solver in solver_ls}
    bus_active_power_records = {solver: [] for solver in solver_ls}
    line_current_records = {solver: [] for solver in solver_ls}
    # Collect all available simulation iterations across the selected solvers.
    all_iterations = sorted(set().union(*[files.keys() for files in pkl_files_dc.values()]))
    # Load and evaluate every available solver/iteration combination.
    for i in tqdm(all_iterations):
        for solver in solver_ls:
            if (solver, i) in failure_set:
                print(f"skip failure: solver={solver}, iteration={i}")
                continue
            if i not in pkl_files_dc[solver]:
                print(f"missing pickle: solver={solver}, iteration={i}")
                continue

            net_ij = from_pickle(pkl_files_dc[solver][i])
            # Ensure that the required bus result tables are available.
            if not hasattr(net_ij, "res_bus"):
                print(f"missing res_bus: solver={solver}, iteration={i}")
                continue

            if not hasattr(net_ij, "res_bus_est"):
                print(f"missing res_bus_est: solver={solver}, iteration={i}")
                continue
            # Calculate the base current for each line: I_base = S_base / (sqrt(3) * V_base).
            i_base = net_ij.sn_mva / (np.sqrt(3) * net_ij.bus.loc[net_ij.line["from_bus"], "vn_kv"].values)
            # Store power-flow and estimated bus results.
            for bus_idx in net_ij.res_bus.index:
                bus_voltage_records[solver].append({
                    "iteration": f"{i:03d}",
                    "bus": str(bus_idx),
                    "powerflow": float(net_ij.res_bus.loc[bus_idx, "vm_pu"]),
                    "estimated": float(net_ij.res_bus_est.loc[bus_idx, "vm_pu"])
                })
                # Normalize the active power using the network base power.
                bus_active_power_records[solver].append({
                    "iteration": f"{i:03d}",
                    "bus": str(bus_idx),
                    "powerflow": float(net_ij.res_bus.loc[bus_idx, "p_mw"] / net_ij.sn_mva),
                    "estimated": float(net_ij.res_bus_est.loc[bus_idx, "p_mw"] / net_ij.sn_mva)
                })
            # Store normalized power-flow and estimated line currents.
            for line_idx in net_ij.res_line.index:
                line_current_records[solver].append({
                    "iteration": f"{i:03d}",
                    "line": str(line_idx),
                    "powerflow": float(net_ij.res_line.loc[line_idx, "i_ka"] / i_base[line_idx]),
                    "estimated": float(net_ij.res_line_est.loc[line_idx, "i_ka"] / i_base[line_idx])
                })
    # Identify slack buses defined by external grids.
    slack_buses: list[int] = net_ij.ext_grid["bus"].astype(int).tolist()
    # Include buses of generators configured as slack generators.
    if "slack" in net_ij.gen.columns:
        slack_buses.extend(
            net_ij.gen.loc[net_ij.gen["slack"].fillna(False).astype(bool), "bus"].astype(int).tolist()
        )
    # Remove duplicate slack-bus indices and sort the result.
    slack_buses = sorted(set(slack_buses))
    # Generate the evaluation plots for each solver.
    for solver in solver_ls:
        write_bus_voltage_multi_html(
            bus_voltage_records[solver],
            eval_path,
            f"bus_voltages_{solver}.html",
            f"Busspannungen je Iteration - {solver}",
        )
        write_bus_power_multi_html(
            bus_active_power_records[solver],
            eval_path,
            f"bus_power_without_slack_{solver}.html",
            f"Busleistung je Iteration - {solver}",
            True,
            False,
            slack_buses
        )
        write_line_current_multi_html(
            line_current_records[solver],
            eval_path,
            f"line_current_{solver}.html",
            f"Leitungsstrom je Iteration - {solver}"
        )


def show_af_simbench() -> None:
    """
    Display allocation-factor types for SimBench grids with up to 200 buses. Larger grids are skipped to keep the
    output manageable.
    """
    # Retrieve all available SimBench grid codes.
    simbench_grid_ls = sb.collect_all_simbench_codes()
    # sb_grid_ls_3 = ["1-MV-semiurb--0-sw", "1-MV-urban--0-sw", "1-MV-comm--0-sw"]
    # Load each grid and display its allocation-factor types.
    for simbench_grid in tqdm(simbench_grid_ls):
        net_simbench = sb.get_simbench_net(simbench_grid)
        if len(net_simbench.bus) <= 200:
            print(
                f"Grid: {net_simbench}\n"
                f"Allocation Factors Load: {net_simbench.load["type"].unique()}\n"
                f"Allocation Factors Generator: {net_simbench.gen["type"].unique()}\n"
                f"Allocation Factors Static Generator{net_simbench.sgen["type"].unique()}\n"
                f"Number of buses: {len(net_simbench.bus)}\n"
            )
        else:
            print(f"Grid: {simbench_grid} to big.")


def load_vc_af_not_in_failures(
        data_vc_path: str = ".",
        data_c_path: str = ".",
        eval_path: str = "."
) -> set[tuple[str, int]]:
    """
    Load successful simulations with violated allocation-factor constraints.

    Entries from ``violation_constraints.txt`` are excluded if the same solver and iteration occur in either
    ``failures.txt`` file. The remaining entries are saved to ``vc_af_not_in_failures.csv``.

    Parameters:
        data_vc_path: Directory containing unconstrained simulation results.
        data_c_path: Directory containing constrained simulation results.
        eval_path: Directory in which the output CSV file is saved.

    Returns:
        Successful ``(solver, iteration)`` pairs with constraint violations.
    """
    # Load failures from simulations without allocation-factor constraints.
    # All columns are initially read as strings to ensure predictable parsing.
    failures_vc = pd.read_csv(
        os.path.join(data_vc_path, "failures.txt"),
        header=None,
        names=["solver", "iteration", "status"],
        skipinitialspace=True,
        dtype={"solver": str, "iteration": str, "status": str}
    )
    # Convert the iteration identifier to an integer so that it can be matched
    # against the iteration identifiers in the other input files.
    failures_vc["iteration"] = failures_vc["iteration"].astype(int)
    # Load failures from simulations with constrained allocation factors.
    failures_c = pd.read_csv(
        os.path.join(data_c_path, "failures.txt"),
        header=None,
        names=["solver", "iteration", "status"],
        skipinitialspace=True,
        dtype={"solver": str, "iteration": str, "status": str}
    )
    failures_c["iteration"] = failures_c["iteration"].astype(int)
    # Load all solver/iteration pairs for which an allocation factor violated
    # its permitted range in an unconstrained simulation.
    violation_constraints_af = pd.read_csv(
        os.path.join(data_vc_path, "violation_constraints.txt"),
        header=None,
        names=["solver", "iteration", "status"],
        skipinitialspace=True,
        dtype={"solver": str, "iteration": str, "status": str}
    )
    violation_constraints_af["iteration"] = violation_constraints_af["iteration"].astype(int)

    # Convert the DataFrame rows into sets of (solver, iteration) tuples.
    # Sets allow duplicate entries to be removed automatically and support
    # efficient difference operations.
    failures_vc_set = set(zip(failures_vc["solver"], failures_vc["iteration"]))
    failures_c_set = set(zip(failures_c["solver"], failures_c["iteration"]))

    vc_af_set = set(zip(violation_constraints_af["solver"], violation_constraints_af["iteration"]))
    # Keep only violations for which both the constrained and unconstrained simulations completed successfully.
    result_set = vc_af_set - failures_vc_set - failures_c_set

    result_df = pd.DataFrame(
        sorted(result_set),
        columns=["solver", "iteration"]
    )

    result_csv = os.path.join(eval_path, "vc_af_not_in_failures.csv")
    if os.path.exists(result_csv):
        print(f"file {result_csv} exists, ignoring")
    else:
        result_df.to_csv(result_csv, index=False)

    return result_set


def eval_vc_af(
        d_c_path: str,
        d_vc_path: str,
        e_vc_path: str
) -> None:
    r"""
    Compare simulation results obtained with constrained and unconstrained allocation factors.

    The constrained simulations enforce the following bounds on the allocation factor :math:`\alpha`:

    .. math::

        0 \leq \alpha \leq 1

    In the unconstrained simulations, the allocation factor may violate these bounds, i.e. :math:`\alpha < 0` or
    :math:`\alpha > 1`.

    Only successfully completed simulation iterations for which an allocation factor violates the constraints are
    evaluated. For each such iteration, the function compares power-flow and state-estimation results for:

        * bus voltage magnitudes,
        * normalized bus active powers, and
        * normalized line currents.

    The comparison is performed separately for the ``AF-WLAV`` and ``AF-LAV`` solvers. The resulting interactive HTML
    visualizations are written to the ``constraints_combined`` subdirectory of ``e_vc_path``.

    Bus active powers are normalized using the network base power ``net_ij.sn_mva``. Line currents are converted to
    per-unit values using the base current

    .. math::

        I_\mathrm{base} =
        \frac{S_\mathrm{base}}
        {\sqrt{3}\,V_\mathrm{base}}.

    Missing files, result tables, and unsupported solver names are reported to standard output and skipped.

    Parameters:
        d_c_path:
            Path to the simulation results generated with constrained allocation factors,
            where 0 <= :math:`\alpha` <= 1.
        d_vc_path:
            Path to the simulation results generated without allocation-factor constraints, where values such as
            1 < :math:`\alpha` and :math:`\alpha` < 0 are permitted.
        e_vc_path: Output directory in which the evaluated results and generated HTML visualizations are stored.


    Returns: None
    """
    # Identify successfully completed unconstrained simulations in which at
    # least one allocation factor violates the interval 0 <= alpha <= 1.
    # violated constraints (vc) constraints (c)
    vc_af_set = load_vc_af_not_in_failures(d_vc_path, d_c_path, e_vc_path)

    # State-estimation solvers included in the comparison.
    solver_ls = ["AF-WLAV", "AF-LAV"]

    # Collect unconstrained AF-WLAV result files and map each simulation
    # iteration to its corresponding pickle file.
    af_vc_wlav_path = os.path.join(d_vc_path, "af_wlav")
    af_vc_wlav_files = collect_pickle_files(af_vc_wlav_path, "af_wlav_")
    # Collect unconstrained AF-LAV result files.
    af_vc_lav_path = os.path.join(d_vc_path, "af_lav")
    af_vc_lav_files = collect_pickle_files(af_vc_lav_path, "af_lav_")

    pkl_files_vc_dc = {
        "AF-WLAV": af_vc_wlav_files,
        "AF-LAV": af_vc_lav_files,
    }

    # Collect result files from simulations with constrained allocation factors.
    af_c_wlav_path = os.path.join(d_c_path, "af_wlav")
    af_c_wlav_files = collect_pickle_files(af_c_wlav_path, "af_wlav_")
    af_c_lav_path = os.path.join(d_c_path, "af_lav")
    af_c_lav_files = collect_pickle_files(af_c_lav_path, "af_lav_")

    pkl_files_c_dc = {
        "AF-WLAV": af_c_wlav_files,
        "AF-LAV": af_c_lav_files,
    }

    # Store bus-voltage comparison data for every case and solver.
    bus_voltage_records = {
        "unconstrained": {solver: [] for solver in solver_ls},
        "constrained": {solver: [] for solver in solver_ls},
    }
    # Store normalized bus active-power comparison data.
    bus_active_power_records = {
        "unconstrained": {solver: [] for solver in solver_ls},
        "constrained": {solver: [] for solver in solver_ls},
    }
    # Store normalized line-current comparison data.
    line_current_records = {
        "unconstrained": {solver: [] for solver in solver_ls},
        "constrained": {solver: [] for solver in solver_ls},
    }
    # Associate each comparison case with its available pickle files.
    pkl_files_by_case = {
        "unconstrained": pkl_files_vc_dc,
        "constrained": pkl_files_c_dc,
    }
    # Process each solver/iteration pair in a reproducible order.
    for solver, i in tqdm(sorted(vc_af_set, key=lambda x: (x[0], x[1]))):

        if solver not in solver_ls:
            print(f"unknown solver: solver={solver}, iteration={i:03d}")
            continue
        # Evaluate the same iteration both with and without allocation-factor constraints.
        for case_name, pkl_files_dc in pkl_files_by_case.items():

            if i not in pkl_files_dc[solver]:
                print(f"missing pickle: case={case_name}, solver={solver}, iteration={i:03d}")
                continue
            # Load the pandapower network and its result tables.
            net_ij = from_pickle(pkl_files_dc[solver][i])
            # Power-flow bus results are required for the comparison.
            if not hasattr(net_ij, "res_bus"):
                print(f"missing res_bus: case={case_name}, solver={solver}, iteration={i:03d}")
                continue
            # Estimated bus results are required for the comparison.
            if not hasattr(net_ij, "res_bus_est"):
                print(f"missing res_bus_est: case={case_name}, solver={solver}, iteration={i:03d}")
                continue
            # Power-flow line results are required for the comparison.
            if not hasattr(net_ij, "res_line"):
                print(f"missing res_line: case={case_name}, solver={solver}, iteration={i:03d}")
                continue
            # Estimated line results are required for the comparison.
            if not hasattr(net_ij, "res_line_est"):
                print(f"missing res_line_est: case={case_name}, solver={solver}, iteration={i:03d}")
                continue

            # Calculate the base current of every line:
            #     I_base = S_base / (sqrt(3) * V_base)
            # sn_mva and vn_kv yield a base current in kA. This value is later
            # used to convert the line currents to per-unit values.
            i_base = pd.Series(
                net_ij.sn_mva / (np.sqrt(3) * net_ij.bus.loc[net_ij.line["from_bus"], "vn_kv"].values),
                index=net_ij.line.index
            )
            # Bus values can only be compared if both result tables use the same bus indices.
            for bus_idx in net_ij.res_bus.index:
                if not net_ij.res_bus.index.equals(net_ij.res_bus_est.index):
                    print(f"bus index mismatch: case={case_name}, solver={solver}, iteration={i:03d}")
                    continue
                # Store the voltage magnitude from the power flow and the corresponding state-estimation result.
                bus_voltage_records[case_name][solver].append({
                    "iteration": f"{i:03d}",
                    "bus": str(bus_idx),
                    "case": case_name,
                    "powerflow": float(net_ij.res_bus.loc[bus_idx, "vm_pu"]),
                    "estimated": float(net_ij.res_bus_est.loc[bus_idx, "vm_pu"]),
                })
                # Normalize active power by the network base power so that values are represented in per unit.
                bus_active_power_records[case_name][solver].append({
                    "iteration": f"{i:03d}",
                    "bus": str(bus_idx),
                    "case": case_name,
                    "powerflow": float(net_ij.res_bus.loc[bus_idx, "p_mw"] / net_ij.sn_mva),
                    "estimated": float(net_ij.res_bus_est.loc[bus_idx, "p_mw"] / net_ij.sn_mva),
                })
            # Line values can only be compared if the power-flow and estimation result tables use identical line
            # indices.
            for line_idx in net_ij.res_line.index:
                if not net_ij.res_line.index.equals(net_ij.res_line_est.index):
                    print(f"line index mismatch: case={case_name}, solver={solver}, iteration={i:03d}")
                    continue
                # Divide the line currents by the corresponding base current to obtain per-unit values.
                line_current_records[case_name][solver].append({
                    "iteration": f"{i:03d}",
                    "line": str(line_idx),
                    "case": case_name,
                    "powerflow": float(
                        net_ij.res_line.loc[line_idx, "i_ka"] / i_base[line_idx]
                    ),
                    "estimated": float(
                        net_ij.res_line_est.loc[line_idx, "i_ka"] / i_base[line_idx]
                    ),
                })
    # Determine all slack buses. They may be defined either by an external grid or by generators whose "slack" flag is
    # enabled.
    slack_buses: list[int] = net_ij.ext_grid["bus"].astype(int).tolist()
    if "slack" in net_ij.gen.columns:
        slack_buses.extend(
            net_ij.gen.loc[net_ij.gen["slack"].fillna(False).astype(bool), "bus"].astype(int).tolist()
        )
    # Remove duplicate bus indices and provide deterministic ordering.
    slack_buses = sorted(set(slack_buses))
    # Prepare combined records so that constrained and unconstrained results can be displayed in the same visualization.
    combined_bus_voltage_records = {solver: [] for solver in solver_ls}
    combined_bus_active_power_records = {solver: [] for solver in solver_ls}
    combined_line_current_records = {solver: [] for solver in solver_ls}

    for solver in solver_ls:
        combined_bus_voltage_records[solver] = (
                bus_voltage_records["unconstrained"][solver]
                + bus_voltage_records["constrained"][solver]
        )

        combined_bus_active_power_records[solver] = (
                bus_active_power_records["unconstrained"][solver]
                + bus_active_power_records["constrained"][solver]
        )

        combined_line_current_records[solver] = (
                line_current_records["unconstrained"][solver]
                + line_current_records["constrained"][solver]
        )
    # Create the output directory for the combined comparison plots.
    eval_path = os.path.join(e_vc_path, "constraints_combined")
    os.makedirs(eval_path, exist_ok=True)
    # Generate one set of interactive HTML visualizations per solver.
    for solver in solver_ls:
        write_bus_voltage_multi_html(
            combined_bus_voltage_records[solver],
            eval_path,
            f"bus_voltages_unconstrained_af_{solver}.html",
            f"bus voltages per iteration - un/constrained af - {solver}",
            vc_af_bool=True
        )

        write_bus_power_multi_html(
            combined_bus_active_power_records[solver],
            eval_path,
            f"bus_power_without_slack_unconstrained_af_{solver}.html",
            f"bus active power per iteration - un/constrained af - {solver}",
            hide_s_bus=True,
            vc_af_bool=True,
            slack_buses=slack_buses
        )

        write_line_current_multi_html(
            combined_line_current_records[solver],
            eval_path,
            f"line_current_unconstrained_af_{solver}.html",
            f"line current per iteration - un/constrained af - {solver}",
            vc_af_bool=True
        )


def _get_allocation_factor_names(net: pandapowerNet) -> tuple[list[str], int]:
    """
    Determine allocation-factor clusters from the element type columns. Loads and static generators with missing or
    empty types are ignored.

    Parameters:
        net: power net with different clusters.

    Returns: A Tuple with a list with all existing clusters and the number of different clusters.
    """
    cluster_names: set[str] = set()

    for table_name in ("load", "sgen"):
        table = getattr(net, table_name, None)

        if table is None or table.empty or "type" not in table.columns:
            continue

        valid_types = table["type"].dropna().astype(str).str.strip()
        cluster_names.update(cluster for cluster in valid_types if cluster and cluster.lower() != "nan")

    number_af = len(net.load["type"].unique()) + len(net.gen["type"].unique()) + len(net.sgen["type"].unique())
    print(f"number of allocation factors: {number_af}")
    return sorted(cluster_names), number_af


def _drop_not_used_measurement_se(net: pandapowerNet) -> None:
    """
    Remove current measurements and bus active/reactive power measurements for state estimation. The measurement table
    of ``net`` is modified in place. Reason state estimation can work better without these measurements?

    Parameters:
        net: power net with different measurements.

    Returns: None
    """
    measurements = net.measurement

    # Select all current measurements.
    current_mask = measurements["measurement_type"].eq("i")

    # Select active and reactive power measurements assigned to buses.
    bus_power_mask = (measurements["measurement_type"].isin(["p", "q"]) & measurements["element_type"].eq("bus"))

    # Remove the selected measurements and rebuild the row index.
    measurements.drop(
        index=measurements.index[current_mask | bus_power_mask],
        inplace=True
    )
    measurements.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    time_start = time.perf_counter()
    load_dotenv()
    ## chose the case
    test_b: bool = False
    test_case_b: bool = False
    case_sb = "lPV"
    mv_b: bool = False
    ieee14_b: bool = False
    ieee30_b: bool = False
    bus18_b: bool = False
    eval_18bus_b: bool = False
    simbench_ls_b: bool = False
    simbench_b: bool = False
    eval_sb_b: bool = False

    ## set simulation parameters
    num_diff_cases: int = 100
    used_seed: int = 112
    used_seed_pf: int | None = None
    used_seed_m: int | None = None
    with_ortools_b: bool = False
    af_constraints_b: bool = True
    with_wls_b: bool = True
    used_rv: float = .01
    used_ri: float = .01
    used_rp: float = .01
    used_rq: float = .01
    scaling_ranges_dc: dict[str, tuple[float, float]] = {
        "Biomass_MV": (0.8, 1.0),
        "Hydro_MV": (0.6, 1.0),
        "PV_MV": (0.2, 1.0),
        "Wind_MV": (0.3, 1.0),
        "commercial": (0.3, 0.6),
        "lv_RES": (0.3, 0.8),
        "residential": (0.5, 0.8),
    }

    parameters = {
        "num_diff_cases": num_diff_cases,
        "used_seed": used_seed,
        "used_seed_pf": used_seed_pf,
        "used_seed_m": used_seed_m,
        "with_ortools_b": with_ortools_b,
        "af_constraints_b": af_constraints_b,
        "with_wls_b": with_wls_b,
        "used_rv": used_rv,
        "used_ri": used_ri,
        "used_rp": used_rp,
        "used_rq": used_rq,
    }
    parameters.update(
        {f"{cluster}_{bound}": value for cluster, values in scaling_ranges_dc.items()
         for bound, value in zip(("min", "max"), values)}
    )
    para_df = pd.DataFrame([parameters])

    if test_b:
        sb_grid_ls = [
            "1-MV-rural--0-sw", "1-MV-semiurb--0-sw", "1-MV-urban--0-sw", "1-MV-comm--0-sw", "1-MV-rural--0-sw"
        ]
        for sb_grid in sb_grid_ls:
            net_sb = sb.get_simbench_net(sb_grid)
            if test_case_b:
                case_val = sb.get_absolute_values(
                    net_sb, profiles_instead_of_study_cases=False
                )  # if true -> time series
                apply_case(net_sb, case_val, case_sb)  # for cases exist ext_grid vm_pu. This will set automatically and
                # overwrite in the following for loop.
                if ("storage", "p_mw") not in case_val and not net_sb.storage.empty:
                    net_sb.storage["p_mw"] = 0.0
            violation_dc = _check_plot_net(net_sb)
            print(f"finished simbench grid {sb_grid} with violation: {violation_dc}")
        print(f"end")

    if mv_b:
        # rv=.01, rp=.03, rq=.03
        net_mv = pn.mv_oberrhein()
        runpp(net_mv)
        _add_measurements_af(net_mv, 112, 15, .0, .0, .0)

    if ieee14_b:
        net14 = pn.case14()
        runpp(net14)
        _add_measurements_af(net14, 112, 2, .0, .0, .0)

    if ieee30_b:
        net30 = pn.case30()
        runpp(net30)
        _add_measurements_af(net30, 112, 5, .0, .0, .0)

    if bus18_b:
        subdir = "003"

        d_path = os.path.join(str(os.getenv("PATH_DATA_18BUS")), subdir)
        os.makedirs(d_path, exist_ok=True)

        create_random_18_bus_grid_random_estimation(
            d_path,
            100,
            112,
            False,
            True,
            .01,
            .01,
            .01,
        )

        e_path = os.path.join(str(os.getenv("PATH_EVAL_18BUS")), subdir)
        evaluation_af(d_path, e_path)
        evaluation_vp(d_path, e_path, 3.0, True)
        evaluation_bus(d_path, e_path, True)

    if eval_18bus_b:
        constraints_dir = "002"
        vc_dir = "003"

        eval_vc_af(
            os.path.join(str(os.getenv("PATH_DATA_18BUS")), constraints_dir),
            os.path.join(str(os.getenv("PATH_DATA_18BUS")), vc_dir),
            os.path.join(str(os.getenv("PATH_EVAL_18BUS")), vc_dir)
        )

    if simbench_ls_b:
        # "1-MV-semiurb--0-sw", "1-MV-urban--0-sw", "1-MV-comm--0-sw"  "1-MV-rural--0-sw"
        sb_grid_ls = ["1-MV-rural--0-sw"]
        subdir = "001"
        for sb_grid in sb_grid_ls:
            d_path = os.path.join(os.getenv("PATH_DATA_SB", "."), sb_grid, subdir)
            os.makedirs(d_path, exist_ok=True)

            path_para = os.path.join(d_path, "simulation_parameters.csv")
            para_df.to_csv(path_para, sep=";", decimal=",", index=False)
            print(f"simulation parameters saved to: {path_para}")

            net_sb = sb.get_simbench_net(sb_grid)
            net_sb.load["type"] = net_sb.load["type"].fillna("residential")  # only one cluster for load

            create_random_estimations_simbench(
                net=net_sb,
                path=d_path,
                itr=num_diff_cases,
                seed=used_seed,
                seed_pf=used_seed_pf,
                seed_m=used_seed_m,
                with_ortools=with_ortools_b,
                with_af_constraints=af_constraints_b,
                with_wls=with_wls_b,
                rv=used_rv,
                ri=used_ri,
                rp=used_rp,
                rq=used_rq,
                scaling_ranges=scaling_ranges_dc
            )
            e_path = os.path.join(os.getenv("PATH_EVAL_SB", "."), sb_grid, subdir)
            evaluation_af(d_path, e_path)
            evaluation_vp(d_path, e_path, 3.0, False)
            evaluation_bus(d_path, e_path, False)
            # net_sb.measurement.drop(net_sb.measurement.index, inplace=True)
            print(f"finished: {sb_grid}")

    if simbench_b:
        subdir = "000"
        sb_grid_name = "1-MV-comm--0-sw"  # "1-MV-rural--0-sw" "1-MV-urban--0-sw" ## "1-MV-comm--0-sw" -> voltage looks good for state estimation
        d_path = os.path.join(os.getenv("PATH_DATA_SB", "."), sb_grid_name, subdir)
        os.makedirs(d_path, exist_ok=True)
        path_para = os.path.join(d_path, "simulation_parameters.csv")
        para_df.to_csv(path_para, sep=";", decimal=",", index=False)
        print(f"simulation parameters saved to: {path_para}")
        path_scale = os.path.join(d_path, "simulation_parameters.csv")

        net_sb = sb.get_simbench_net(sb_grid_name)

        # delete biomass -> some problems with allocation factors
        mask = net_sb.sgen["type"].eq("Biomass_MV")
        sgen_indices = net_sb.sgen.index[mask]
        print(f"deleted sgen: {net_sb.sgen.loc[sgen_indices]}")
        net_sb.sgen.drop(index=sgen_indices, inplace=True)

        # deactivate_sgen_by_type(net_sb, "Biomass_MV")  # wls get problems with in_service = False ToDo: check this
        # net_elements_ls = get_non_empty_table_names(net_sb)

        _drop_not_used_measurement_se(net_sb)

        # create new clusters for allocation factors
        p_loads = net_sb.load["p_mw"].abs()
        if sb_grid_name == "1-MV-rural--0-sw":
            net_sb.load["type"] = np.select(  # "1-MV-rural--0-sw"  -> wls algorithm does not work
                [
                    p_loads <= 0.3,
                    p_loads > 0.3
                ],
                [
                    "residential",
                    "commercial"
                ],
                default="unknown"
            )
        if sb_grid_name == "1-MV-urban--0-sw":
            net_sb.load["type"] = np.select(  # "1-MV-urban--0-sw"  -> wlav strange results, check these
                [
                    p_loads <= 0.35,
                    p_loads > 0.35
                ],
                [
                    "residential",
                    "commercial"
                ],
                default="unknown"
            )
        if sb_grid_name == "1-MV-comm--0-sw":
            net_sb.load["type"] = np.select(  # "1-MV-comm--0-sw"
                [
                    p_loads <= 0.70,
                    p_loads > 0.70
                ],
                [
                    "residential",
                    "commercial"
                ],
                default="unknown"
            )

        cluster_ls, cluster_nb = _get_allocation_factor_names(net_sb)

        missing_scaling = [af for af in cluster_ls if af not in scaling_ranges_dc]
        if missing_scaling:
            raise ValueError(f"Missing scaling ranges for: {missing_scaling}")

        create_random_estimations_simbench(
            net=net_sb,
            path=d_path,
            itr=num_diff_cases,
            seed=used_seed,
            seed_pf=used_seed_pf,
            seed_m=used_seed_m,
            with_ortools=with_ortools_b,
            with_af_constraints=af_constraints_b,
            with_wls=with_wls_b,
            rv=used_rv,
            ri=used_ri,
            rp=used_rp,
            rq=used_rq,
            scaling_ranges=scaling_ranges_dc
        )

        e_path = os.path.join(os.getenv("PATH_EVAL_SB", "."), sb_grid_name, subdir)
        evaluation_af(d_path, e_path)
        evaluation_vp(d_path, e_path, 3.0, with_wls_b)
        evaluation_bus(d_path, e_path, with_wls_b)

    if eval_sb_b:
        constraints_dir = "000"
        vc_dir = "001"
        sb_grid_name = "1-MV-comm--0-sw"
        eval_vc_af(
            os.path.join(str(os.getenv("PATH_DATA_SB")), sb_grid_name, constraints_dir),
            os.path.join(str(os.getenv("PATH_DATA_SB")), sb_grid_name, vc_dir),
            os.path.join(str(os.getenv("PATH_EVAL_SB")), sb_grid_name, vc_dir)
        )

    linprog_b: bool = False
    if linprog_b:
        net_prob = from_pickle(
            "/mnt/data/pandapower/state-estimation/simbench_grid/1-MV-comm--0-sw/014/af_wlav/af_wlav_065.p"
            # '/mnt/data/pandapower/state-estimation/simbench_grid/1-MV-comm--0-sw/011/af_wlav/prob_af_wlav_048.p'
        )

        af_w_lav = copy.deepcopy(net_prob)
        af_lav = copy.deepcopy(net_prob)
        af_wls = copy.deepcopy(net_prob)

        res_lav = estimate(af_lav, algorithm="af-lp", wlav=False, with_ortools=False)
        res_wls = estimate(af_wls, algorithm="af-wls")
        res_w_lav = estimate(
            af_w_lav,
            algorithm="af-lp",
            wlav=True,
            with_ortools=False,
            linprog_method="highs-ipm",
            maximum_iterations=200
        )

    wls_check_b: bool = False
    if wls_check_b:
        sb_grid_name = "1-MV-comm--0-sw"  ## "1-MV-rural--0-sw" "1-MV-urban--0-sw"
        d_path = os.path.join(os.getenv("PATH_DATA_SB", "."), sb_grid_name)
        net_sb = sb.get_simbench_net(sb_grid_name)
        # runpp(net_sb)

        p_loads = net_sb.load["p_mw"].abs()
        if sb_grid_name == "1-MV-urban--0-sw":
            net_sb.load["type"] = np.select(  # "1-MV-urban--0-sw"  -> wlav strange results, check these
                [
                    p_loads <= 0.35,
                    p_loads > 0.35
                ],
                [
                    "residential",
                    "commercial"
                ],
                default="unknown"
            )
        if sb_grid_name == "1-MV-rural--0-sw":
            net_sb.load["type"] = np.select(  # "1-MV-rural--0-sw"  -> wls algorithm does not work
                [
                    p_loads <= 0.3,
                    p_loads > 0.3
                ],
                [
                    "residential",
                    "commercial"
                ],
                default="unknown"
            )
        if sb_grid_name == "1-MV-comm--0-sw":
            net_sb.load["type"] = np.select(  # "1-MV-comm--0-sw"
                [
                    p_loads <= 0.70,
                    p_loads > 0.70
                ],
                [
                    "residential",
                    "commercial"
                ],
                default="unknown"
            )

        cluster_ls, cluster_nb = _get_allocation_factor_names(net_sb)
        missing_scaling = [af for af in cluster_ls if af not in scaling_ranges_dc]
        if missing_scaling:
            raise ValueError(f"Missing scaling ranges for: {missing_scaling}")

        _drop_not_used_measurement_se(net_sb)

        np.random.seed(112)
        k = _create_simbench_mc_case(net_sb, None, scaling_ranges=scaling_ranges_dc)
        _fill_measurement_values_from_powerflow(net_sb, None, .01, .01, .01, .01)

        # new geodata for simbench grid
        graph = create_nxgraph(net_sb)
        create_generic_coordinates(net_sb, graph, overwrite=True)

        meas_traces = create_measurement_trace(net_sb)
        fig_s_plot = simple_plotly(
            net_sb,
            filename=os.path.join(d_path, f"{sb_grid_name}.html"),
            auto_open=False,
            figsize=2.0,
            bus_size=8,
            additional_traces=meas_traces
        )

        res_wlav = estimate(
            net_sb,
            algorithm="af-lp",
            wlav=True,
            with_ortools=False,
            with_af_constraints=True,
            linprog_method="highs-ipm",
            maximum_iterations=100
        )
        res_wls = estimate(net_sb, algorithm="af-wls", maximum_iterations=200)
        res_lav = af_lav = estimate(
            net_sb,
            algorithm="af-lp",
            wlav=False,
            with_ortools=False,
            with_af_constraints=True,
            linprog_method="highs-ipm",
            maximum_iterations=100
        )
        print(f"wls check ende")

    runtime = time.perf_counter() - time_start
    print(f"calculated in: {timedelta(seconds=runtime)}")
    print(f"you shall not pass")
