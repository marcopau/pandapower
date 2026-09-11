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
    load_range: tuple[float, float] = (.5, .8),
    sgen_range: tuple[float, float] = (.3, .5),
) -> dict[str, dict]:
    """
    Generate a random operating point for a SimBench network and perform a power flow calculation.

    A copy of the input network is created and the active and reactive powers of all loads and static generators are
    scaled by uniformly distributed random factors. The resulting network represents the "true" operating state for the
    current Monte Carlo iteration.

    After the power flow calculation, the result tables (``res_*``) of the perturbed network are copied back to the
    original network. Consequently, the original network retains its nominal load and generation values while containing
    the power flow results of the randomly perturbed operating point. This enables state estimation with nominal values
    and measurements derived from varying operating conditions.

    Parameters:
        net: SimBench network containing the nominal load and generation values.
        seed_pf: Attention if None, no seed will be used different to normal use case.
        load_range: Lower and upper bounds of the uniformly distributed scaling factors applied to loads.
        sgen_range: Lower and upper bounds of the uniformly distributed scaling factors applied to sgens.

    Returns:
        Scaling parameters for loads and sgens.
    """

    # ToDo: Check what the scaling factor in load, sgen, gen does
    if seed_pf is not None:
        np.random.seed(seed_pf)

    net_pf = copy.deepcopy(net)

    k = {
        "load": {},
        "sgen": {},
    }

    # scale loads in net_pf
    for idx in net_pf.load.index:
        factor = np.random.uniform(*load_range)
        k["load"][idx] = factor

        net_pf.load.at[idx, "p_mw"] *= factor
        net_pf.load.at[idx, "q_mvar"] *= factor

    # sclae sgen in net_pf
    for idx in net_pf.sgen.index:
        factor = np.random.uniform(*sgen_range)
        k["sgen"][idx] = factor

        net_pf.sgen.at[idx, "p_mw"] *= factor
        net_pf.sgen.at[idx, "q_mvar"] *= factor

    # run powerflow with scaled values
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
        v_b = 11.0,
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
    end   = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
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
        bus = buses[idx + 1] # buses 2-18
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
        neg_af: list,
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
        neg_af: List where information about negative af will save.
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
                af_wls = estimate(net_af_wls, algorithm="af-wls", maximum_iterations=100)  # , af_target_value=.4, af_std_value=.15
                # ToDo: add TypeDict for state estimation
                af_wls["allocation_factors"].index = ["AF-WLS"]  # type: ignore[attr-defined] # set index for saving data
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
            if af_lav["allocation_factors"].loc["AF-LAV"].min() < 0:
                # af_lav["allocation_factors"]["sum"] = af_lav["allocation_factors"].loc["AF-LAV"].sum()
                neg_af.append(f"AF-LAV, {num_it}, negative allocation factors")
                print(f"AF (AF-LAV) should not be negative")

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
            if af_wlav["allocation_factors"].loc["AF-WLAV"].min() < 0:
                neg_af.append(f"AF-WLAV, {num_it}, negative allocation factors")
                print(f"AF (AF-WLAV) should not be negative")
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
    neg_af: list = []
    for i in tqdm(range(itr)):
        name_str = f"{i:03d}"  # number 1 -> 001, 56 -> 056 etc.
        net18, k = _create_18_bus_grid(
            load_range=load_range, com_range=com_range, pv_range=pv_range, wind_range=wind_range
        )
        _create_measurement_18_bus_grid(net=net18, rv=rv, rp=rp, rq=rq)  #
        _calc_different_se(net18, failures, neg_af, name_str, with_ortools, with_wls, data_path)

    # If failures and negative af are not empty, the list will save as txt file.
    if not failures:
        print("failures is empty, very good")
    else:
        with open(os.path.join(data_path, "failures.txt"), "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(failure + "\n")
        print(f"List is not empty: {failures}")

    if not neg_af:
        print("neg_af is empty, also good")
    else:
        with open(os.path.join(data_path, "neg_af.txt"), "w", encoding="utf-8") as f:
            for n_af in neg_af:
                f.write(n_af + "\n")
        print(f"List is not empty: {neg_af}")


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
        load_range: tuple[float, float] = (.5, .8),
        sgen_range: tuple[float, float] = (.3, .5),
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
        load_range:
            Lower and upper bounds of the uniformly distributed scaling factors applied to loads for
            :func:`_create_simbench_mc_case`.
        sgen_range:
            Lower and upper bounds of the uniformly distributed scaling factors applied to sgens for
            :func:`_create_simbench_mc_case`.
    Returns: None
    """

    np.random.seed(seed)
    failures: list = []  # The list contains information about the final status of the state estimation
    neg_af: list = []
    for i in range(itr):
        name_str = f"{i:03d}"  # number 1 -> 001, 56 -> 056 etc.
        k = _create_simbench_mc_case(net, seed_pf, load_range, sgen_range)
        _fill_measurement_values_from_powerflow(net, seed_m, rv, ri, rp, rq)
        _calc_different_se(net, failures, neg_af, name_str, with_ortools, with_af_constraints, with_wls, path)

    if not failures:
        print("List is empty, very good")
    else:
        with open(os.path.join(path, "failures.txt"), "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(failure + "\n")
        print(f"List is not empty: {failures}")
    if not neg_af:
        print("neg_af is empty, also good")
    else:
        with open(os.path.join(path, "neg_af.txt"), "w", encoding="utf-8") as f:
            for n_af in neg_af:
                f.write(n_af + "\n")
        print(f"List is not empty: {neg_af}")


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


def evaluation_af(data_path: str = ".", eval_path: str = "." ) -> None:
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
        neg_af_bool: bool = False
) -> None:
    save_path = os.path.join(eval_path, "bus")
    os.makedirs(save_path, exist_ok=True)

    save_html = os.path.join(save_path, html_name)

    df = pd.DataFrame(records)

    if df.empty:
        print(f"No records for {save_html}")
        return

    if neg_af_bool and "case" not in df.columns:
        print(f"Missing column 'case' for neg_af plot: {save_html}")
        return

    iterations = sorted(df["iteration"].unique())

    html_parts = [
        "<html><head><meta charset='utf-8'></head><body>",
        f"<h1>{title}</h1>",
    ]

    for k, iteration in enumerate(iterations):
        group = df[df["iteration"] == iteration]

        fig = go.Figure()

        if neg_af_bool:
            group = group.copy()
            group["bus_sort"] = group["bus"].astype(int)
            group = group.sort_values(["bus_sort", "case"])

            neg_group = group[group["case"] == "neg"]
            pos_group = group[group["case"] == "pos"]

            colors = {
                "powerflow": "rgba(120, 120, 120, 0.75)",
                "neg_estimated": "rgba(31, 119, 180, 0.95)",
                "pos_estimated": "rgba(255, 127, 14, 0.95)",
            }

            if not neg_group.empty:
                fig.add_trace(go.Scatter(
                    x=neg_group["bus"],
                    y=neg_group["powerflow"],
                    name="Powerflow",
                    mode="lines+markers",
                    line=dict(color=colors["powerflow"], dash="dash"),
                    marker=dict(color=colors["powerflow"]),
                ))

                fig.add_trace(go.Scatter(
                    x=neg_group["bus"],
                    y=neg_group["estimated"],
                    name="SE without constraints on af",
                    mode="lines+markers",
                    line=dict(color=colors["neg_estimated"]),
                    marker=dict(color=colors["neg_estimated"]),
                ))

            if not pos_group.empty:
                fig.add_trace(go.Scatter(
                    x=pos_group["bus"],
                    y=pos_group["estimated"],
                    name="SE with constraints on af",
                    mode="lines+markers",
                    line=dict(color=colors["pos_estimated"]),
                    marker=dict(color=colors["pos_estimated"]),
                ))

        else:
            fig.add_trace(go.Scatter(
                x=group["bus"],
                y=group["powerflow"],
                name="Powerflow",
                mode="lines+markers",
            ))

            fig.add_trace(go.Scatter(
                x=group["bus"],
                y=group["estimated"],
                name="Estimated",
                mode="lines+markers",
            ))

        fig.update_layout(
            title=f"Iteration {iteration}",
            xaxis_title="Bus",
            yaxis_title="Spannung [p.u.]",
            height=450,
            legend_title="Daten",
        )

        # fig.update_yaxes(range=[0.9, 1.05])

        html_parts.append(f"<h2>Iteration {iteration}</h2>")
        html_parts.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if k == 0 else False
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
        neg_af_bool: bool = False,
        slack_buses: list[int] | None = None,
) -> None:
    save_path = os.path.join(eval_path, "bus")
    os.makedirs(save_path, exist_ok=True)

    save_html = os.path.join(save_path, html_name)

    df = pd.DataFrame(records)

    if df.empty:
        print(f"No records for {save_html}")
        return

    if neg_af_bool and "case" not in df.columns:
        print(f"Missing column 'case' for neg_af plot: {save_html}")
        return

    iterations = sorted(df["iteration"].unique())

    html_parts = [
        "<html><head><meta charset='utf-8'></head><body>",
        f"<h1>{title}</h1>",
    ]

    for k, iteration in enumerate(iterations):
        group = df[df["iteration"] == iteration]

        if hide_s_bus and slack_buses is not None:
            slack_buses_str = {str(bus) for bus in slack_buses}
            group = group[~group["bus"].isin(slack_buses_str)]

        fig = go.Figure()

        if neg_af_bool:
            group = group.copy()
            group["bus_sort"] = group["bus"].astype(int)
            group = group.sort_values(["bus_sort", "case"])

            neg_group = group[group["case"] == "neg"]
            pos_group = group[group["case"] == "pos"]

            colors = {
                "powerflow": "rgba(120, 120, 120, 0.45)",
                "neg_estimated": "rgba(31, 119, 180, 0.85)",
                "pos_estimated": "rgba(255, 127, 14, 0.85)",
            }

            if not neg_group.empty:
                fig.add_trace(go.Bar(
                    x=neg_group["bus"],
                    y=neg_group["powerflow"],
                    name="Powerflow",
                    offsetgroup="powerflow",
                    marker_color=colors["powerflow"],
                ))

                fig.add_trace(go.Bar(
                    x=neg_group["bus"],
                    y=neg_group["estimated"],
                    name="SE without constraints on af",
                    offsetgroup="neg_estimated",
                    marker_color=colors["neg_estimated"],
                ))

            if not pos_group.empty:
                fig.add_trace(go.Bar(
                    x=pos_group["bus"],
                    y=pos_group["estimated"],
                    name="SE with constraints on af",
                    offsetgroup="pos_estimated",
                    marker_color=colors["pos_estimated"],
                ))

        else:
            fig.add_trace(go.Bar(
                x=group["bus"],
                y=group["powerflow"],
                name="Powerflow",
            ))

            fig.add_trace(go.Bar(
                x=group["bus"],
                y=group["estimated"],
                name="Estimated",
            ))

        fig.update_layout(
            title=f"Iteration {iteration}",
            xaxis_title="Bus",
            yaxis_title="Power [p.u.]",
            barmode="group",
            height=450,
            legend_title="Daten",
        )

        html_parts.append(f"<h2>Iteration {iteration}</h2>")
        html_parts.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if k == 0 else False
            )
        )

    html_parts.append("</body></html>")

    with open(save_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"saved to {save_html}")


def write_line_current_multi_html(
        records: list[dict],
        eval_path: str,
        html_name: str,
        title: str,
        neg_af_bool: bool = False
) -> None:

    save_path = os.path.join(eval_path, "line")
    os.makedirs(save_path, exist_ok=True)

    save_html = os.path.join(save_path, html_name)

    df = pd.DataFrame(records)

    if df.empty:
        print(f"No records for {save_html}")
        return

    if neg_af_bool and "case" not in df.columns:
        print(f"Missing column 'case' for neg_af plot: {save_html}")
        return

    iterations = sorted(df["iteration"].unique())

    html_parts = [
        "<html><head><meta charset='utf-8'></head><body>",
        f"<h1>{title}</h1>",
    ]

    for k, iteration in enumerate(iterations):
        group = df[df["iteration"] == iteration]

        fig = go.Figure()

        if neg_af_bool:
            group = group.copy()
            group["line_sort"] = group["line"].astype(int)
            group = group.sort_values(["line_sort", "case"])

            neg_group = group[group["case"] == "neg"]
            pos_group = group[group["case"] == "pos"]

            colors = {
                "powerflow": "rgba(120, 120, 120, 0.45)",
                "neg_estimated": "rgba(31, 119, 180, 0.85)",
                "pos_estimated": "rgba(255, 127, 14, 0.85)",
            }

            if not neg_group.empty:
                fig.add_trace(go.Bar(
                    x=neg_group["line"],
                    y=neg_group["powerflow"],
                    name="Powerflow",
                    offsetgroup="powerflow",
                    marker_color=colors["powerflow"],
                ))

                fig.add_trace(go.Bar(
                    x=neg_group["line"],
                    y=neg_group["estimated"],
                    name="SE without constraints on af",
                    offsetgroup="neg_estimated",
                    marker_color=colors["neg_estimated"],
                ))

            if not pos_group.empty:
                fig.add_trace(go.Bar(
                    x=pos_group["line"],
                    y=pos_group["estimated"],
                    name="SE with constraints on af",
                    offsetgroup="pos_estimated",
                    marker_color=colors["pos_estimated"],
                ))

        else:
            fig.add_trace(go.Bar(
                x=group["line"],
                y=group["powerflow"],
                name="Powerflow",
            ))

            fig.add_trace(go.Bar(
                x=group["line"],
                y=group["estimated"],
                name="Estimated",
            ))

        fig.update_layout(
            title=f"Iteration {iteration}",
            xaxis_title="Line",
            yaxis_title="Current [p.u.]",
            barmode="group",
            height=450,
            legend_title="Daten",
        )

        # fig.update_yaxes(range=[0.9, 1.05])

        html_parts.append(f"<h2>Iteration {iteration}</h2>")
        html_parts.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if k == 0 else False
            )
        )

    html_parts.append("</body></html>")

    with open(save_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"saved to {save_html}")


def evaluation_bus(data_path: str, eval_path: str, with_wls: bool = True) -> None:
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
            "AF-LAV": af_lav_files
        }
    else:
        pkl_files_dc = {
            "AF-WLAV": af_wlav_files,
            "AF-LAV": af_lav_files
        }

    bus_voltage_records = {solver: [] for solver in solver_ls}
    bus_active_power_records = {solver: [] for solver in solver_ls}
    line_current_records = {solver: [] for solver in solver_ls}

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

            i_base = net_ij.sn_mva / (np.sqrt(3) * net_ij.bus.loc[net_ij.line["from_bus"], "vn_kv"].values)

            for bus_idx in net_ij.res_bus.index:
                bus_voltage_records[solver].append({
                    "iteration": f"{i:03d}",
                    "bus": str(bus_idx),
                    "powerflow": float(net_ij.res_bus.loc[bus_idx, "vm_pu"]),
                    "estimated": float(net_ij.res_bus_est.loc[bus_idx, "vm_pu"])
                })
                bus_active_power_records[solver].append({
                    "iteration": f"{i:03d}",
                    "bus": str(bus_idx),
                    "powerflow": float(net_ij.res_bus.loc[bus_idx, "p_mw"] / net_ij.sn_mva),
                    "estimated": float(net_ij.res_bus_est.loc[bus_idx, "p_mw"] / net_ij.sn_mva)
                })
            for line_idx in net_ij.res_line.index:
                line_current_records[solver].append({
                    "iteration": f"{i:03d}",
                    "line": str(line_idx),
                    "powerflow": float(net_ij.res_line.loc[line_idx, "i_ka"] / i_base[line_idx]),
                    "estimated": float(net_ij.res_line_est.loc[line_idx, "i_ka"] / i_base[line_idx])
                })

    slack_buses: list[int] = net_ij.ext_grid["bus"].astype(int).tolist()

    if "slack" in net_ij.gen.columns:
        slack_buses.extend(
            net_ij.gen.loc[net_ij.gen["slack"].fillna(False).astype(bool), "bus"].astype(int).tolist()
        )
    slack_buses = sorted(set(slack_buses))

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
            f"bus_power_without_slack{solver}.html",
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


def show_af_simbench():
    simbench_grid_ls = sb.collect_all_simbench_codes()
    sb_grid_ls_3 = ["1-MV-semiurb--0-sw", "1-MV-urban--0-sw", "1-MV-comm--0-sw"]
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

def load_neg_af_not_in_failures(
    data_neg_path: str = ".",
    data_pos_path: str = ".",
    eval_path: str = "."
) -> set[tuple[str, int]]:
    """
    Load entries from neg_af.txt that are not present in failures.txt.

    Returns:
        Set of (solver, iteration) tuples that occur in neg_af.txt
        but not in failures.txt.
    """

    failures_neg = pd.read_csv(
        os.path.join(data_neg_path, "failures.txt"),
        header=None,
        names=["solver", "iteration", "status"],
        skipinitialspace=True,
        dtype={"solver": str, "iteration": str, "status": str}
    )
    failures_neg["iteration"] = failures_neg["iteration"].astype(int)

    failures_pos = pd.read_csv(
        os.path.join(data_pos_path, "failures.txt"),
        header=None,
        names=["solver", "iteration", "status"],
        skipinitialspace=True,
        dtype={"solver": str, "iteration": str, "status": str}
    )
    failures_pos["iteration"] = failures_pos["iteration"].astype(int)

    neg_af = pd.read_csv(
        os.path.join(data_neg_path, "neg_af.txt"),
        header=None,
        names=["solver", "iteration", "status"],
        skipinitialspace=True,
        dtype={"solver": str, "iteration": str, "status": str}
    )
    neg_af["iteration"] = neg_af["iteration"].astype(int)

    failures_neg_set = set(zip(failures_neg["solver"], failures_neg["iteration"]))
    failures_pos_set = set(zip(failures_pos["solver"], failures_pos["iteration"]))

    neg_af_set = set(zip(neg_af["solver"], neg_af["iteration"]))

    result_set = neg_af_set - failures_neg_set - failures_pos_set

    result_df = pd.DataFrame(
        sorted(result_set),
        columns=["solver", "iteration"]
    )

    result_csv = os.path.join(eval_path, "neg_af_not_in_failures.csv")
    if os.path.exists(result_csv):
        print(f"file {result_csv} exists, ignoring")
    else:
        result_df.to_csv(result_csv, index=False)

    return result_set


def eval_neg_af(
        d_pos_path: str,
        d_neg_path: str,
        e_pos_path: str,
        e_neg_path: str
) -> None:

    neg_af_set = load_neg_af_not_in_failures(d_neg_path, d_pos_path, e_neg_path)

    solver_ls = ["AF-WLAV", "AF-LAV"]


    af_neg_wlav_path = os.path.join(d_neg_path, "af_wlav")
    af_neg_wlav_files = collect_pickle_files(af_neg_wlav_path, "af_wlav_")

    af_neg_lav_path = os.path.join(d_neg_path, "af_lav")
    af_neg_lav_files = collect_pickle_files(af_neg_lav_path, "af_lav_")

    pkl_files_neg_dc = {
        "AF-WLAV": af_neg_wlav_files,
        "AF-LAV": af_neg_lav_files,
    }

    af_pos_wlav_path = os.path.join(d_pos_path, "af_wlav")
    af_pos_wlav_files = collect_pickle_files(af_pos_wlav_path, "af_wlav_")
    af_pos_lav_path = os.path.join(d_pos_path, "af_lav")
    af_pos_lav_files = collect_pickle_files(af_pos_lav_path, "af_lav_")

    pkl_files_pos_dc = {
        "AF-WLAV": af_pos_wlav_files,
        "AF-LAV": af_pos_lav_files,
    }

    bus_voltage_records = {
        "neg": {solver: [] for solver in solver_ls},
        "pos": {solver: [] for solver in solver_ls},
    }

    bus_active_power_records = {
        "neg": {solver: [] for solver in solver_ls},
        "pos": {solver: [] for solver in solver_ls},
    }

    line_current_records = {
        "neg": {solver: [] for solver in solver_ls},
        "pos": {solver: [] for solver in solver_ls},
    }

    pkl_files_by_case = {
        "neg": pkl_files_neg_dc,
        "pos": pkl_files_pos_dc,
    }

    for solver, i in tqdm(sorted(neg_af_set, key=lambda x: (x[0], x[1]))):

        if solver not in solver_ls:
            print(f"unknown solver: solver={solver}, iteration={i:03d}")
            continue

        for case_name, pkl_files_dc in pkl_files_by_case.items():

            if i not in pkl_files_dc[solver]:
                print(f"missing pickle: case={case_name}, solver={solver}, iteration={i:03d}")
                continue

            net_ij = from_pickle(pkl_files_dc[solver][i])

            if not hasattr(net_ij, "res_bus"):
                print(f"missing res_bus: case={case_name}, solver={solver}, iteration={i:03d}")
                continue

            if not hasattr(net_ij, "res_bus_est"):
                print(f"missing res_bus_est: case={case_name}, solver={solver}, iteration={i:03d}")
                continue

            if not hasattr(net_ij, "res_line"):
                print(f"missing res_line: case={case_name}, solver={solver}, iteration={i:03d}")
                continue

            if not hasattr(net_ij, "res_line_est"):
                print(f"missing res_line_est: case={case_name}, solver={solver}, iteration={i:03d}")
                continue

            i_base = pd.Series(
                net_ij.sn_mva / (np.sqrt(3) * net_ij.bus.loc[net_ij.line["from_bus"], "vn_kv"].values),
                index=net_ij.line.index
            )

            for bus_idx in net_ij.res_bus.index:
                if not net_ij.res_bus.index.equals(net_ij.res_bus_est.index):
                    print(f"bus index mismatch: case={case_name}, solver={solver}, iteration={i:03d}")
                    continue

                bus_voltage_records[case_name][solver].append({
                    "iteration": f"{i:03d}",
                    "bus": str(bus_idx),
                    "case": case_name,
                    "powerflow": float(net_ij.res_bus.loc[bus_idx, "vm_pu"]),
                    "estimated": float(net_ij.res_bus_est.loc[bus_idx, "vm_pu"]),
                })

                bus_active_power_records[case_name][solver].append({
                    "iteration": f"{i:03d}",
                    "bus": str(bus_idx),
                    "case": case_name,
                    "powerflow": float(net_ij.res_bus.loc[bus_idx, "p_mw"] / net_ij.sn_mva),
                    "estimated": float(net_ij.res_bus_est.loc[bus_idx, "p_mw"] / net_ij.sn_mva),
                })

            for line_idx in net_ij.res_line.index:
                if not net_ij.res_line.index.equals(net_ij.res_line_est.index):
                    print(f"line index mismatch: case={case_name}, solver={solver}, iteration={i:03d}")
                    continue
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

    slack_buses: list[int] = net_ij.ext_grid["bus"].astype(int).tolist()
    if "slack" in net_ij.gen.columns:
        slack_buses.extend(
            net_ij.gen.loc[net_ij.gen["slack"].fillna(False).astype(bool), "bus"].astype(int).tolist()
        )
    slack_buses = sorted(set(slack_buses))

    combined_bus_voltage_records = {solver: [] for solver in solver_ls}
    combined_bus_active_power_records = {solver: [] for solver in solver_ls}
    combined_line_current_records = {solver: [] for solver in solver_ls}

    for solver in solver_ls:
        combined_bus_voltage_records[solver] = (
                bus_voltage_records["neg"][solver]
                + bus_voltage_records["pos"][solver]
        )

        combined_bus_active_power_records[solver] = (
                bus_active_power_records["neg"][solver]
                + bus_active_power_records["pos"][solver]
        )

        combined_line_current_records[solver] = (
                line_current_records["neg"][solver]
                + line_current_records["pos"][solver]
        )

    eval_path = os.path.join(e_neg_path, "pos_neg_combined")
    os.makedirs(eval_path, exist_ok=True)

    for solver in solver_ls:
        write_bus_voltage_multi_html(
            combined_bus_voltage_records[solver],
            eval_path,
            f"bus_voltages_pos_neg_{solver}.html",
            f"Busspannungen je Iteration - pos/neg - {solver}",
            neg_af_bool=True
        )

        write_bus_power_multi_html(
            combined_bus_active_power_records[solver],
            eval_path,
            f"bus_power_without_slack_pos_neg_{solver}.html",
            f"Busleistung je Iteration - pos/neg - {solver}",
            hide_s_bus=True,
            neg_af_bool=True,
            slack_buses=slack_buses
        )

        write_line_current_multi_html(
            combined_line_current_records[solver],
            eval_path,
            f"line_current_pos_neg_{solver}.html",
            f"Leitungsstrom je Iteration - pos/neg - {solver}",
            neg_af_bool=True
        )


def _get_allocation_factor_names(net: pandapowerNet) -> list[str]:
    """
    Determine allocation-factor clusters from the element type columns.

    Loads and static generators with missing or empty types are ignored.
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
    return sorted(cluster_names)


def _build_radial_children(
    net: pandapowerNet,
    root_bus: int
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, int]]:
    """
    Orient the network as a tree starting from root_bus.

    Returns
    -------
    children:
        Mapping parent bus -> list of (child bus, line index).
    parent:
        Mapping child bus -> parent bus.
    """
    adjacency: dict[int, list[tuple[int, int]]] = {
        int(bus): [] for bus in net.bus.index
    }

    for line_idx, line in net.line.iterrows():
        if "in_service" in line and not bool(line["in_service"]):
            continue

        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])

        adjacency[from_bus].append((to_bus, int(line_idx)))
        adjacency[to_bus].append((from_bus, int(line_idx)))

    children: dict[int, list[tuple[int, int]]] = {
        int(bus): [] for bus in net.bus.index
    }
    parent: dict[int, int] = {}
    visited = {root_bus}
    queue = [root_bus]

    while queue:
        current = queue.pop(0)

        for neighbor, line_idx in adjacency[current]:
            if neighbor in visited:
                continue

            visited.add(neighbor)
            parent[neighbor] = current
            children[current].append((neighbor, line_idx))
            queue.append(neighbor)

    if len(visited) != len(net.bus):
        missing = sorted(set(net.bus.index.astype(int)) - visited)
        raise ValueError(
            "The network is disconnected or contains buses that cannot be "
            f"reached from root bus {root_bus}: {missing}"
        )

    return children, parent


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
    af_constraints_b: bool = False
    with_wls_b: bool = True
    used_rv: float = .01
    used_ri: float = .01
    used_rp: float = .01
    used_rq: float = .01
    l_range: tuple[float, float] = (.5, .8)
    s_range: tuple[float, float] = (.3, .5)

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
        "load_range_min": l_range[0],
        "load_range_max": l_range[1],
        "sgen_range_min": s_range[0],
        "sgen_range_max": s_range[1],
    }
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
        _add_measurements_af(net_mv, 112,15, .0, .0, .0)

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
        pos_dir = "002"
        neg_dir = "003"

        eval_neg_af(
            os.path.join(str(os.getenv("PATH_DATA_18BUS")), pos_dir),
            os.path.join(str(os.getenv("PATH_DATA_18BUS")), neg_dir),
            os.path.join(str(os.getenv("PATH_EVAL_18BUS")), pos_dir),
            os.path.join(str(os.getenv("PATH_EVAL_18BUS")), neg_dir)
        )

    if simbench_ls_b:
        sb_grid_ls = ["1-MV-rural--0-sw"]  #"1-MV-semiurb--0-sw", "1-MV-urban--0-sw", "1-MV-comm--0-sw"  "1-MV-rural--0-sw"
        subdir = "001"
        for sb_grid in sb_grid_ls:
            d_path = os.path.join(os.getenv("PATH_DATA_SB", "."), sb_grid, subdir)
            os.makedirs(d_path, exist_ok=True)

            path_para = os.path.join(d_path, "simulation_parameters.csv")
            para_df.to_csv(path_para, sep=";", decimal=",", index=False)
            print(f"simulation parameters saved to: {path_para}")

            net_sb = sb.get_simbench_net(sb_grid)
            net_sb.load["type"] = net_sb.load["type"].fillna("residential")

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
                load_range=l_range,
                sgen_range=s_range
            )
            e_path = os.path.join(os.getenv("PATH_EVAL_SB", "."), sb_grid, subdir)
            evaluation_af(d_path, e_path)
            evaluation_vp(d_path, e_path, 3.0, False)
            evaluation_bus(d_path, e_path, False)
            # net_sb.measurement.drop(net_sb.measurement.index, inplace=True)
            print(f"finished: {sb_grid}")

    if simbench_b:
        subdir = "040"
        sb_grid_name = "1-MV-comm--0-sw"  # "1-MV-rural--0-sw" "1-MV-urban--0-sw" ## "1-MV-comm--0-sw" -> voltage looks good for state estimation
        d_path = os.path.join(os.getenv("PATH_DATA_SB", "."), sb_grid_name, subdir)
        os.makedirs(d_path, exist_ok=True)
        path_para = os.path.join(d_path, "simulation_parameters.csv")
        para_df.to_csv(path_para, sep=";", decimal=",", index=False)
        print(f"simulation parameters saved to: {path_para}")

        net_sb = sb.get_simbench_net(sb_grid_name)

        # delete biomass -> some problems with allocation factors
        mask = net_sb.sgen["type"].eq("Biomass_MV")
        sgen_indices = net_sb.sgen.index[mask]
        net_sb.sgen.drop(index=sgen_indices, inplace=True)
        print(f"deleted sgen: {net_sb.sgen.loc[sgen_indices]}")

        # deactivate_sgen_by_type(net_sb, "Biomass_MV")  # wls get problems with in_service = False ToDo: check this
        # net_elements_ls = get_non_empty_table_names(net_sb)

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
            load_range=l_range,
            sgen_range=s_range
        )

        e_path = os.path.join(os.getenv("PATH_EVAL_SB", "."), sb_grid_name, subdir)
        evaluation_af(d_path, e_path)
        evaluation_vp(d_path, e_path, 3.0, with_wls_b)
        evaluation_bus(d_path, e_path, with_wls_b)

    if eval_sb_b:
        pos_dir = "000"
        neg_dir = "001"
        sb_grid_name = "1-MV-comm--0-sw"
        eval_neg_af(
            os.path.join(str(os.getenv("PATH_DATA_SB")), sb_grid_name, pos_dir),
            os.path.join(str(os.getenv("PATH_DATA_SB")), sb_grid_name, neg_dir),
            os.path.join(str(os.getenv("PATH_EVAL_SB")), sb_grid_name, pos_dir),
            os.path.join(str(os.getenv("PATH_EVAL_SB")), sb_grid_name, neg_dir)
        )

    linprog_b: bool = False
    if linprog_b:
        net_prob = from_pickle(
            "/mnt/data/pandapower/state-estimation/simbench_grid/1-MV-comm--0-sw/014/af_wlav/af_wlav_065.p"  # '/mnt/data/pandapower/state-estimation/simbench_grid/1-MV-comm--0-sw/011/af_wlav/prob_af_wlav_048.p'
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

    wls_check_b: bool = True
    if wls_check_b:
        sb_grid_name = "1-MV-comm--0-sw"
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

        np.random.seed(112)
        k = _create_simbench_mc_case(net_sb, None)
        _fill_measurement_values_from_powerflow(net_sb, None, .01, .01, .01, .01)

        # new geodata for simbench grid
        graph = create_nxgraph(net_sb)
        create_generic_coordinates(net_sb, graph, overwrite=True)

        meas_traces = create_measurement_trace(net_sb)
        fig_s_plot = simple_plotly(
            net_sb,
            filename=os.path.join(d_path, "final.html"),
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
