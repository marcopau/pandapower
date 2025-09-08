# -*- coding: utf-8 -*-

# This code was written by Matsiushonak Siarhei and Zografos Dimitrios.
# Contributions made on 2025.

import pandapower as pp
import networkx as nx
import numpy as np
from pandapower.estimation.ppc_conversion import ExtendedPPCI, pp2eppci
from pandapower.estimation.observability_analysis.algorithm.analyzer import ObservabilityAnalyzer
from pandapower.estimation.observability_analysis.results import add_connected_components_to_eppci, add_connected_components_to_net
from copy import deepcopy

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)


def run_full_observability(eppci):

    graph = run_observability_analysis_for_eppci(eppci)
    connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

    num_islands = len(connected_components)
    if num_islands > 1:
        std_logger.warning("Attention: multiple islands " \
        "have been identified. State estimation will be run on each observable island of the grid.")
        for i in range(num_islands):
            eppci = define_slack_on_islands(eppci, connected_components[i])
        eppci = adjust_eppci_for_observable_islands(eppci.data, eppci.algorithm)

    return eppci


def run_observability_analysis_for_eppci(eppci: ExtendedPPCI):
    """
    Runs the observability analysis on the given ExtendedPPCI object and updates it with
    the results.

    Parameters:
        eppci (ExtendedPPCI): The extended power flow data structure to analyze.

    Returns:
        nx.MultiGraph: The graph representing observability relationships in the network.
    """
    # Initialize the observability analyzer with the given eppci
    analyzer = ObservabilityAnalyzer(eppci)

    # Execute the observability analysis and obtain the resulting graph
    graph = analyzer.run_observability_analysis()

    # Process the graph to add observable islands to the eppci data
    add_connected_components_to_eppci(graph, eppci)

    return graph


def adjust_eppci_for_observable_islands(ppci, algorithm):

    ppci_new = deepcopy(ppci)

    obs_buses = ppci["bus"][:,-1] != -1
    obs_branches = ppci["branch"][:,-1] != -1
    obs_gens = obs_buses[ppci["gen"][:,0].astype(int)]
    
    ppci_new["bus"] = ppci_new["bus"][obs_buses,:]
    ppci_new["branch"] = ppci_new["branch"][obs_branches,:]
    ppci_new["gen"] = ppci_new["gen"][obs_gens,:]

    eppci = ExtendedPPCI(ppci_new,algorithm)
    
    eppci["bus"][:,0] = np.arange(len(ppci_new["bus"]))
    eppci.obs_bus_mask = obs_buses
    eppci.obs_bus_lookup = -np.ones(len(obs_buses), dtype=int)
    eppci.obs_bus_lookup[obs_buses] = np.arange(len(eppci["bus"]))

    from_indexes = eppci["branch"][:,0].astype(int)
    to_indexes = eppci["branch"][:,1].astype(int)
    
    eppci["branch"][:,0] = eppci.obs_bus_lookup[from_indexes]
    eppci["branch"][:,1] = eppci.obs_bus_lookup[to_indexes]
    eppci.obs_branch_mask = obs_branches
    eppci.obs_branch_lookup = -np.ones(len(obs_branches), dtype=int)
    eppci.obs_branch_lookup[obs_branches] = np.arange(len(eppci["branch"]))

    eppci["gen"][:,0] = eppci.obs_bus_lookup[eppci["gen"][:,0].astype(int)]

    eppci.ppci_original = ppci

    return eppci


def define_slack_on_islands(eppci, connected_components):
    buses_island = list(connected_components)
    bus_type = eppci["bus"][buses_island,1]
    if ~(bus_type==3).any():
        eppci["bus"][buses_island[0],1] = 3
    return eppci


def run_observability_analysis_for_ppnet(
        net: pp.pandapowerNet,
        v_start=None,
        delta_start=None,
        calculate_voltage_angles=True,
        zero_injection=None,
        algorithm='wls'
):
    """
    Runs observability analysis for a given pandapower network and updates it with
    the results in net._observability_lookup.

    Parameters:
        net (pp.pandapowerNet): The pandapower network to analyze.
        v_start (Optional[np.ndarray]): Initial voltage magnitudes (optional).
        delta_start (Optional[np.ndarray]): Initial voltage angles (optional).
        calculate_voltage_angles (bool): Flag to indicate if voltage angles should be calculated.
        zero_injection (Optional[bool]): Option to include zero injection buses in the analysis.
        algorithm (str): The state estimation algorithm to use (default: 'wls').

    Returns:
        nx.MultiGraph: The graph representing observability relationships in the network.
    """
    # Convert pandapower network (ppnet) to extended power flow data structure (eppci)
    _, _, eppci = pp2eppci(
        net,
        v_start=v_start,
        delta_start=delta_start,
        calculate_voltage_angles=calculate_voltage_angles,
        zero_injection=zero_injection,
        algorithm=algorithm
    )

    # Perform observability analysis on the eppci data
    graph = run_observability_analysis_for_eppci(eppci)

    # Map the results back to the pandapower network
    add_connected_components_to_net(eppci, net)

    return graph
