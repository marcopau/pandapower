from collections import defaultdict
from itertools import chain

import networkx as nx
import numpy as np
from pandapower.estimation.ppc_conversion import ExtendedPPCI

import pandapower as pp
from pandapower.pypower.idx_brch import branch_cols
from pandapower.topology.create_graph import add_edges

try:
    from pandapower.topology.graph_tool_interface import GraphToolInterface

    graph_tool_available = True
except:
    graph_tool_available = False

from pandapower.pypower.idx_bus import bus_cols
from pandapower.estimation.idx_bus import P
from pandapower.estimation.idx_brch import (P_FROM, P_TO)

INDEX = 0
F_BUS = 1
T_BUS = 2


def get_elements_without_measurements(eppci: ExtendedPPCI) -> list[int]:
    """
    Function to identify branches without measurements and without injection at connected buses.
    """

    elements_to_drop = []
    ppci = eppci.data
    for idx, branch in enumerate(ppci['branch']):
        bus_from, bus_to = int(branch[0]), int(branch[1])
        bus_from_has_p_injection = ~np.isnan(ppci["bus"][bus_from][bus_cols + P])
        bus_to_has_p_injection = ~np.isnan(ppci["bus"][bus_to][bus_cols + P])
        has_p_from_flow = ~np.isnan(branch[branch_cols + P_FROM])
        has_p_to_flow = ~np.isnan(branch[branch_cols + P_TO])
        if not any([bus_from_has_p_injection, bus_to_has_p_injection, has_p_from_flow, has_p_to_flow]):
            elements_to_drop.append(idx)

    return elements_to_drop


def init_par(tab: np.ndarray):
    n = tab.shape[0]
    indices = np.zeros((n, 3), dtype=np.int64)
    indices[:, INDEX] = list(tab[:, -1])
    parameters = np.ones((n, 1), dtype=float)
    return indices, parameters, [True for _ in range(n)]


def create_graph_from_eppci(eppci: ExtendedPPCI) -> nx.MultiGraph:
   # think about second FR
    mg = nx.MultiGraph()
    branch = eppci.data["branch"]

    indices, parameter, in_service = init_par(branch)
    indices[:, F_BUS] = branch[:, 0]
    indices[:, T_BUS] = branch[:, 1]

    add_edges(mg, indices, parameter, in_service, None, "line", False, 'pu')

    # add all buses that were not added when creating branches
    bus = eppci.data["bus"]
    if len(mg.nodes()) < bus.shape[0]:
        for b in set(list(range(bus.shape[0]))) - set(mg.nodes()):
            mg.add_node(b)
    return mg


def print_connected_components(mg, net: pp.pandapowerNet):
    eppci_bus_to_ppnet_map = defaultdict(list)
    for i, v in enumerate(net._pd2ppc_lookups["bus"]):
        if v != -1:
            eppci_bus_to_ppnet_map[v].append(i)

    connected_components = list(nx.connected_components(mg))
    max_bus_index = max(net.bus.index)
    print("\nResult: ")
    all_busses_nested = []
    counter = 0
    for component in connected_components:
        bus_idx = [[j for j in eppci_bus_to_ppnet_map[i] if j <= max_bus_index] for i in component]
        all_busses_nested += bus_idx
        if any(bus_idx):
            bus_idx = [i for i in bus_idx if i]
            print(f"Component {counter}: Len : {len(bus_idx)}  Bus: {bus_idx}")
            counter += 1

    all_busses = list(chain.from_iterable(all_busses_nested))
    if len(all_busses) != len(net.bus):
        raise Exception("!!!!!result doesn't have all buses!!!!!!")
