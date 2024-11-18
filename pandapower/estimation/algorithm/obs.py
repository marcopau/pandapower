# Get lists of branch elements

import networkx as nx
import numpy as np

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

WEIGHT = 0
BR_R = 1
BR_X = 2
BR_Z = 3


# Helper function to check for power injection measurements at a bus
def has_injection_measurements(eppci, bus_position):
    # Check if there are measurements of type 'p'
    ppci = eppci.data
    has_p_injection = ~np.isnan(ppci["bus"][bus_position][bus_cols + P])
    return has_p_injection


#
# # Function to identify branches without measurements and without injection at connected buses
def get_elements_without_measurements(eppci):
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


def init_par(tab):
    n = tab.shape[0]
    indices = np.zeros((n, 3), dtype=np.int64)
    indices[:, INDEX] = list(range(n))

    parameters = np.ones((n, 1), dtype=float)

    return indices, parameters, [True for _ in range(n)]


def create_graph_from_eppci(eppci):
    mg = nx.MultiGraph()
    branch = eppci.data["branch"]

    indices, parameter, in_service = init_par(branch)
    indices[:, F_BUS] = branch[:, 0]
    indices[:, T_BUS] = branch[:, 1]

    add_edges(mg, indices, parameter, in_service, None, "line", False,
              'pu')

    # add all buses that were not added when creating branches
    bus = eppci.data["bus"]
    if len(mg.nodes()) < bus.shape[0]:
        for b in set(list(range(bus.shape[0]))) - set(mg.nodes()):
            mg.add_node(b)
    return mg


def print_connected_components(mg, net: pp.pandapowerNet):
    connected_components = list(nx.connected_components(mg))
    number_of_buses = len(net.bus)

    for counter, component in enumerate(connected_components):
        eppci_bus_idx = [i for i in component if i<number_of_buses]
        bus_idx = net.bus.iloc[eppci_bus_idx].index
        # eppci_trafo3w_idx = [i-number_of_buses for i in component if i >= number_of_buses]
        # trafo3w_idx = net.trafo3w.iloc[eppci_trafo3w_idx].index
        print(f"Component {counter}: Bus: {bus_idx.tolist()}")
