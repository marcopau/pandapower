import pandapower as pp
from pandapower.estimation.ppc_conversion import ExtendedPPCI, pp2eppci
from pandapower.observability_analysis.algorithm.analyzer import ObservabilityAnalyzer
from pandapower.observability_analysis.network_utils import print_connected_components
from pandapower.observability_analysis.results import add_connected_components_to_eppci, add_connected_components_to_net


def run_observability_analysis_for_eppci(eppci: ExtendedPPCI):
    analyzer = ObservabilityAnalyzer(eppci)
    graph = analyzer.run_observability_analysis()
    add_connected_components_to_eppci(graph, eppci)
    return graph


def run_observability_analysis_for_ppnet(
        net: pp.pandapowerNet,
        v_start=None,
        delta_start=None,
        calculate_voltage_angles=True,
        zero_injection=None,
        algorithm='wls'
):
    _, _, eppci = pp2eppci(net, v_start=v_start, delta_start=delta_start,
                           calculate_voltage_angles=calculate_voltage_angles,
                           zero_injection=zero_injection, algorithm=algorithm,
                           )
    graph = run_observability_analysis_for_eppci(eppci)
    add_connected_components_to_net(eppci, net)
    # print components for debug:
    print_connected_components(graph, net)
    return graph


