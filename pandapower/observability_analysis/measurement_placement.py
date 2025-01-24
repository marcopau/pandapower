# -*- coding: utf-8 -*-

# This code was written by Matsiushonak Siarhei and Zografos Dimitrios.
# Contributions made on 2025.

import pandapower as pp
from pandapower.estimation.ppc_conversion import ExtendedPPCI, pp2eppci
from pandapower.observability_analysis.algorithm.pseudo_measurements import PseudoMeasurementsHandler



def _convert_result_to_ppnet(net: "pp.pandapowerNet", ppci_result: list[int]) -> dict[int, list[int]]:
    """
   Converts results from the bus-branch model (PPCI) to the pandapower network format.

    This function maps each bus index in the PPCI results to its corresponding indices
    in the pandapower network based on the `_pd2ppc_lookups['bus']` mapping.

    Args:
        net (pp.pandapowerNet): The pandapower network object.
        ppci_result (list[int]): A list of bus indices from the PPCI result.

    Returns:
        dict[int, list[int]]: A dictionary where each key is a bus index from `ppci_result`,
                              and the value is a list of corresponding indices in the pandapower network.
    """
    # Extract the bus lookup mapping from the pandapower network
    bus_lookup = net._pd2ppc_lookups['bus']

    # Create a dictionary mapping each bus index in ppci_result to its corresponding indices
    res = {bus: [i for i, lookup_bus in enumerate(bus_lookup) if lookup_bus == bus] for bus in ppci_result}

    return res


def run_measurement_placement_for_eppci(eppci: "ExtendedPPCI", max_iter: int = 150, tolerance: float = 1e-10) -> list:
    """
    Runs the measurement placement algorithm for an ExtendedPPCI model.

    This function uses the PseudoMeasurementsHandler to determine where pseudo-measurements
    should be introduced in a power system model represented in the bus-branch format.
    The result is a list of bus indices in the bus-branch model where pseudo-measurements
    are required.

    Args:
        eppci (ExtendedPPCI): The extended bus-branch model of the power system.
        max_iter (int, optional): The maximum number of iterations for the algorithm.
                                  Default is 150.
        tolerance (float, optional): The convergence tolerance. The algorithm stops if
                                      the change between iterations is below this value.
                                      Default is 1e-10.

    Returns:
        list: A list of bus indices (in the bus-branch model) where pseudo-measurements
              should be introduced.
    """
    # Initialize the pseudo-measurement handler with the provided eppci model
    analyzer = PseudoMeasurementsHandler(eppci)

    # Perform the measurement placement and retrieve the results
    result = analyzer.handle(max_iter=max_iter, tolerance=tolerance)

    # Return the list of bus indices where pseudo-measurements are required
    return result


def run_measurement_placement_for_ppnet(
        net: "pp.pandapowerNet",
        max_iter: int = 150,
        tolerance: float = 1e-10,
        v_start='flat',
        delta_start='flat',
        algorithm='wls',
        calculate_voltage_angles=True,
        zero_injection='auto',
        drop_measurements=False,
) -> dict:
    """
    Runs the measurement placement algorithm for a pandapower network.

    This function converts a pandapower network (ppnet) to an extended power flow data
    structure (eppci), performs measurement placement on it, and maps the result back
    to the pandapower network.

    Args:
        net (pp.pandapowerNet): The pandapower network object.
        max_iter (int, optional): Maximum number of iterations for the measurement
                                  placement algorithm. Default is 150.
        tolerance (float, optional): Convergence tolerance for the algorithm.
                                      Default is 1e-10.
        v_start (list[float], optional): Initial voltage magnitudes for the power flow
                                         calculation. Default is None.
        delta_start (list[float], optional): Initial voltage angles for the power flow
                                             calculation. Default is None.
        calculate_voltage_angles (bool, optional): Whether to calculate voltage angles.
                                                   Default is True.
        zero_injection (list[int], optional): List of zero-injection buses. Default is None.
        algorithm (str, optional): Algorithm to use for power flow calculation.
                                   Default is 'wls'.

    Returns:
        dict: A dictionary where:
            - Key: Bus indices from the pandapower network.
            - Value: A list of buses where pseudo-measurements should be introduced.
                     These are the mapped results from the bus-branch model to the
                     pandapower network.
    """
    # Convert the pandapower network to the extended power flow data structure
    net, _, eppci = pp2eppci(net, v_start=v_start, delta_start=delta_start,
                             calculate_voltage_angles=calculate_voltage_angles,
                             zero_injection=zero_injection, algorithm=algorithm,drop_measurements=drop_measurements
                             )

    # Perform measurement placement on the extended power flow data structure
    result = run_measurement_placement_for_eppci(eppci, max_iter=max_iter, tolerance=tolerance)

    # Map the result back to the pandapower network
    result_with_net_buses = _convert_result_to_ppnet(net=net, ppci_result=result)

    return result_with_net_buses
