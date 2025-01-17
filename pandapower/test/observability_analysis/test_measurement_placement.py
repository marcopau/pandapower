import unittest
from copy import deepcopy

import pandapower as pp
from pandapower.observability_analysis.measurement_placement import run_measurement_placement_for_ppnet


class Test6BusSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create an empty Pandapower network
        net = pp.create_empty_network()

        # Add buses
        buses = {}
        for i in range(1, 7):  # Buses are numbered from 1 to 6
            buses[i] = pp.create_bus(net, vn_kv=110, index=i)  # Assume 110 kV voltage level for simplicity

        # Add lines based on the connections in the diagram
        lines = [
            (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6)
        ]

        for line in lines:
            pp.create_line_from_parameters(
                net, from_bus=buses[line[0]], to_bus=buses[line[1]],
                length_km=1.0,  # Assume 1 km for all lines
                r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10, max_i_ka=1
            )

        # Add external grid connection at bus 6
        pp.create_ext_grid(net, buses[1], vm_pu=1.03)

        cls.net = net

    def test_pseudo_measurements_placement_case6_A1(self):
        net = deepcopy(self.net)

        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=6)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=4)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=1)  # Active power injection
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=4, side="from")  # Line active power

        pseudo_measurements_candidates = run_measurement_placement_for_ppnet(net, zero_injection=None)
        self.assertDictEqual(pseudo_measurements_candidates, {})

    def test_pseudo_measurements_placement_case6_A2(self):
        net = deepcopy(self.net)

        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=4)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=1)  # Active power injection
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=4, side="from")  # Line active power

        pseudo_measurements_candidates = run_measurement_placement_for_ppnet(net, zero_injection=None)
        # Key: 5 (index in ppci), Value: [6] (index in net)
        self.assertDictEqual(pseudo_measurements_candidates, {5: [6]})

        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=6)
        pseudo_measurements_candidates = run_measurement_placement_for_ppnet(net, zero_injection=None)
        self.assertDictEqual(pseudo_measurements_candidates, {})


class Test14BusSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create an empty Pandapower network
        net = pp.create_empty_network()

        # Add buses
        buses = {}
        for i in range(1, 15):  # Buses are numbered from 1 to 14
            buses[i] = pp.create_bus(net, vn_kv=110, index=i)  # Assume 110 kV voltage level for simplicity

        # Add lines based on the connections in the diagram
        lines = [
            (1, 5), (1, 2), (2, 5), (2, 4), (2, 3), (3, 4), (5, 4), (5, 6), (4, 7), (4, 9),
            (7, 8), (7, 9),
            (9, 14), (9, 11), (11, 10), (6, 10), (6, 12), (6, 13), (12, 13), (13, 14),
        ]

        for line in lines:
            pp.create_line_from_parameters(
                net, from_bus=buses[line[0]], to_bus=buses[line[1]],
                length_km=1.0,  # Assume 1 km for all lines
                r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10, max_i_ka=1
            )

        # Add example load at a bus
        pp.create_load(net, buses[8], p_mw=5, q_mvar=2)

        # Add example generator at a bus
        pp.create_gen(net, buses[1], p_mw=10, vm_pu=1.02)

        # Add external grid connection at bus 6
        pp.create_ext_grid(net, buses[6], vm_pu=1.03)

        cls.net = net

    def test_pseudo_measurements_placement_case14_A1(self):
        net = deepcopy(self.net)
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=13)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=12)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=11)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=9)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=6)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=5)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=4)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=3)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=2)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=1)  # Active power injection

        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=8, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=10, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=11, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=9, side="from")  # Line active power

        pseudo_measurements_candidates = run_measurement_placement_for_ppnet(net, zero_injection=None)
        self.assertDictEqual(pseudo_measurements_candidates, {})

    def test_pseudo_measurements_placement_case14_A2(self):
        net = deepcopy(self.net)
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=13)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=12)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=11)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=9)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=6)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=4)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=3)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=2)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=1)  # Active power injection

        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=8, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=10, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=11, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=9, side="from")  # Line active power

        pseudo_measurements_candidates = run_measurement_placement_for_ppnet(net, zero_injection=None)
        # Key: 13 (index in ppci), Value: [14] (index in net)
        self.assertDictEqual(pseudo_measurements_candidates, {13: [14]})

        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=14)
        pseudo_measurements_candidates = run_measurement_placement_for_ppnet(net, zero_injection=None)
        self.assertDictEqual(pseudo_measurements_candidates, {})


class TestMonticelliBusSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create an empty Pandapower network
        net = pp.create_empty_network()

        # Add buses
        buses = {}
        for i in range(1, 9):  # Buses are numbered from 1 to 8
            buses[i] = pp.create_bus(net, vn_kv=110, index=i)  # Assume 110 kV voltage level for simplicity

        # Add lines based on the connections in the diagram
        lines = [
            (1, 3), (2, 4), (3, 7), (3, 5), (4, 7), (4, 6), (6, 8), (5, 8), (7, 8)
        ]

        for line in lines:
            pp.create_line_from_parameters(
                net, from_bus=buses[line[0]], to_bus=buses[line[1]],
                length_km=1.0,  # Assume 1 km for all lines
                r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10, max_i_ka=1
            )

        # Add external grid connection at bus 6
        pp.create_ext_grid(net, buses[1], vm_pu=1.03)

        cls.net = net

    def test_pseudo_measurements_placement(self):
        net = deepcopy(self.net)
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=3)  # Active power injection

        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=0, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=5, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=3, side="from")  # Line active power

        pseudo_measurements_candidates = run_measurement_placement_for_ppnet(net, zero_injection=None)
        # Explanation of the result:
        # Key: 6 (index in ppci), Value: [7] (index in net)
        # Key: 7 (index in ppci), Value: [8] (index in net)
        self.assertDictEqual(pseudo_measurements_candidates, {6: [7], 7: [8]})

        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=7)
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=8)
        pseudo_measurements_candidates = run_measurement_placement_for_ppnet(net, zero_injection=None)
        self.assertDictEqual(pseudo_measurements_candidates, {})


class TestAburBusSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create an empty Pandapower network
        net = pp.create_empty_network()

        # Add buses
        buses = {}
        for i in range(1, 7):  # Buses are numbered from 1 to 6
            buses[i] = pp.create_bus(net, vn_kv=110, index=i)  # Assume 110 kV voltage level for simplicity

        # Add lines based on the connections in the diagram
        lines = [
            (1, 2), (1, 6), (2, 3), (2, 6), (2, 5), (3, 4), (4, 5), (5, 6)
        ]

        for line in lines:
            pp.create_line_from_parameters(
                net, from_bus=buses[line[0]], to_bus=buses[line[1]],
                length_km=1.0,  # Assume 1 km for all lines
                r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10, max_i_ka=1
            )

        # Add external grid connection at bus 6
        pp.create_ext_grid(net, buses[1], vm_pu=1.03)

        cls.net = net

    def test_pseudo_measurements_placement(self):

        net = deepcopy(self.net)
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=3)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=6)  # Active power injection

        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=5, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=2, side="from")  # Line active power

        pseudo_measurements_candidates = run_measurement_placement_for_ppnet(net, zero_injection=None)
        # Explanation of the result:
        # Key: 4 (index in ppci), Value: [5] (index in net)
        self.assertDictEqual(pseudo_measurements_candidates, {4: [5]})

        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=5)
        pseudo_measurements_candidates = run_measurement_placement_for_ppnet(net, zero_injection=None)
        self.assertDictEqual(pseudo_measurements_candidates, {})
