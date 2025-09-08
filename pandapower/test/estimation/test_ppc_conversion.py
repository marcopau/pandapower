import numpy as np
import pandapower as pp
from pandapower.estimation.idx_brch import P_TO, P_TO_STD
from pandapower.estimation.idx_bus import P, Q
from pandapower.estimation.ppc_conversion import pp2eppci
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols
import pytest


def test_duplicate_measurements_at_trafo3w():
    # Create an empty network
    net = pp.create_empty_network()

    # Create buses for two three-winding transformers (HV, MV, LV each)
    hv_bus1 = pp.create_bus(net, vn_kv=110, name="HV Bus 1")
    mv_bus1 = pp.create_bus(net, vn_kv=20, name="MV Bus 1")
    lv_bus1 = pp.create_bus(net, vn_kv=10, name="LV Bus 1")

    hv_bus2 = pp.create_bus(net, vn_kv=110, name="HV Bus 2")
    mv_bus2 = pp.create_bus(net, vn_kv=20, name="MV Bus 2")
    lv_bus2 = pp.create_bus(net, vn_kv=10, name="LV Bus 2")

    # Create external grids for slack reference at HV sides
    pp.create_ext_grid(net, bus=hv_bus1, vm_pu=1.02, name="Grid Slack 1")
    pp.create_ext_grid(net, bus=hv_bus2, vm_pu=1.02, name="Grid Slack 2")

    # Create two three-winding transformers using standard types
    trafo1 = pp.create_transformer3w(net,
                                     hv_bus=hv_bus1,
                                     mv_bus=mv_bus1,
                                     lv_bus=lv_bus1,
                                     std_type="63/25/38 MVA 110/20/10 kV",
                                     name="3W Trafo 1")

    trafo2 = pp.create_transformer3w(net,
                                     hv_bus=hv_bus2,
                                     mv_bus=mv_bus2,
                                     lv_bus=lv_bus2,
                                     std_type="63/25/38 MVA 110/20/10 kV",
                                     name="3W Trafo 2")

    # Add measurements: measure active (p) and reactive (q) power on each side of each transformer
    for trafo in [trafo1, trafo2]:
        for side in ["hv", "mv"]:
            pp.create_measurement(net,
                                  meas_type="p",
                                  element_type="trafo3w",
                                  element=trafo,
                                  side=side,
                                  value=2.0,
                                  std_dev=1,
                                  name=f"P_{side.upper()}_Trafo{trafo}")
            pp.create_measurement(net,
                                  meas_type="p",
                                  element_type="trafo3w",
                                  element=trafo,
                                  side=side,
                                  value=4.0,
                                  std_dev=1,
                                  name=f"P2_{side.upper()}_Trafo{trafo}")

    _, _, eppci = pp2eppci(net, v_start="flat", delta_start="flat", zero_injection="aux_bus")
    vals = eppci.data["branch"][[2, 3], branch_cols + P_TO]
    np.testing.assert_array_equal(vals, [3, 3])

    std_dev = eppci.data["branch"][[2, 3], branch_cols + P_TO_STD]
    # 0.707107 =  ((1^2 + 1^2)^0.5)/2
    np.testing.assert_array_almost_equal(std_dev, [0.707107, 0.707107], decimal=6)


def create_grid():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, name="bus1", vn_kv=10.)
    b2 = pp.create_bus(net, name="bus2", vn_kv=10.)
    b3 = pp.create_bus(net, name="bus3", vn_kv=10.)
    b4 = pp.create_bus(net, name="bus4", vn_kv=10.)
    b5 = pp.create_bus(net, name="bus5", vn_kv=10.)
    b6 = pp.create_bus(net, name="bus6", vn_kv=10.)

    pp.create_ext_grid(net, b1)
    l1 = pp.create_line_from_parameters(net, b1, b2, 10, r_ohm_per_km=.59, x_ohm_per_km=.35, c_nf_per_km=10.1,
                                max_i_ka=1)
    l2 = pp.create_line_from_parameters(net, b5, b6, 10, r_ohm_per_km=.59, x_ohm_per_km=.35, c_nf_per_km=10.1,
                                max_i_ka=1)
    pp.create_switch(net, b2, element=b3, et='b')
    pp.create_switch(net, b3, element=b4, et='b')
    pp.create_switch(net, b4, element=b5, et='b')

    pp.create_load(net, b2, p_mw=.350, q_mvar=.100)
    pp.create_load(net, b3, p_mw=.120, q_mvar=.050)
    pp.create_load(net, b4, p_mw=.100, q_mvar=.025)
    pp.create_load(net, b6, p_mw=.350, q_mvar=.100)

    pp.runpp(net)

    pp.create_measurement(net, "v", "bus", net.res_bus.vm_pu.iloc[b1], .002, element=b1)
    pp.create_measurement(net, "v", "bus", net.res_bus.vm_pu.iloc[b3], .002, element=b3)

    pp.create_measurement(net, "p", "bus", net.res_bus.p_mw.iloc[b2], .002, element=b2)
    pp.create_measurement(net, "q", "bus", net.res_bus.q_mvar.iloc[b2], .002, element=b2)
    pp.create_measurement(net, "p", "bus", net.res_bus.p_mw.iloc[b3], .002, element=b3)
    pp.create_measurement(net, "q", "bus", net.res_bus.q_mvar.iloc[b3], .002, element=b3)
    pp.create_measurement(net, "p", "bus", net.res_bus.p_mw.iloc[b6], .002, element=b6)
    pp.create_measurement(net, "q", "bus", net.res_bus.q_mvar.iloc[b2], .002, element=b6)

    pp.create_measurement(net, "p", "line", net.res_line.p_from_mw.iloc[l1], .002, element=l1, side="from")
    pp.create_measurement(net, "q", "line", net.res_line.q_from_mvar.iloc[l1], .002, element=l1, side="from")

    return net

def test_pwr_inj_at_merged_buses_without_all_measurements():
    net = create_grid()
        
    _, _, eppci = pp2eppci(net, v_start="flat", delta_start="flat", zero_injection="no_inj_bus")
    P_val = eppci.data["bus"][1, bus_cols + P]
    Q_val = eppci.data["bus"][1, bus_cols + Q]

    np.testing.assert_array_equal([P_val, Q_val], [np.nan, np.nan])


def test_pwr_inj_at_merged_buses_with_all_measurements_but_no_zero_inj_considered():
    net = create_grid()
    pp.create_measurement(net, "p", "bus", net.res_bus.p_mw.iloc[3], .002, element=3)
    pp.create_measurement(net, "q", "bus", net.res_bus.q_mvar.iloc[3], .002, element=3)
        
    _, _, eppci = pp2eppci(net, v_start="flat", delta_start="flat", zero_injection="aux_bus")
    P_val = eppci.data["bus"][1, bus_cols + P]
    Q_val = eppci.data["bus"][1, bus_cols + Q]

    np.testing.assert_array_equal([P_val, Q_val], [np.nan, np.nan])


def test_pwr_inj_at_merged_buses_with_all_measurements_and_zero_inj_considered():
    net = create_grid()
    pp.create_measurement(net, "p", "bus", net.res_bus.p_mw.iloc[3], .002, element=3)
    pp.create_measurement(net, "q", "bus", net.res_bus.q_mvar.iloc[3], .002, element=3)
        
    _, _, eppci = pp2eppci(net, v_start="flat", delta_start="flat", zero_injection="no_inj_bus")
    P_val = eppci.data["bus"][1, bus_cols + P]
    Q_val = eppci.data["bus"][1, bus_cols + Q]

    np.testing.assert_array_equal([P_val, Q_val], [-0.57, -0.175])


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])