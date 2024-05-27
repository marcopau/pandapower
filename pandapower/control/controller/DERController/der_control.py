# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np

from pandapower.auxiliary import ensure_iterability
from pandapower.control.controller.pq_control import PQController
from pandapower.control.controller.DERController.QModels import QModel
from pandapower.control.controller.DERController.PQVAreas import BaseArea, PQVArea4110, QVArea4110

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
""" DERController """
# -------------------------------------------------------------------------------------------------


class DERController(PQController):
    """
    Flexible controller to model plenty types of distributed energy resource (DER) control
    characteristics, such as

    + const Q
    + cosphi fixed (different types)
    + cosphi(P) curve
    + Q(V)

    and restrict the behavior to defined PQV areas, such as

    + PQVArea4130V1, PQVArea4130V2, PQVArea4130V3
    + PQVArea4120V1, PQVArea4120V2, PQVArea4120V3
    + PQVArea4110
    + PQVArea4105
    + PQAreaSTATCOM
    + PQVAreaPOLYGON (PQAreaPOLYGON, QVAreaPOLYGON)

    .. tip: For the DER controller, an extensive `tutorial <https://github.com/e2nIEE/pandapower/tree/develop/tutorials>`_ is available.

    .. note:: sn_mva of the controlled elements is expected to be the rated power (generation) of
        the elements (called P_{b,installed} in the VDE AR N standards). Scalings and limits are
        usually relative to that (sn_mva) values.

    INPUT:
        **net** (pandapower net)

        **gid** (int[]) - IDs of the controlled elements

    OPTIONAL:
        **element** (str, "sgen") - element type which is controlled

        **q_model"" (object, None) - an q_model, such as provided in this file, should be passed to
        model how the q value should be determined.

        **pqv_area** (object, None) - an pqv_area, such as provided in this file, should be passed
        to model q values are allowed.

        **saturate_sn_mva** (float, NaN) - Maximum apparent power of the inverter. If given, the
        p or q values (depending on q_prio) are reduced to this maximum apparent power. Usually,
        it is not necessary to pass this values since the inverter needs to be dimensioned to provide
        the standardized reactive power requirements.

        **q_prio** (bool, True) - If True, the active power is reduced first in case of power
        reduction due to saturate_sn_mva. Otherwise, the reactive power is reduced first.

        **damping_coef** (float, 2) - damping coefficient to influence the power updating process
        of the control loop. A higher value mean slower changes of p and q towards the latest target
        values

        **max_p_error** (float, 0.0001) - Maximum error of active power

        **max_q_error** (float, 0.0001) - Maximum error of reactive power

        **p_ac** (float, 1.0) - Simultaneity factor applied to P and Q

        **f_sizing** (float, 1.0) - Sizing of the converter factor limiting P

        **data_source** ( , None) - A DataSource that contains profiles

        **p_profile** (str[], None) - The profile names of the controlled elements in the data
        source for active power time series values

        **profile_from_name** (bool, False) - If True, the profile names of the controlled elements
        in the data source for active power time series values will be set be the name of the
        controlled elements, e.g. for controlled sgen "SGEN_1", the active power profile "P_SGEN_1"
        is applied

        **profile_scale** (float, 1.0) - A scaling factor applied to the values of profiles

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **ts_absolute** (bool, True) - Whether the time step values are absolute power values or
        scaling factors
    """
    def __init__(self, net, gid, element="sgen",
                 q_model=None, pqv_area=None,
                 saturate_sn_mva=np.nan, q_prio=True, damping_coef=2,
                 max_p_error=1e-6, max_q_error=1e-6, p_ac=1., f_sizing=1.,
                 data_source=None, p_profile=None, profile_from_name=False,
                 profile_scale=1.0, in_service=True, ts_absolute=True,
                 order=0, level=0, drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        gid = list(ensure_iterability(gid))
        if matching_params is None:
            matching_params = {"gid": gid}
        super().__init__(net, gid=gid, element=element, max_p_error=max_p_error,
                         max_q_error=max_q_error, p_ac=p_ac,
                         f_sizing=f_sizing, data_source=data_source,
                         profile_scale=profile_scale, in_service=in_service,
                         ts_absolute=ts_absolute, initial_run=True,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_powerflow=False,
                         order=order, level=level, **kwargs)

        # --- init DER Model params
        self.q_model = q_model
        self.pqv_area = pqv_area
        self.saturate_sn_mva = saturate_sn_mva
        self.q_prio = q_prio
        self.damping_coef = damping_coef

        if p_profile is not None:
            p_profile = ensure_iterability(p_profile, len(gid))
        self.set_p_profile(p_profile, profile_from_name)

        # --- log unexpected param values
        if n_nan_sn := sum(self.sn_mva.isnull()):
            logger.error(f"The DERController relates to sn_mva, but for {n_nan_sn} elements "
                         "sn_mva is NaN.")
        if self.saturate_sn_mva <= 0:
            raise ValueError(f"saturate_sn_mva cannot be <= 0 but is {self.saturate_sn_mva}")
        if self.q_model is not None and not isinstance(self.q_model, QModel):
            logger.warning(f"The Q model is expected of type QModel, however {type(self.q_model)} "
                           "is provided.")
        if self.pqv_area is not None and not isinstance(self.pqv_area, BaseArea):
            logger.warning(f"The PQV area is expected of type BaseArea, however "
                           f"{type(self.pqv_area)} is provided.")

    def time_step(self, net, time):
        # get new values from profiles
        self.read_profiles(time)
        self.p_series_mw = self.p_mw
        self.q_series_mvar = self.q_mvar

#        self.write_to_net(net)

    def is_converged(self, net):
        vm = net.res_bus.loc[self.bus, "vm_pu"].set_axis(self.gid)
        p_series_mw = getattr(self, "p_series_mw", getattr(self, "p_mw", self.sn_mva))
        q_series_mvar = getattr(self, "q_series_mw", self.q_mvar)

        # --- calculate target p and q -------------------------------------------------------------

        if np.any(p_series_mw < 0):
            logger.info("p_series_mw is forced to be greater/equal zero")
            p_series_mw[p_series_mw < 0] = 0.

        # --- First Step: Calculate/Select P, Q
        p = self._step_p(p_series_mw)
        q = self._step_q(p_series_mw=p_series_mw, q_series_mvar=q_series_mvar, vm=vm)

        # --- Second Step: Saturates P, Q according to SnMVA and PQV_AREA
        if self.saturate_sn_mva or (self.pqv_area is not None):
            p, q = self._saturate(p, q, vm)

        # --- Third Step: Convert relative P, Q to p_mw, q_mvar
        target_p_mw, target_q_mvar = p * self.sn_mva, q * self.sn_mva

        # --- Apply target p and q considering the damping factor coefficient ----------------------
        self.target_p_mw = self.p_mw + (target_p_mw - self.p_mw) / self.damping_coef
        self.target_q_mvar = self.q_mvar + (target_q_mvar - self.q_mvar) / self.damping_coef

        return np.allclose(self.target_q_mvar, self.q_mvar, atol=self.max_q_error) and\
            np.allclose(self.target_p_mw, self.p_mw, atol=self.max_p_error)

    def control_step(self, net):
        self.p_mw, self.q_mvar = self.target_p_mw, self.target_q_mvar

        self.write_to_net(net)

    def _step_p(self, p_series_mw=None, p_setpoint_mw=None):
        return p_series_mw / self.sn_mva

    def _step_q(self, p_series_mw=None, q_series_mvar=None, vm=None):
        """Q priority: Q setpoint > Q model > Q series"""
        if self.q_model is not None:
            q = self.q_model.step(vm=vm, p=p_series_mw/self.sn_mva)
        else:
            if q_series_mvar is None:
                raise Exception("No Q_model and no q_profile available.")
            q = q_series_mvar / self.sn_mva
        return q

    def _saturate(self, p, q, vm):
        assert p is not None and q is not None

        # Saturation on given pqv_area
        if self.pqv_area is not None:
            in_area = self.pqv_area.in_area(p, q, vm)
            if not all(in_area):
                min_max_q = self.pqv_area.q_flexibility(p=p[~in_area], vm=vm[~in_area])
                q[~in_area] = np.minimum(np.maximum(q[~in_area], min_max_q[:, 0]), min_max_q[:, 1])

        if not np.isnan(self.saturate_sn_mva):
            p, q = self._saturate_sn_mva_step(p, q, vm)
        return p, q

    def _saturate_sn_mva_step(self, p, q, vm):
        # Saturation on SnMVA according to priority mode
        sat_s = self.saturate_sn_mva / self.sn_mva # sat_s is relative to sn_mva
        to_saturate = p**2 + q**2 > sat_s**2
        if any(to_saturate):
            if self.q_prio:
                if (
                    isinstance(self.pqv_area, PQVArea4110) or isinstance(self.pqv_area, QVArea4110)
                   ) and any(
                    (0.95 < vm[to_saturate]) & (vm[to_saturate] < 1.05) &
                    (-0.328684 < q[to_saturate]) & any(q[to_saturate] < 0.328684)
                   ):
                    logger.warning(f"Such kind of saturation is performed that is not in line with"
                                   " VDE AR N 4110: p reduction within 0.95 < vm < 1.05 and "
                                   "0.95 < cosphi.")
                q[to_saturate] = np.clip(q[to_saturate], -sat_s[to_saturate], sat_s[to_saturate])
                p[to_saturate] = np.sqrt(sat_s[to_saturate]**2 - q[to_saturate]**2)
            else:
                p[to_saturate] = np.clip(p[to_saturate], 0., sat_s[to_saturate])
                q[to_saturate] = np.sqrt(sat_s[to_saturate]**2 - p[to_saturate]**2) * np.sign(
                    q[to_saturate])
        return p, q


    def __str__(self):
        return super().__str__() +\
            "q_model:" + str(self.q_model) +\
            ", pqv_area:" + str(self.pqv_area) +\
            ", saturate_sn_mva:" + str(self.saturate_sn_mva) +\
            ", q_priority:" + str(self.q_prio)


if __name__ == "__main__":
    pass
