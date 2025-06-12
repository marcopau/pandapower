# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from typing import Dict

import numpy as np
import pandas as pd

from pandapower.auxiliary import pandapowerNet
from pandapower.estimation.idx_brch import (P_FROM, P_FROM_IDX, P_FROM_STD,
                                            Q_FROM, Q_FROM_IDX, Q_FROM_STD,
                                            IM_FROM, IM_FROM_IDX, IM_FROM_STD,
                                            IA_FROM, IA_FROM_IDX, IA_FROM_STD,
                                            P_TO, P_TO_IDX, P_TO_STD,
                                            Q_TO, Q_TO_IDX, Q_TO_STD,
                                            IM_TO, IM_TO_IDX, IM_TO_STD,
                                            IA_TO, IA_TO_IDX, IA_TO_STD)
from pandapower.estimation.ppc_conversion.utils import _calculate_weighted_measurements

# Constant Lookup
BR_SIDE = {"line": {"f": "from", "t": "to"},
           "trafo": {"f": "hv", "t": "lv"}}
BR_MEAS_PPCI_IX = {("p", "f"): {"VALUE": P_FROM, "IDX": P_FROM_IDX, "STD": P_FROM_STD},
                   ("q", "f"): {"VALUE": Q_FROM, "IDX": Q_FROM_IDX, "STD": Q_FROM_STD},
                   ("i", "f"): {"VALUE": IM_FROM, "IDX": IM_FROM_IDX, "STD": IM_FROM_STD},
                   ("ia", "f"): {"VALUE": IA_FROM, "IDX": IA_FROM_IDX, "STD": IA_FROM_STD},
                   ("p", "t"): {"VALUE": P_TO, "IDX": P_TO_IDX, "STD": P_TO_STD},
                   ("q", "t"): {"VALUE": Q_TO, "IDX": Q_TO_IDX, "STD": Q_TO_STD},
                   ("i", "t"): {"VALUE": IM_TO, "IDX": IM_TO_IDX, "STD": IM_TO_STD},
                   ("ia", "t"): {"VALUE": IA_TO, "IDX": IA_TO_IDX, "STD": IA_TO_STD}}


def _add_measurements_to_branch(
        branch_append: np.ndarray,
        meas: pd.DataFrame,
        element_name: str,
        side_map: Dict[str, str],
        map_branch: pd.Series,
) -> None:
    """
        Appends weighted branch measurements (power, current, etc.) to the branch_append array.

        Parameters:
        - branch_append: NumPy array (ppci branch matrix) to append measurement values to.
        - meas: DataFrame of measurements.
        - element_name: Name of the element type, e.g., 'line', 'trafo', 'trafo3w'.
        - side_map: Mapping of measurement side values to branch side strings (e.g., {'hv': 'from', 'lv': 'to'}).
        - map_branch: pd.Series mapping element indices to PPCI branch indices.
    """

    for meas_type in ('p', 'q', 'i'):
        filtered = meas[
            (meas.measurement_type == meas_type)
            & (meas.element_type == element_name)
            & meas.element.isin(map_branch.index)
            ]
        if filtered.empty:
            continue

        for side_val, br_side in side_map.items():
            side_df = filtered[filtered.side == side_val].copy()
            if side_df.empty:
                continue

            # Remap element index → PPCI branch index
            side_df['element'] = side_df['element'].map(map_branch)

            # Create index map from PPCI index to original measurement index (for later reference)
            idx_map = (
                side_df[['element']]
                .drop_duplicates(subset='element', keep='first')
                .reset_index()[['element', 'index']]
                .set_index('element')['index']
            )

            # Calculate weighted measurement and std dev by branch index
            merged = _calculate_weighted_measurements(side_df, 'element')

            # Get column indices for VALUE, STD, IDX in branch_append array
            specs = BR_MEAS_PPCI_IX[(meas_type, br_side)]

            branch_append[merged.index, specs['VALUE']] = merged.weighted_measurement
            branch_append[merged.index, specs['STD']] = merged.merged_weight
            branch_append[merged.index, specs['IDX']] = merged.index.map(idx_map)


def _get_branch_map(
        net: pandapowerNet,
        br_is_mask: np.ndarray,
        element_type: str,
) -> pd.Series:
    """
    Builds a Series mapping element indices (e.g., line, trafo) to their corresponding PPCI branch indices.

    Parameters:
    - net: pandapower network object.
    - br_is_mask: Boolean array (ppci['internal']['branch_is']) indicating which PPCI rows are active.
    - element_type: Type of branch element (e.g., 'line', 'trafo', 'trafo3w').

    Returns:
    - map_branch: pd.Series with element indices (from net.<element_type>.index) as index
                  and PPCI branch indices as values.
    """
    # Get the start and end positions in the PPCI branch array for this element type
    start, end = net._pd2ppc_lookups['branch'][element_type]

    # Extract mask for just the entries related to the current element type
    mask = br_is_mask[start:end]

    # Compute PPCI offset: number of active branches before the current element type
    offset = np.sum(br_is_mask[:start])

    # Get the original element indices (e.g., net.line.index) filtered by active mask
    element_indices = getattr(net, element_type).index.values[mask]

    # Compute PPCI indices corresponding to these elements
    ppc_indices = np.arange(offset, offset + mask.sum())

    # Return mapping from element index to PPCI branch index
    return pd.Series(data=ppc_indices, index=element_indices)


def _add_measurements_to_line(
        net: pandapowerNet,
        branch_append: np.ndarray,
        meas: pd.DataFrame,
        br_is_mask: np.ndarray
) -> None:
    """
        Adds line-related measurements (power, current, etc.) to the PPCI branch array.

        Parameters:
        - net: pandapower network object.
        - branch_append: NumPy array representing the PPCI branch matrix where measurements are stored.
        - meas: DataFrame of measurements.
        - br_is_mask:  Boolean array (ppci['internal']['branch_is']) indicating the active branches in ppci['branch'].
    """
    if net.line.empty:
        return

    map_branch = _get_branch_map(net, br_is_mask, "line")

    _add_measurements_to_branch(
        branch_append=branch_append,
        meas=meas,
        element_name="line",
        side_map={"from": "f", "to": "t"},
        map_branch=map_branch
    )


def _add_measurements_to_trafo(
        net: pandapowerNet,
        branch_append: np.ndarray,
        meas: pd.DataFrame,
        br_is_mask: np.ndarray
) -> None:
    """
        Adds transformer (2-winding) related measurements to the PPCI branch matrix.

        Parameters:
        - net: pandapower network object.
        - branch_append: NumPy array representing the PPCI branch matrix where measurements will be written.
        - meas: DataFrame containing measurements.
        - br_is_mask: Boolean array (ppci['internal']['branch_is']) marking active rows in the PPCI branch matrix.
    """

    if net.trafo.empty:
        return

    map_branch = _get_branch_map(net, br_is_mask, "trafo")

    _add_measurements_to_branch(
        branch_append=branch_append,
        meas=meas,
        element_name="trafo",
        side_map={"hv": "f", "lv": "t"},
        map_branch=map_branch,
    )


def _add_measurements_to_trafo3w(
        net: pandapowerNet,
        branch_append: np.ndarray,
        meas: pd.DataFrame,
        br_is_mask: np.ndarray
) -> None:
    """
    Adds measurements for 3-winding transformers (HV, MV, LV sides) to the PPCI branch matrix.

    Parameters:
    - net: pandapower network object.
    - branch_append: NumPy array representing the PPCI branch matrix where measurements will be stored.
    - meas: DataFrame of measurements.
    - br_is_mask: Boolean array (ppci['internal']['branch_is']) marking active rows in the PPCI branch matrix.
    """
    if net.trafo3w.empty:
        return

    # Retrieve starting index for trafo3w entries in the PPCI branch matrix
    trafo3w_ix_start = net["_pd2ppc_lookups"]["branch"]["trafo3w"][0]
    num_trafo3w = len(net.trafo3w)

    # Get mask for HV-side branches (used as base reference for all three sides)
    trafo3w_is_hv = br_is_mask[trafo3w_ix_start: trafo3w_ix_start + num_trafo3w]
    num_active = trafo3w_is_hv.sum()

    # Compute PPCI offset: number of active branches before trafo3w section
    offset = np.count_nonzero(br_is_mask[:trafo3w_ix_start])

    # Indices of active 3-winding trafos
    indices = net.trafo3w.index[trafo3w_is_hv]
    ppc_indices = np.arange(offset, offset + num_active)

    # Build mapping per side: HV, MV, LV
    map_branch_by_side = {
        "hv": pd.Series(data=ppc_indices, index=indices),
        "mv": pd.Series(data=ppc_indices + num_active, index=indices),
        "lv": pd.Series(data=ppc_indices + 2 * num_active, index=indices)
    }

    # Define PPCI side labels for each trafo3w side
    side_map_labels = {"hv": "f", "mv": "t", "lv": "t"}  # HV is 'from', MV and LV are 'to'

    # Process and store measurements per side
    for side, map_branch in map_branch_by_side.items():
        _add_measurements_to_branch(
            branch_append=branch_append,
            meas=meas,
            element_name="trafo3w",
            side_map={side: side_map_labels[side]},
            map_branch=map_branch
        )
