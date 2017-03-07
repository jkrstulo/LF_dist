# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import warnings

from pypower.ppoption import ppoption
from pypower.idx_bus import VM

from pandapower.pypower_extensions.runpf import _runpf
from pandapower.auxiliary import ppException, _select_is_elements, _clean_up
from pandapower.pd2ppc import _pd2ppc, _update_ppc
from pandapower.pypower_extensions.opf import opf
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc, reset_results, \
    _extract_results_opf

from directDistPF import _run_fbsw

class LoadflowNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass


class OPFNotConverged(ppException):
    """
    Exception being raised in case optimal powerflow did not converge.
    """
    pass


def runpp_dd(net, init="flat", calculate_voltage_angles=False, tolerance_kva=1e-5, trafo_model="t",
             trafo_loading="current", enforce_q_lims=False, numba=True, recycle=None, sparse=True, **kwargs):
    """
    modification of pandapower's runpp in order to run Direct Distribution PF run_ddpf
    """
    ac = True
    # recycle parameters
    if recycle == None:
        recycle = dict(is_elems=False, ppc=False, Ybus=False)

    _runpppf_dd(net, init, ac, calculate_voltage_angles, tolerance_kva, trafo_model,
             trafo_loading, enforce_q_lims, numba, recycle, sparse, **kwargs)



def _runpppf_dd(net, init, ac, calculate_voltage_angles, tolerance_kva, trafo_model,
             trafo_loading, enforce_q_lims, numba, recycle, sparse, **kwargs):
    """
    Gets called by runpp or rundcpp with different arguments.
    """

    net["converged"] = False
    if (ac and not init == "results") or not ac:
        reset_results(net)

    # select elements in service (time consuming, so we do it once)
    is_elems = _select_is_elements(net, recycle)

    if recycle["ppc"] and "_ppc" in net and net["_ppc"] is not None and "_bus_lookup" in net:
        # update the ppc from last cycle
        ppc, ppci, bus_lookup = _update_ppc(net, is_elems, recycle, calculate_voltage_angles, enforce_q_lims,
                                            trafo_model)
    else:
        # convert pandapower net to ppc
        ppc, ppci, bus_lookup = _pd2ppc(net, is_elems, calculate_voltage_angles, enforce_q_lims,
                                        trafo_model, init_results=(init == "results"))

    # store variables
    net["_ppc"] = ppc
    net["_bus_lookup"] = bus_lookup
    net["_is_elems"] = is_elems

    if not "VERBOSE" in kwargs:
        kwargs["VERBOSE"] = 0

    # run the powerflow
    result = _run_fbsw(ppci, sparse, ppopt=ppoption(ENFORCE_Q_LIMS=enforce_q_lims,
                                            PF_TOL=tolerance_kva * 1e-3, **kwargs))[0]

    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    result = _copy_results_ppci_to_ppc(result, ppc, bus_lookup)

    # raise if PF was not successful. If DC -> success is always 1
    if result["success"] != 1:
        raise LoadflowNotConverged("Loadflow did not converge!")
    else:
        net["_ppc"] = result
        net["converged"] = True

    _extract_results(net, result, is_elems, bus_lookup, trafo_loading, ac)
    _clean_up(net)

