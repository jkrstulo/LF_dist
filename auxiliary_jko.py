# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pandas as pd
import numpy as np
from collections import MutableMapping
import six

import scipy as sp
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import BUS_I, BUS_TYPE, NONE, PD, QD

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)


class ADict(dict, MutableMapping):
    def __init__(self, *args, **kwargs):
        super(ADict, self).__init__(*args, **kwargs)

        # to prevent overwrite of internal attributes by new keys
        # see _valid_name()
        self._setattr('_allow_invalid_attributes', False)

    def _build(self, obj, **kwargs):
        """
        We only want dict like elements to be treated as recursive AttrDicts.
        """
        return obj

    # --- taken from AttrDict

    def __getstate__(self):
        return self.copy(), self._allow_invalid_attributes

    def __setstate__(self, state):
        mapping, allow_invalid_attributes = state
        self.update(mapping)
        self._setattr('_allow_invalid_attributes', allow_invalid_attributes)

    @classmethod
    def _constructor(cls, mapping):
        return cls(mapping)

    # --- taken from MutableAttr

    def _setattr(self, key, value):
        """
        Add an attribute to the object, without attempting to add it as
        a key to the mapping (i.e. internals)
        """
        super(MutableMapping, self).__setattr__(key, value)

    def __setattr__(self, key, value):
        """
        Add an attribute.
        key: The name of the attribute
        value: The attributes contents
        """
        if self._valid_name(key):
            self[key] = value
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__setattr__(key, value)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute creation.".format(
                    cls=self.__class__.__name__
                )
            )

    def _delattr(self, key):
        """
        Delete an attribute from the object, without attempting to
        remove it from the mapping (i.e. internals)
        """
        super(MutableMapping, self).__delattr__(key)

    def __delattr__(self, key, force=False):
        """
        Delete an attribute.
        key: The name of the attribute
        """
        if self._valid_name(key):
            del self[key]
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__delattr__(key)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute deletion.".format(
                    cls=self.__class__.__name__
                )
            )

    def __call__(self, key):
        """
        Dynamically access a key-value pair.
        key: A key associated with a value in the mapping.
        This differs from __getitem__, because it returns a new instance
        of an Attr (if the value is a Mapping object).
        """
        if key not in self:
            raise AttributeError(
                "'{cls} instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    def __getattr__(self, key):
        """
        Access an item as an attribute.
        """
        if key not in self or not self._valid_name(key):
            raise AttributeError(
                "'{cls}' instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    @classmethod
    def _valid_name(cls, key):
        """
        Check whether a key is a valid attribute name.
        A key may be used as an attribute if:
         * It is a string
         * The key doesn't overlap with any class attributes (for Attr,
            those would be 'get', 'items', 'keys', 'values', 'mro', and
            'register').
        """
        return (
            isinstance(key, six.string_types) and
            not hasattr(cls, key)
        )


class PandapowerNet(ADict):
    def __init__(self, *args, **kwargs):
        super(PandapowerNet, self).__init__(*args, **kwargs)

    def __repr__(self):
        r = "This pandapower network includes the following parameter tables:"
        par = []
        res = []
        for tb in list(self.keys()):
            if isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
                if 'res_' in tb:
                    res.append(tb)
                else:
                    par.append(tb)
        for tb in par:
            r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        if res:
            r += "\n and the following results tables:"
            for tb in res:
                r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        return r


def _preserve_dtypes(df, dtypes):
    for item, dtype in list(dtypes.iteritems()):
        if df.dtypes.at[item] != dtype:
            try:
                df[item] = df[item].astype(dtype)
            except ValueError:
                df[item] = df[item].astype(float)


def get_free_id(df):
    """
    Returns next free ID in a dataframe
    """
    return np.int64(0) if len(df) == 0 else df.index.values.max() + 1


class ppException(Exception):
    """
    General pandapower custom parent exception.
    """
    pass


def _sum_by_group(bus, first_val, second_val):
    order = np.argsort(bus)
    bus = bus[order]
    index = np.ones(len(bus), 'bool')
    index[:-1] = bus[1:] != bus[:-1]
    bus = bus[index]
    first_val = first_val[order]
    first_val.cumsum(out=first_val)
    first_val = first_val[index]
    first_val[1:] = first_val[1:] - first_val[:-1]
    second_val = second_val[order]
    second_val.cumsum(out=second_val)
    second_val = second_val[index]
    second_val[1:] = second_val[1:] - second_val[:-1]
    return bus, first_val, second_val


def get_indices(selection, lookup, fused_indices=True):
    """
    Helper function during pd2mpc conversion. It resolves the mapping from a
    given selection of indices to the actual indices, using a dict lookup being
    passed as well.
    :param selection: Indices we want to select
    :param lookup: The mapping itself
    :param fused_indices: Flag which way the conversion is working.
    :return:
    """
    if fused_indices:
        return np.array([lookup[k] for k in selection], dtype="int")
    else:
        return np.array([lookup["before_fuse"][k] for k in selection], dtype="int")


def get_values(source, selection, lookup):
    """
    Returns values for a selection of values after a lookup.
    :param source: The array of values to select from.
    :param selection: An array of keys, for the selection.
    :param lookup: The mapping to resolve actual indices of the
    value array from the selection.
    :return:
    """
    return np.array([source[lookup[k]] for k in selection])


def _select_is_elements(net, recycle=None):
    """
    Selects certain "in_service" elements from net.
    This is quite time consuming so it is done once at the beginning
    @param net: Pandapower Network
    @return: is_elems Certain in service elements
    :rtype: object
    """

    if recycle is not None and recycle["is_elems"]:
        if "_is_elems" not in net or net["_is_elems"] is None:
            # sort elements according to their in service status
            elems = ['bus', 'line']
            for elm in elems:
                net[elm] = net[elm].sort_values(by=['in_service'], ascending=0)

            # select in service buses. needed for the other elements to be selected
            bus_is = net["bus"]["in_service"].values.astype(bool)
            line_is = net["line"]["in_service"].values.astype(bool)
            bus_is_ind = net["bus"][bus_is].index
            # check if in service elements are at in service buses
            is_elems = {
                "gen": net['gen'][np.in1d(net["gen"].bus.values, bus_is_ind) \
                                  & net["gen"]["in_service"].values.astype(bool)]
                , "load": np.in1d(net["load"].bus.values, bus_is_ind) \
                          & net["load"].in_service.values.astype(bool)
                , "sgen": np.in1d(net["sgen"].bus.values, bus_is_ind) \
                          & net["sgen"].in_service.values.astype(bool)
                , "ward": np.in1d(net["ward"].bus.values, bus_is_ind) \
                          & net["ward"].in_service.values.astype(bool)
                , "xward": np.in1d(net["xward"].bus.values, bus_is_ind) \
                           & net["xward"].in_service.values.astype(bool)
                , "shunt": np.in1d(net["shunt"].bus.values, bus_is_ind) \
                           & net["shunt"].in_service.values.astype(bool)
                , "ext_grid": net["ext_grid"][np.in1d(net["ext_grid"].bus.values, bus_is_ind) \
                                              & net["ext_grid"]["in_service"].values.astype(bool)]
                , 'bus': net['bus'].iloc[:np.count_nonzero(bus_is)]
                , 'line': net['line'].iloc[:np.count_nonzero(line_is)]
            }
        else:
            # just update the elements
            is_elems = net['_is_elems']

            bus_is_ind = is_elems['bus'].index
            # update elements
            elems = ['gen', 'ext_grid']
            for elm in elems:
                is_elems[elm] = net[elm][np.in1d(net[elm].bus.values, bus_is_ind) \
                                         & net[elm]["in_service"].values.astype(bool)]

    else:
        # select in service buses. needed for the other elements to be selected
        bus_is = net["bus"]["in_service"].values.astype(bool)
        line_is = net["line"]["in_service"].values.astype(bool)
        bus_is_ind = net["bus"][bus_is].index
        # check if in service elements are at in service buses
        is_elems = {
            "gen": net['gen'][np.in1d(net["gen"].bus.values, bus_is_ind) \
                              & net["gen"]["in_service"].values.astype(bool)]
            , "load": np.in1d(net["load"].bus.values, bus_is_ind) \
                      & net["load"].in_service.values.astype(bool)
            , "sgen": np.in1d(net["sgen"].bus.values, bus_is_ind) \
                      & net["sgen"].in_service.values.astype(bool)
            , "ward": np.in1d(net["ward"].bus.values, bus_is_ind) \
                      & net["ward"].in_service.values.astype(bool)
            , "xward": np.in1d(net["xward"].bus.values, bus_is_ind) \
                       & net["xward"].in_service.values.astype(bool)
            , "shunt": np.in1d(net["shunt"].bus.values, bus_is_ind) \
                       & net["shunt"].in_service.values.astype(bool)
            , "ext_grid": net["ext_grid"][np.in1d(net["ext_grid"].bus.values, bus_is_ind) \
                                          & net["ext_grid"]["in_service"].values.astype(bool)]
            , 'bus': net['bus'][bus_is]
            , 'line': net['line'][line_is]
        }

    return is_elems


def _create_options_dict(init="flat", calculate_voltage_angles=False, tolerance_kva=1e-5, trafo_model="t",
                         trafo_loading="current", enforce_q_lims=False, numba=True, recycle=None,
                         check_connectivity=False, ac=True, copy_constraints_to_ppc=False, mode="pf", r_switch=0.0,
                         algorithm='nr'):
    """
    creates dictionary for pf, opf and short circuit calculations from input parameters.
    """

    # recycle parameters
    if recycle == None:
        recycle = dict(is_elems=False, ppc=False, Ybus=False)

    options = {
        "init": init
        , "calculate_voltage_angles": calculate_voltage_angles
        , "copy_constraints_to_ppc": copy_constraints_to_ppc
        , "tolerance_kva": tolerance_kva
        , "trafo_model": trafo_model
        , "trafo_loading": trafo_loading
        , "enforce_q_lims": enforce_q_lims
        , "numba": numba
        , "recycle": recycle
        , "check_connectivity": check_connectivity
        , "ac": ac
        , "mode": mode
        , "r_switch": r_switch
        , "algorithm": algorithm
    }

    return options


def _clean_up(net):
    mode = net._options["mode"]
    if len(net["trafo3w"]) > 0:
        buses_3w = net.trafo3w["ad_bus"].values
        if mode == "sc":
            net["res_bus_sc"].drop(buses_3w, inplace=True)
        else:
            net["res_bus"].drop(buses_3w, inplace=True)
        net["bus"].drop(buses_3w, inplace=True)
        net["trafo3w"].drop(["ad_bus"], axis=1, inplace=True)

    if len(net["xward"]) > 0:
        xward_buses = net["xward"]["ad_bus"].values
        net["bus"].drop(xward_buses, inplace=True)
        if mode == "sc":
            net["res_bus_sc"].drop(xward_buses, inplace=True)
        else:
            net["res_bus"].drop(xward_buses, inplace=True)
        net["xward"].drop(["ad_bus"], axis=1, inplace=True)

    if len(net["dcline"]) > 0:
        dc_gens = net.gen.index[(len(net.gen) - len(net.dcline) * 2):]
        net.gen.drop(dc_gens, inplace=True)
        net.res_gen.drop(dc_gens, inplace=True)
    if mode == "sc":
        for var in ["kappa_max", "z_equiv", "kappa_korr", "kappa", "c_max", "c_min", "x", "r",
                    "z_equiv", "kappa_max"]:
            for element in ["bus", "line", "ext_grid", "trafo"]:
                if var in net[element]:
                    net[element].drop(var, axis=1, inplace=True)


def _set_isolated_buses_out_of_service(net, ppc):
    # set disconnected buses out of service
    # first check if buses are connected to branches
    disco = np.setxor1d(ppc["bus"][:, 0].astype(int),
                        ppc["branch"][ppc["branch"][:, 10] == 1, :2].real.astype(int).flatten())

    # but also check if they may be the only connection to an ext_grid
    disco = np.setdiff1d(disco, ppc['bus'][ppc['bus'][:, 1] == 3, :1].real.astype(int))
    ppc["bus"][disco, 1] = 4


def calculate_line_results(net, use_res_bus_est=False):
    """
    Calculates complex line currents, powers at both bus sides and saves them in the result table.
    Requires the res_bus or res_bus_est table of the network to be filled.
    :param net: pandapower network
    :param use_res_bus_est: use res_bus_est dataframe instead of res_bus
    :return: new dataframe, which can be assigned to either res_line or res_line_est
    """
    res_line = pd.DataFrame(columns=["p_from_kw", "q_from_kvar", "p_to_kw", "q_to_kvar", "pl_kw",
                                     "ql_kvar", "i_from_ka", "i_to_ka", "i_ka", "loading_percent"],
                            index=net.line.index)
    # calculate impedances and complex voltages
    Zij = net.line['length_km'] * (net.line['r_ohm_per_km'] + 1j * net.line['x_ohm_per_km'])
    Zcbij = 0.5j * 2 * np.pi * 50 * net.line['c_nf_per_km'] * 1e-9
    if use_res_bus_est:
        V = net.res_bus_est.vm_pu * net.bus.vn_kv * 1e3 * np.exp(
            1j * np.pi / 180 * net.res_bus_est.va_degree)
    else:
        V = net.res_bus.vm_pu * net.bus.vn_kv * 1e3 * np.exp(
            1j * np.pi / 180 * net.res_bus.va_degree)
    fb = net.line.from_bus
    tb = net.line.to_bus
    # calculate line currents of from bus side
    line_currents_from = ((V[fb].values - V[tb].values) / np.sqrt(3) / Zij + V[fb].values
                          * Zcbij).values
    open_lines_from = net.switch.element.loc[(net.switch.et == 'l') & (net.switch.closed == False)]
    line_currents_from[open_lines_from.values] = 0.
    charging_from = open_lines_from[open_lines_from.index[
        net.line.to_bus.loc[open_lines_from].values ==
        net.switch.bus.loc[(net.switch.et == 'l') & (net.switch.closed == False)].values]].values
    line_currents_from[charging_from] = V[net.line.ix[charging_from].from_bus].values \
                                        * Zcbij[charging_from] * (1 + Zij[charging_from])
    # calculate line currents on to bus side
    line_currents_to = ((V[tb].values - V[fb].values) / np.sqrt(3) / Zij + V[tb].values
                        * Zcbij).values
    open_lines_to = net.switch.element.loc[(net.switch.et == 'l') & (net.switch.closed == False)]
    line_currents_to[open_lines_to.values] = 0.
    charging_to = open_lines_to[open_lines_to.index[
        net.line.from_bus.loc[open_lines_to].values ==
        net.switch.bus.loc[(net.switch.et == 'l') & (net.switch.closed == False)].values]].values
    line_currents_to[charging_to] = V[net.line.ix[charging_to].to_bus] * Zcbij[charging_to] \
                                    * (1 + Zij[charging_to])
    # derive other values
    line_powers_from = np.sqrt(3) * V[fb].values * np.conj(line_currents_from) / 1e3
    line_powers_to = np.sqrt(3) * V[tb].values * np.conj(line_currents_to) / 1e3
    res_line.i_from_ka = np.abs(line_currents_from) / 1e3
    res_line.i_to_ka = np.abs(line_currents_to) / 1e3
    res_line.i_ka = np.fmax(res_line.i_from_ka, res_line.i_to_ka)
    res_line.loading_percent = res_line.i_ka * 100. / net.line.max_i_ka.values \
                               / net.line.df.values / net.line.parallel.values
    res_line.p_from_kw = line_powers_from.real
    res_line.q_from_kvar = line_powers_from.imag
    res_line.p_to_kw = line_powers_to.real
    res_line.q_to_kvar = line_powers_to.imag
    res_line.pl_kw = res_line.p_from_kw + res_line.p_to_kw
    res_line.ql_kvar = res_line.q_from_kvar + res_line.q_to_kvar
    return res_line


def _write_lookup_to_net(net, element, element_lookup):
    """
    Updates selected lookups in net
    """
    net["_pd2ppc_lookups"][element] = element_lookup


def _check_connectivity(ppc):
    """
    Checks if the ppc contains isolated buses. If yes this isolated buses are set out of service
    :param ppc: pyPoer matrix
    :return:
    """
    nobranch = ppc['branch'].shape[0]
    nobus = ppc['bus'].shape[0]
    bus_from = ppc['branch'][:, F_BUS].real.astype(int)
    bus_to = ppc['branch'][:, T_BUS].real.astype(int)

    adj_matrix = sp.sparse.csr_matrix((np.ones(nobranch), (bus_from, bus_to)),
                                      shape=(nobus, nobus))

    slacks = ppc['bus'][ppc['bus'][:, BUS_TYPE] == 3, BUS_I]

    all_nodes = set(ppc['bus'][:, BUS_I].astype(int))

    visited_nodes = set()

    for slack in slacks:
        node_array = sp.sparse.csgraph.depth_first_order(adj_matrix, slack, False, False)
        node_set = set(node_array)
        visited_nodes = visited_nodes.union(node_set)

    isolated_nodes = all_nodes.difference(visited_nodes)

    if isolated_nodes:
        logger.debug("There are isolated buses in the network!")
        index_array = np.array(list(isolated_nodes), dtype=np.int)
        # set buses in ppc out of service
        ppc['bus'][index_array, BUS_TYPE] = NONE

        iso_p = abs(ppc['bus'][index_array, PD] * 1e3).sum()
        iso_q = abs(ppc['bus'][index_array, QD] * 1e3).sum()
        if iso_p > 0 or iso_q > 0:
            logger.debug("%.0f kW active and %.0f kVar reactive power are unsupplied" % (iso_p, iso_q))
    else:
        iso_p = iso_q = 0
    return isolated_nodes, iso_p, iso_q


def _create_ppc2pd_bus_lookup(net):
    # pd to ppc lookup
    pd2ppc_bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # valid entries in pd2ppc lookup
    valid_entries = pd2ppc_bus_lookup >= 0
    # init reverse (ppc2pd) lookup with -1
    ppc2pd_bus_lookup = np.ones(max(pd2ppc_bus_lookup[valid_entries]) + 1, dtype=int) * -1
    # index of pd2ppc lookup
    ind_pd2ppc_bus_lookup = np.array(range(len(pd2ppc_bus_lookup)), dtype=int)
    # update reverse lookup
    ppc2pd_bus_lookup[pd2ppc_bus_lookup[valid_entries]] = ind_pd2ppc_bus_lookup[valid_entries]
    # store reverse lookup innet
    net["_ppc2pd_lookups"]["bus"] = ppc2pd_bus_lookup


def _remove_isolated_elements_from_is_elements(net, isolated_nodes):
    pcc2pd_bus_lookup = net["_ppc2pd_lookups"]["bus"]
    is_elems = net["_is_elems"]
    pp_nodes = [n for n in isolated_nodes if not (n > len(pcc2pd_bus_lookup))]
    isolated_nodes_pp = pcc2pd_bus_lookup[pp_nodes]
    # remove isolated buses from is_elems["bus"]
    is_elems["bus"] = is_elems["bus"].drop(set(isolated_nodes_pp) & set(is_elems["bus"].index))
    bus_is_ind = is_elems["bus"].index
    # check if in service elements are at in service buses

    elems_to_update = ["load", "sgen", "ward", "xward", "shunt"]
    for elem in elems_to_update:
        is_elems[elem] = np.in1d(net[elem].bus.values, bus_is_ind) \
                         & net[elem].in_service.values.astype(bool)

    is_elems["gen"] = net['gen'][np.in1d(net["gen"].bus.values, bus_is_ind) \
                                 & net["gen"]["in_service"].values.astype(bool)]

    net["_is_elems"] = is_elems