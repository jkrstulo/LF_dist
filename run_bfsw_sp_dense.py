import sys

import numpy as np
import scipy as sp
import networkx as nx

from time import time
# import timeit.default_timer as time

from scipy.sparse import issparse, csr_matrix

from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, VM, VA, REF, GS, BS
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, PF, QF, PT, QT, TAP, BR_STATUS
from pypower.idx_gen import GEN_BUS, PG, QG, PMAX, PMIN, QMAX, QMIN, GEN_STATUS, VG

# these functions are imported from pypower_extensions in order to avoid warnings because of complex branch mtx
from pandapower.pypower_extensions.makeYbus_pypower import makeYbus
from pandapower.pypower_extensions.pfsoln import pfsoln
from pandapower.pypower_extensions.bustypes import bustypes

from pypower.makeSbus import makeSbus
from pypower.ppoption import ppoption

from collections import defaultdict, deque


class ConvergenceError(Exception):
    pass


def graph_dict(branches):
    """
    :param branches:
    :return: dictionary of graph: each node is a key to a set of incident nodes
    """
    gd = defaultdict(list)
    for brch in branches:
        gd[brch[0]].append(brch[1])
        gd[brch[1]].append(brch[0])
    return gd


def bfs_edges(G, start):
    """
    breadth first search of a dictionary-based graph
    :param G: graph defined as a dictionary
    :param start: start node
    :return: ordered list of buses, ordered list of branches, list of edges which close the loops
    """
    edges_ordered_list = []
    branches_loops = []
    neighbors = G[start]
    visited = set([start])
    visited_branches = set()
    queue = deque([(start, neighbors)])
    bus_ordered_list = [start]
    while queue:
        parent, children = queue[0]
        for child in children:
            if child not in visited:
                edges_ordered_list.append([parent, child])
                visited.add(child)
                visited_branches.add((parent, child))
                visited_branches.add((child, parent))
                neighbors = G[child]
                queue.append((child, neighbors))
                bus_ordered_list.append(child)
            elif (parent, child) not in visited_branches:  # loop detected
                if parent > child:
                    branches_loops.append((child, parent))
                else:
                    branches_loops.append((parent, child))
                visited_branches.add((child, parent))
                visited_branches.add((parent, child))
        queue.popleft()

    return bus_ordered_list, edges_ordered_list, branches_loops


def bus_reindex(ppc, bus_ind_dict):
    """
    reindexing buses according to dictionary
    :param ppc: matpower-type power system
    :param bus_ind_dict:  dict for bus reindexing
    :return: ppc with buses and branches ordered according to bus_ordered_list
    """
    ppc_bfs = ppc.copy()
    buses = ppc_bfs['bus'].copy()
    branches = ppc_bfs['branch'].copy()
    generators = ppc_bfs['gen'].copy()

    buses[:, BUS_I] = [bus_ind_dict[bus] for bus in buses[:, BUS_I]]

    branches[:, F_BUS] = [bus_ind_dict[bus] for bus in branches[:, F_BUS]]
    branches[:, T_BUS] = [bus_ind_dict[bus] for bus in branches[:, T_BUS]]
    branches[:, F_BUS:T_BUS + 1] = np.sort(branches[:, F_BUS:T_BUS + 1], axis=1)  # sort in order to T_BUS > F_BUS

    generators[:, GEN_BUS] = [bus_ind_dict[bus] for bus in generators[:, GEN_BUS]]

    # sort buses, branches and generators according to new numbering
    ppc_bfs['bus'] = buses[np.argsort(buses[:, BUS_I])]
    ppc_bfs['branch'] = branches[np.lexsort((branches[:, T_BUS], branches[:, F_BUS]))]
    ppc_bfs['gen'] = generators[np.argsort(generators[:, GEN_BUS])]

    return ppc_bfs


def bibc_bcbv_dense(ppc):
    """
    creates 2 matrices required for direct LF:
        BIBC - Bus Injection to Branch-Current
        BCBV - Branch-Current to Bus-Voltage
    :param ppc:
    matpower-type power system dictionary
    :return: DLF matrix DLF = BIBC * BCBV in a dense form
    """
    ppci = ppc
    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]
    nbus = bus.shape[0]
    nbrch = branch.shape[0]

    # get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    root_bus = ref[0]  # reference bus is assumed as root bus for a radial network


    branches = branch[:, F_BUS:T_BUS + 1].real.astype(int)

    # power system graph created from list of branches as a dictionary
    system_graph = graph_dict(branches)

    # ###
    # bus ordering according to BFS search

    # TODO check if bfs bus ordering is necessary for the direct power flow
    buses_ordered_bfs, edges_ordered_bfs, branches_loops = bfs_edges(system_graph, root_bus)
    buses_bfs_dict = dict(zip(buses_ordered_bfs, range(0, nbus)))  # old to new bus names
    ppc_bfs = bus_reindex(ppci, buses_bfs_dict)

    branches_loops_bfs = []
    if len(branches_loops) > 0:
        print ("  {0} loops detected\n  following branches are cut to obtain radial network: {1}".format(
            len(branches_loops), branches_loops))
        branches_loops_bfs = [(buses_bfs_dict[fbus], buses_bfs_dict[tbus]) for fbus, tbus in branches_loops]
        branches_loops_bfs = np.sort(branches_loops_bfs, axis=1)  # sort in order to T_BUS > F_BUS
        branches_loops_bfs = list(zip(branches_loops_bfs[:, 0], branches_loops_bfs[:, 1]))


    buses = ppc_bfs['bus'][:, BUS_I]
    branches_bfs = zip(ppc_bfs['branch'][:, F_BUS], ppc_bfs['branch'][:, T_BUS])

    nbus = buses.shape[0]

    mask_root = ~ (ppc_bfs['bus'][:, BUS_TYPE] == 3)
    root_bus = buses[~mask_root][0]

    # dictionaries with bus/branch indices
    bus_ind_dict = dict(zip(buses[mask_root], range(nbus - 1)))  # dictionary bus_name-> bus_ind
    bus_ind_dict[root_bus] = -1
    # dictionary brch_name-> brch_ind
    brch_ind_dict = dict(zip(branches_bfs, range(nbrch)))

    # list of branches in the radial network, i.e. without branches that close loops
    branches_radial = list(branches_bfs)
    for brch_l in branches_loops_bfs:
        branches_radial.remove(brch_l)

    # reorder branches so that loop-forming branches are at the end

    Z_brchs = []
    nloops = len(branches_loops_bfs)

    BIBC = np.zeros((nbrch - nloops, nbus - 1))
    BCBV = np.zeros((nbus - 1, nbrch - nloops), dtype=complex)

    for br_i, branch in enumerate(branches_radial):
        bus_f = branch[0]
        bus_t = branch[1]
        bus_f_i = bus_ind_dict[bus_f]
        bus_t_i = bus_ind_dict[bus_t]
        brch_ind = brch_ind_dict[branch]
        if bus_f != root_bus and bus_t != root_bus:
            BIBC[:, bus_t_i] = BIBC[:, bus_f_i].copy()
            BCBV[bus_t_i, :] = BCBV[bus_f_i, :].copy()

        if branch[T_BUS] != root_bus:
            BIBC[br_i, bus_t_i] += 1
            Z = (ppc_bfs['branch'][brch_ind, BR_R] + 1j * ppc_bfs['branch'][brch_ind, BR_X])
            BCBV[bus_t_i, br_i] = Z
            Z_brchs.append(Z)

    for br_i, branch in enumerate(branches_loops_bfs):
        bus_f = branch[0]
        bus_t = branch[1]
        bus_f_i = bus_ind_dict[bus_f]
        bus_t_i = bus_ind_dict[bus_t]
        brch_ind = brch_ind_dict[branch]

        BIBC = np.vstack((BIBC, np.zeros(BIBC.shape[1])))  # add row
        BIBC = np.hstack((BIBC, np.zeros((BIBC.shape[0], 1))))  # add column
        last_col_i = BIBC.shape[1] - 1  # last column index
        BIBC[:, last_col_i] = BIBC[:, bus_f_i] - BIBC[:, bus_t_i]
        BIBC[last_col_i, last_col_i] += 1

        BCBV = np.vstack((BCBV, np.zeros(BCBV.shape[1])))  # add row
        BCBV = np.hstack((BCBV, np.zeros((BCBV.shape[0], 1))))  # add column
        Z = (ppc_bfs['branch'][brch_ind, BR_R] + 1j * ppc_bfs['branch'][brch_ind, BR_X])
        Z_brchs.append(Z)
        BCBV[BCBV.shape[0] - 1, :] = np.array(Z_brchs) * BIBC[:, last_col_i]  # KVL for the loop

    if BCBV.shape[0] > nbus - 1:  # if nbrch > nbus - 1 -> network has loops
        DLF_loop = sp.dot(BCBV, BIBC)
        A = DLF_loop[0:nbus - 1, 0:nbus - 1]
        M = DLF_loop[nbus - 1:, 0:nbus - 1]
        N = DLF_loop[nbus - 1:, nbus - 1:]
        # TODO: use more efficient inversion !?
        DLF = A - sp.dot(sp.dot(M.T, sp.linalg.inv(N)), M)  # Krons Reduction
    else:  # no loops -> radial network
        DLF = sp.dot(BCBV, BIBC)

    return DLF, ppc_bfs, buses_ordered_bfs


def bibc_bcbv(ppc):
    """
    performs depth-first-search bus ordering and creates Direct Load Flow (DLF) matrix
    which establishes direct relation between bus current injections and voltage drops from each bus to the root bus

    :param ppc: matpower-type case data
    :return: DLF matrix DLF = BIBC * BCBV where
                    BIBC - Bus Injection to Branch-Current
                    BCBV - Branch-Current to Bus-Voltage
            ppc with bfs ordering
            original bus names bfs ordered (used to convert voltage array back to normal)
    """

    ppci = ppc
    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]
    nbus = bus.shape[0]

    # get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    root_bus = ref[0]  # reference bus is assumed as root bus for a radial network

    branches = branch[:, F_BUS:T_BUS + 1].real.astype(int)

    # creating networkx graph from list of branches
    G = nx.Graph()
    G.add_edges_from(branches)

    # ordering buses according to breadth-first-search (bfs)
    edges_ordered_bfs = list(nx.bfs_edges(G, root_bus))
    indices = np.unique(np.array(edges_ordered_bfs).flatten(), return_index=True)[1]
    buses_ordered_bfs = np.array(edges_ordered_bfs).flatten()[sorted(indices)]
    buses_bfs_dict = dict(zip(buses_ordered_bfs, range(0, nbus)))  # old to new bus names dictionary
    # renaming buses in graph and in ppc
    G = nx.relabel_nodes(G, buses_bfs_dict)
    root_bus = buses_bfs_dict[root_bus]
    ppc_bfs = bus_reindex(ppci, buses_bfs_dict)
    # ordered list of branches
    branches_ord = zip(ppc_bfs['branch'][:, F_BUS].real.astype(int), ppc_bfs['branch'][:, T_BUS].real.astype(int))

    # searching loops in the graph if it is not a tree
    loops = []
    branches_loops = []
    if not nx.is_tree(G):  # network is meshed, i.e. has loops
        G_bfs_tree = nx.bfs_tree(G, root_bus)
        branches_loops = list(set(G.edges()) - set(G_bfs_tree.edges()))
        G.remove_edges_from(branches_loops)
        # finding loops
        for i, j in branches_loops:
            G.add_edge(i, j)
            loops.append(nx.find_cycle(G))
            G.remove_edge(i, j)

    nloops = len(loops)
    nbr_rad = len(G.edges())  # number of edges in the radial network

    # searching leaves of the tree
    succ = nx.bfs_successors(G, root_bus)
    leaves = set(G.nodes()) - set(succ.keys())

    # dictionary with impedance values keyed by branch tuple (frombus, tobus)
    Z_brch_dict = dict(zip(branches_ord, ppc_bfs['branch'][:, BR_R].real + 1j * ppc_bfs['branch'][:, BR_X].real))


    # #------ building BIBC and BCBV martrices ------

    # order branches for BIBC and BCBV matrices and set loop-closing branches to the end
    branches_ord_radial = list(branches_ord)
    for brch in branches_loops:  # TODO eliminated this for loop
        branches_ord_radial.remove(brch)
    branches_ind_dict = dict(zip(branches_ord_radial, range(0, nbr_rad)))
    branches_ind_dict.update(dict(zip(branches_loops, range(nbr_rad, nbr_rad + nloops))))  # add loop-closing branches

    rowi_BIBC = []
    coli_BIBC = []
    data_BIBC = []
    data_BCBV = []
    for bri, (i, j) in enumerate(branches_ord_radial):
        G.remove_edge(i, j)
        buses_down = set()
        for leaf in leaves:
            try:
                buses_down.update(nx.shortest_path(G, leaf, j))
            except:
                pass
        rowi_BIBC += [bri] * len(buses_down)
        coli_BIBC += list(buses_down)
        data_BCBV += [Z_brch_dict[(i, j)]] * len(buses_down)
        data_BIBC += [1] * len(buses_down)
        G.add_edge(i, j)

    for loop_i, loop in enumerate(loops):
        loop_size = len(loop)
        coli_BIBC += [nbus + loop_i] * loop_size
        for brch in loop:
            if brch[0] < brch[1]:
                i, j = brch
                brch_direct = 1
                data_BIBC.append(brch_direct)
            else:
                j, i = brch
                brch_direct = -1
                data_BIBC.append(brch_direct)
            rowi_BIBC.append(branches_ind_dict[(i, j)])

            data_BCBV.append(Z_brch_dict[(i, j)] * brch_direct)

    # construction of the BIBC matrix
    # column indices correspond to buses: assuming root bus is always 0 after ordering indices are subtracted by 1
    BIBC = csr_matrix((data_BIBC, (rowi_BIBC, np.array(coli_BIBC) - 1)),
                  shape=(nbus - 1 + nloops, nbus - 1 + nloops))
    BCBV = csr_matrix((data_BCBV, (rowi_BIBC, np.array(coli_BIBC) - 1)),
                  shape=(nbus - 1 + nloops, nbus - 1 + nloops)).transpose()

    if BCBV.shape[0] > nbus - 1:  # if nbrch > nbus - 1 -> network has loops
        DLF_loop = BCBV * BIBC
        # DLF = [A  M.T ]
        #       [M  N   ]
        A = DLF_loop[0:nbus - 1, 0:nbus - 1]
        M = DLF_loop[nbus - 1:, 0:nbus - 1]
        N = DLF_loop[nbus - 1:, nbus - 1:].A
        # considering the fact that number of loops is relatively small, N matrix is expected to be small and dense
        # ...in that case dense version is more efficient, i.e. N is transformed to dense and
        # inverted using sp.linalg.inv(N)
        DLF = A - M.T * csr_matrix(sp.linalg.inv(N)) * M  # Kron's Reduction
    else:  # no loops -> radial network
        DLF = BCBV * BIBC

    return DLF, ppc_bfs, buses_ordered_bfs


def bfsw_dense(DLF, bus, gen, branch, baseMVA, Ybus, Sbus, V0, ref, pv, pq, ppopt=None):
    """
    distribution power flow solution according to [1]
    :param ppc: power system matpower-type
    :param BIBC: bus-injection to branch-current matrix
    :param BCBV: branch-currentn to bus-voltage matrix
    :param epsilon: termination criterion
    :param maxiter: maximum number of iterations before raising divergence error
    :return: power flow result
    :References:
    [1] Jen-Hao Teng, "A Direct Approach for Distribution System Load Flow Solutions", IEEE Transactions on Power Delivery, vol. 18, no. 3, pp. 882-887, July 2003.
    """
    ## options
    if ppopt is None:
        ppopt = ppoption()
    tol = ppopt['PF_TOL']
    max_it = ppopt['PF_MAX_IT_GS']  # maximum iterations from Gauss-Seidel
    verbose = ppopt['VERBOSE']

    nbus = bus.shape[0]

    mask_root = ~ (bus[:, BUS_TYPE] == 3)  # mask for eliminating root bus
    root_bus_i = ref
    Vref = V0[ref]

    bus_ind_mask_dict = dict(zip(bus[mask_root, BUS_I], range(nbus - 1)))

    # detect PV buses
    busPV_ind_mask = [bus_ind_mask_dict[pvbus] for pvbus in pv]
    genPV_ind = np.where(np.in1d(gen[:, GEN_BUS], pv))[0].flatten()

    # compute shunt admittance
    # if Psh is the real power consumed by the shunt at V = 1.0 p.u. and Qsh is the reactive power injected by
    # the shunt at V = 1.0 p.u. then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    # vector of shunt admittances
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    # Line charging susceptance BR_B is also added as shunt admittance:
    # summation of charging susceptances per each bus
    Gch_f = - np.bincount(branch[:, F_BUS].real.astype(int), weights=branch[:, BR_B].imag / 2, minlength=nbus)
    Bch_f = np.bincount(branch[:, F_BUS].real.astype(int), weights=branch[:, BR_B].real / 2, minlength=nbus)
    Gch_t = - np.bincount(branch[:, T_BUS].real.astype(int), weights=branch[:, BR_B].imag / 2, minlength=nbus)
    Bch_t = np.bincount(branch[:, T_BUS].real.astype(int), weights=branch[:, BR_B].real / 2, minlength=nbus)

    Ysh += (Gch_f + Gch_t) + 1j * (Bch_f + Bch_t)  # adding line charging to shunt impedance vector

    V_iter = V0[mask_root].copy()  # initial voltage vector without root bus
    V = V0.copy()
    Iinj = np.conj(Sbus[mask_root] / V_iter) - Ysh[mask_root] * V_iter  # Initial current injections

    n_iter = 0
    converged = 0

    while not converged and n_iter < max_it:
        n_iter_inner = 0
        n_iter += 1

        deltaV = sp.dot(DLF, Iinj)
        V_new = np.ones(nbus - 1) * Vref + deltaV

        # ##
        # inner loop for considering PV buses
        inner_loop_citerion = False
        V_inner = V_new.copy()

        success_inner = 1
        while not inner_loop_citerion and len(pv) > 0:
            Vmis = (np.abs(gen[genPV_ind, VG])) ** 2 - (np.abs(V_inner[busPV_ind_mask])) ** 2
            dQ = Vmis / (2 * DLF[busPV_ind_mask, busPV_ind_mask].imag)

            gen[genPV_ind, QG] += dQ
            if (gen[genPV_ind, QG] < gen[genPV_ind, QMIN]).any():
                Qviol_ind = np.argwhere((gen[genPV_ind, QG] < gen[genPV_ind, QMIN])).flatten()
                gen[genPV_ind[Qviol_ind], QG] = gen[genPV_ind[Qviol_ind], QMIN]
            elif (gen[genPV_ind, QG] > gen[genPV_ind, QMAX]).any():
                Qviol_ind = np.argwhere((gen[genPV_ind, QG] < gen[genPV_ind, QMIN])).flatten()
                gen[genPV_ind[Qviol_ind], QG] = gen[genPV_ind[Qviol_ind], QMAX]

            Sbus = makeSbus(baseMVA, bus, gen)
            Iinj = np.conj(Sbus[mask_root] / V_inner) - Ysh[mask_root] * V_inner
            deltaV = sp.dot(DLF, Iinj)
            V_inner = np.ones(nbus - 1) * V0[root_bus_i] + deltaV

            if n_iter_inner > 20 or np.any(np.abs(V_inner[busPV_ind_mask]) > 2):
                success_inner = 0
                break
                # raise ConvergenceError("\n\t inner iterations (PV nodes) did not converge!!!")

            n_iter_inner += 1

            if np.all(np.abs(dQ) < 1.e-2):  # inner loop termination criterion
                inner_loop_citerion = True
                V_new = V_inner.copy()

        if not success_inner:
            if verbose:
                sys.stdout.write("\nFwd-back sweep power flow did not converge in "
                                 "{0} iterations.\n".format(n_iter))
            break

        # testing termination criterion -
        V = np.insert(V_new, root_bus_i, Vref)
        mis = V * np.conj(Ybus * V) - Sbus
        F = np.r_[mis[pv].real,
                  mis[pq].real,
                  mis[pq].imag]

        # check tolerance
        normF = np.linalg.norm(F, np.Inf)

        # deltaVmax = np.max(np.abs(V_new - V_iter))

        if normF < tol:
            converged = 1
            if verbose:
                sys.stdout.write("\nFwd-back sweep power flow converged in "
                                 "{0} iterations.\n".format(n_iter))

        V_iter = V_new.copy()  # update iterating complex voltage vector

        # updating injected currents
        Iinj = np.conj(Sbus[mask_root] / V_iter) - Ysh[mask_root] * V_iter

    return V, converged


def bfsw(DLF, bus, gen, branch, baseMVA, Ybus, Sbus, V0, ref, pv, pq, ppopt=None, tol_inner=1e-2):
    """
    distribution power flow solution according to [1]
    :param DLF: direct-Load-Flow matrix which relates bus current injections to voltage drops from the root bus

    :param bus: buses martix
    :param gen: generators matrix
    :param branch: branches matrix
    :param baseMVA:
    :param Ybus: bus admittance matrix
    :param Sbus: vector of power injections
    :param V0: initial voltage state vector
    :param ref: reference bus index
    :param pv: PV buses indices
    :param pq: PQ buses indices

    :return: power flow result

    :References:
    [1] Jen-Hao Teng, "A Direct Approach for Distribution System Load Flow Solutions", IEEE Transactions on Power Delivery, vol. 18, no. 3, pp. 882-887, July 2003.
    """
    ## options
    if ppopt is None:
        ppopt = ppoption()
    tol = ppopt['PF_TOL']
    max_it = ppopt['PF_MAX_IT_GS']  # maximum iterations from Gauss-Seidel
    verbose = ppopt['VERBOSE']

    nbus = bus.shape[0]

    mask_root = ~ (bus[:, BUS_TYPE] == 3)  # mask for eliminating root bus
    root_bus_i = ref
    Vref = V0[ref]

    bus_ind_mask_dict = dict(zip(bus[mask_root, BUS_I], range(nbus - 1)))

    # detect PV buses
    busPV_ind_mask = [bus_ind_mask_dict[pvbus] for pvbus in pv]
    genPV_ind = np.where(np.in1d(gen[:, GEN_BUS], pv))[0].flatten()

    # compute shunt admittance
    # if Psh is the real power consumed by the shunt at V = 1.0 p.u. and Qsh is the reactive power injected by
    # the shunt at V = 1.0 p.u. then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    # vector of shunt admittances
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    # Line charging susceptance BR_B is also added as shunt admittance:
    # summation of charging susceptances per each bus
    Gch_f = - np.bincount(branch[:, F_BUS].real.astype(int), weights=branch[:, BR_B].imag / 2, minlength=nbus)
    Bch_f = np.bincount(branch[:, F_BUS].real.astype(int), weights=branch[:, BR_B].real / 2, minlength=nbus)
    Gch_t = - np.bincount(branch[:, T_BUS].real.astype(int), weights=branch[:, BR_B].imag / 2, minlength=nbus)
    Bch_t = np.bincount(branch[:, T_BUS].real.astype(int), weights=branch[:, BR_B].real / 2, minlength=nbus)

    Ysh += (Gch_f + Gch_t) + 1j * (Bch_f + Bch_t)  # adding line charging to shunt impedance vector

    V_iter = V0[mask_root].copy()  # initial voltage vector without root bus
    V = V0.copy()
    Iinj = np.conj(Sbus[mask_root] / V_iter) - Ysh[mask_root] * V_iter  # Initial current injections

    n_iter = 0
    converged = 0

    while not converged and n_iter < max_it:
        n_iter_inner = 0
        n_iter += 1

        deltaV = DLF * Iinj
        V_new = np.ones(nbus - 1) * Vref + deltaV

        # ##
        # inner loop for considering PV buses
        inner_loop_converged = False
        V_inner = V_new.copy()

        success_inner = 1
        while not inner_loop_converged and len(pv) > 0:
            Vmis = (np.abs(gen[genPV_ind, VG])) ** 2 - (np.abs(V_inner[busPV_ind_mask])) ** 2
            dQ = (Vmis / (2 * DLF[busPV_ind_mask, busPV_ind_mask].A1.imag)).flatten()

            gen[genPV_ind, QG] += dQ
            if (gen[genPV_ind, QG] < gen[genPV_ind, QMIN]).any():
                Qviol_ind = np.argwhere((gen[genPV_ind, QG] < gen[genPV_ind, QMIN])).flatten()
                gen[genPV_ind[Qviol_ind], QG] = gen[genPV_ind[Qviol_ind], QMIN]
            elif (gen[genPV_ind, QG] > gen[genPV_ind, QMAX]).any():
                Qviol_ind = np.argwhere((gen[genPV_ind, QG] < gen[genPV_ind, QMIN])).flatten()
                gen[genPV_ind[Qviol_ind], QG] = gen[genPV_ind[Qviol_ind], QMAX]

            Sbus = makeSbus(baseMVA, bus, gen)
            Iinj = np.conj(Sbus[mask_root] / V_inner) - Ysh[mask_root] * V_inner
            deltaV = DLF * Iinj
            V_inner = np.ones(nbus - 1) * V0[root_bus_i] + deltaV

            if n_iter_inner > 20 or np.any(np.abs(V_inner[busPV_ind_mask]) > 2):
                success_inner = 0
                break
                # raise ConvergenceError("\n\t inner iterations (PV nodes) did not converge!!!")

            n_iter_inner += 1

            if np.all(np.abs(dQ) < tol_inner):  # inner loop termination criterion
                inner_loop_converged = True
                V_new = V_inner.copy()

        if not success_inner:
            if verbose:
                sys.stdout.write("\nFwd-back sweep power flow did not converge in "
                                 "{0} iterations.\n".format(n_iter))
            break

        # testing termination criterion -
        V = np.insert(V_new, root_bus_i, Vref)
        mis = V * np.conj(Ybus * V) - Sbus
        F = np.r_[mis[pv].real,
                  mis[pq].real,
                  mis[pq].imag]

        # check tolerance
        normF = np.linalg.norm(F, np.Inf)

        # deltaVmax = np.max(np.abs(V_new - V_iter))

        if normF < tol:
            converged = 1
            if verbose:
                sys.stdout.write("\nFwd-back sweep power flow converged in "
                                 "{0} iterations.\n".format(n_iter))

        V_iter = V_new.copy()  # update iterating complex voltage vector

        # updating injected currents
        Iinj = np.conj(Sbus[mask_root] / V_iter) - Ysh[mask_root] * V_iter

    return V, converged




def _run_bfsw_dense(ppc, ppopt=None):
    """
    DENSE version of distribution power flow solution according to [1]
    :References:
    [1] Jen-Hao Teng, "A Direct Approach for Distribution System Load Flow Solutions", IEEE Transactions on Power Delivery, vol. 18, no. 3, pp. 882-887, July 2003.

    :param ppc: matpower-style case data
    :return: results (pypower style), success (flag about PF convergence)
    """
    # time_start = time()

    # path_search = nx.shortest_path

    # ppci = ext2int(ppc)
    ppci = ppc

    ppopt = ppoption(ppopt)

    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]
    nbus = bus.shape[0]

    # get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    root_bus = ref[0]  # reference bus is assumed as root bus for a radial network

    DLF, ppc_bfsw, buses_ordered_bfsw = bibc_bcbv_dense(ppci)

    baseMVA_bfsw, bus_bfsw, gen_bfsw, branch_bfsw = \
        ppc_bfsw["baseMVA"], ppc_bfsw["bus"], ppc_bfsw["gen"], ppc_bfsw["branch"]

    time_start = time()

    # initialize voltages to flat start and buses with gens to their setpoints
    V0 = np.ones(nbus, dtype=complex)
    V0[gen[:, GEN_BUS].astype(int)] = gen[:, VG]

    Sbus_bfsw = makeSbus(baseMVA_bfsw, bus_bfsw, gen_bfsw)

    # update data matrices with solution
    Ybus_bfsw, Yf_bfsw, Yt_bfsw = makeYbus(baseMVA_bfsw, bus_bfsw, branch_bfsw)
    ## get bus index lists of each type of bus
    ref_bfsw, pv_bfsw, pq_bfsw = bustypes(bus_bfsw, gen_bfsw)

    ##-----  run the power flow  -----

    # ###
    # LF initialization and calculation
    V_final, success = bfsw_dense(DLF, bus_bfsw, gen_bfsw, branch_bfsw, baseMVA_bfsw, Ybus_bfsw, Sbus_bfsw, V0,
                            ref_bfsw, pv_bfsw, pq_bfsw, ppopt=ppopt)
    V_final = V_final[np.argsort(buses_ordered_bfsw)]  # return bus voltages in original bus order

    ppci["et"] = time() - time_start

    Sbus = makeSbus(baseMVA, bus, gen)
    # update data matrices with solution
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V_final, ref, pv, pq)

    ppci["success"] = success

    ##-----  output results  -----
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    results = ppci

    return results, success



def _run_bfsw_ppc(ppc, ppopt=None):
    """
    SPARSE version of distribution power flow solution according to [1]
    :References:
    [1] Jen-Hao Teng, "A Direct Approach for Distribution System Load Flow Solutions", IEEE Transactions on Power Delivery, vol. 18, no. 3, pp. 882-887, July 2003.

    :param ppc: matpower-style case data
    :return: results (pypower style), success (flag about PF convergence)
    """
    ppci = ppc

    ppopt = ppoption(ppopt)

    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]
    nbus = bus.shape[0]

    # get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    # depth-first-search bus ordering and generating Direct Load Flow matrix DLF = BCBV * BIBC
    DLF, ppc_bfsw, buses_ordered_bfsw = bibc_bcbv(ppci)


    baseMVA_bfsw, bus_bfsw, gen_bfsw, branch_bfsw = \
        ppc_bfsw["baseMVA"], ppc_bfsw["bus"], ppc_bfsw["gen"], ppc_bfsw["branch"]

    time_start = time() # starting pf calculation timing

    # initialize voltages to flat start and buses with gens to their setpoints
    V0 = np.ones(nbus, dtype=complex)
    V0[gen[:, GEN_BUS].astype(int)] = gen[:, VG]

    Sbus_bfsw = makeSbus(baseMVA_bfsw, bus_bfsw, gen_bfsw)

    # update data matrices with solution
    Ybus_bfsw, Yf_bfsw, Yt_bfsw = makeYbus(baseMVA_bfsw, bus_bfsw, branch_bfsw)
    ## get bus index lists of each type of bus
    ref_bfsw, pv_bfsw, pq_bfsw = bustypes(bus_bfsw, gen_bfsw)

    # #-----  run the power flow  -----
    V_final, success = bfsw(DLF, bus_bfsw, gen_bfsw, branch_bfsw, baseMVA_bfsw, Ybus_bfsw, Sbus_bfsw, V0,
                                   ref_bfsw, pv_bfsw, pq_bfsw, ppopt=ppopt)

    V_final = V_final[np.argsort(buses_ordered_bfsw)]  # return bus voltages in original bus order


    # #----- output results to ppc ------
    ppci["et"] = time() - time_start    # pf time end

    # generate results for original bus ordering
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V_final, ref, pv, pq)

    ppci["success"] = success

    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch

    return ppci, success


