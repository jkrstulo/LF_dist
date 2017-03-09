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

from pandapower.pypower_extensions.makeYbus_pypower import makeYbus
from pandapower.pypower_extensions.pfsoln import pfsoln
from pandapower.pypower_extensions.bustypes import bustypes
from pandapower.run import LoadflowNotConverged

from pypower.makeSbus import makeSbus
from pypower.ppoption import ppoption


class ConvergenceError(Exception):
    pass



def reindex_bus_ppc(ppc, bus_ind_dict):
    """
    reindexing buses according to dictionary old_bus_index -> new_bus_index
    :param ppc: matpower-type power system
    :param bus_ind_dict:  dictionary for bus reindexing
    :return: ppc with bus indices updated
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
    ppc_bfs = reindex_bus_ppc(ppci, buses_bfs_dict)
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
                break   # TODO: special notice for divergence due to inner iterations for PV nodes


            n_iter_inner += 1

            if np.all(np.abs(dQ) < tol_inner):  # inner loop termination criterion
                inner_loop_converged = True
                V_new = V_inner.copy()

        if not success_inner:
            if verbose:
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
                print("\nFwd-back sweep power flow converged in "
                                 "{0} iterations.\n".format(n_iter))

        V_iter = V_new.copy()  # update iterating complex voltage vector

        # updating injected currents
        Iinj = np.conj(Sbus[mask_root] / V_iter) - Ysh[mask_root] * V_iter

    return V, converged





def _run_bfsw(ppc, ppopt=None):
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


