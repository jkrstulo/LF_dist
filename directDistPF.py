

import numpy as np
import scipy as sp
import networkx as nx

from time import time
# import timeit.default_timer as time

from scipy.sparse import issparse, csr_matrix as sparse

from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, VM, VA, REF, GS, BS
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, PF, QF, PT, QT, TAP, BR_STATUS
from pypower.idx_gen import GEN_BUS, PG, QG, PMAX, PMIN, QMAX, QMIN, GEN_STATUS, VG

from pypower.pfsoln import pfsoln
from pypower.makeYbus import makeYbus
from pypower.bustypes import bustypes

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
    bus_ordered_list= [start]
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
            elif (parent, child) not in visited_branches:    # loop detected
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
    branches[:, F_BUS:T_BUS + 1] = np.sort(branches[:, F_BUS:T_BUS + 1], axis=1)    # sort in order to T_BUS > F_BUS

    generators[:, GEN_BUS] = [bus_ind_dict[bus] for bus in generators[:, GEN_BUS]]

    # sort buses, branches and generators according to new numbering
    ppc_bfs['bus'] = buses[np.argsort(buses[:, BUS_I])]
    ppc_bfs['branch'] = branches[np.lexsort((branches[:, T_BUS],branches[:, F_BUS]))]
    ppc_bfs['gen'] = generators[np.argsort(generators[:, GEN_BUS])]

    return ppc_bfs



def bibc_bcbv(ppc, branches_loop):
    """
    creates 2 matrices required for direct LF:
        BIBC - Bus Injection to Branch-Current
        BCBV - Branch-Current to Bus-Voltage
    :param ppc: matpower-type power system dictionary
    :param branches_loop: branches that are defined as loop closers (list of tuples)
    :return: matrices BIBC and BCBV in a dense form
    """

    buses = ppc['bus'][:, BUS_I]
    nbrch = ppc['branch'].shape[0]
    nbus = buses.shape[0]

    mask_root = ~ (ppc['bus'][:, BUS_TYPE] == 3)
    root_bus = buses[~mask_root][0]

    # dictionaries with bus/branch indices
    bus_ind_dict = dict(zip(buses[mask_root],range(nbus-1))) #dictionary bus_name-> bus_ind
    bus_ind_dict[root_bus] = -1
    brch_ind_dict = dict(zip(zip(ppc['branch'][:,F_BUS],ppc['branch'][:,T_BUS]),range(nbrch)))#dictionary brch_name-> brch_ind

    # list of branches in the radial network, i.e. without branches that close loops
    branches_radial = zip(ppc['branch'][:, F_BUS], ppc['branch'][:, T_BUS])
    for brch_l in branches_loop:
        branches_radial.remove(brch_l)

    #reorder branches so that loop-forming branches are at the end

    Z_brchs = []
    nloops = len(branches_loop)

    BIBC = np.zeros((nbrch-nloops,nbus-1))
    BCBV = np.zeros((nbus-1,nbrch-nloops),dtype=complex)

    for br_i,branch in enumerate(branches_radial):
        bus_f = branch[0]
        bus_t = branch[1]
        bus_f_i = bus_ind_dict[bus_f]
        bus_t_i = bus_ind_dict[bus_t]
        brch_ind = brch_ind_dict[branch]
        if bus_f != root_bus and bus_t != root_bus:
            BIBC[:,bus_t_i] = BIBC[:,bus_f_i].copy()
            BCBV[bus_t_i,:] = BCBV[bus_f_i,:].copy()

        if branch[T_BUS] != root_bus:
            BIBC[br_i,bus_t_i] += 1
            Z = (ppc['branch'][brch_ind,BR_R] + 1j * ppc['branch'][brch_ind,BR_X])
            BCBV[bus_t_i,br_i] = Z
            Z_brchs.append(Z)

    for br_i,branch in enumerate(branches_loop):
        bus_f = branch[0]
        bus_t = branch[1]
        bus_f_i = bus_ind_dict[bus_f]
        bus_t_i = bus_ind_dict[bus_t]
        brch_ind = brch_ind_dict[branch]

        BIBC = np.vstack((BIBC,np.zeros(BIBC.shape[1]))) #add row
        BIBC = np.hstack((BIBC,np.zeros((BIBC.shape[0],1)))) #add column
        last_col_i = BIBC.shape[1] - 1  #last column index
        BIBC[:,last_col_i] = BIBC[:,bus_f_i] - BIBC[:,bus_t_i]
        BIBC[last_col_i,last_col_i] += 1

        BCBV = np.vstack((BCBV,np.zeros(BCBV.shape[1]))) #add row
        BCBV = np.hstack((BCBV,np.zeros((BCBV.shape[0],1)))) #add column
        Z = (ppc['branch'][brch_ind,BR_R] + 1j * ppc['branch'][brch_ind,BR_X])
        Z_brchs.append(Z)
        BCBV[BCBV.shape[0]-1,:] = np.array(Z_brchs) * BIBC[:,last_col_i]  #KVL for the loop

    return BIBC, BCBV




def runpf_bibc_bcbv(ppc, BIBC, BCBV, epsilon = 1.e-5, maxiter = 50):
    """
    distribution power flow solution according to [1]
    :param ppc: power system matpower-type
    :param BIBC: bus-injection to branch-current matrix
    :param BCBV: branch-currentn to bus-voltage matrix
    :param epsilon: termination criterion
    :param maxiter: maximum number of iterations before raising divergence error
    :return: power flow result
    :References:
    [1] Jen-Hao Teng, "A Direct Approach for Distribution System Load Flow Solutions"
    """

    buses = ppc['bus']
    baseMVA = ppc['baseMVA']
    branches = ppc['branch']
    gens = ppc['gen']

    nbus = buses.shape[0]
    nbrch = branches.shape[0]
    ngen = gens.shape[0]

    bus_ind_dict = dict(zip(buses[:, BUS_I], range(nbus)))

    mask_root = ~ (buses[:,BUS_TYPE] == 3)  # mask for eliminating root bus
    root_bus = buses[~mask_root, BUS_I][0]
    root_bus_i = bus_ind_dict[root_bus]

    bus_ind_mask_dict = dict(zip(buses[mask_root, BUS_I], range(nbus - 1)))

    # initialize voltages to flat start
    V = np.ones(nbus, dtype=complex)

    # detect PV buses
    busPV_ind = np.where(buses[:,BUS_TYPE] == 2)[0]
    busPV = buses[busPV_ind,BUS_I]
    busPV_ind_mask = [bus_ind_mask_dict[bus] for bus in busPV]
    genPV_ind = np.where(np.in1d(gens[:,GEN_BUS], busPV))[0].flatten()

    # compute shunt admittance
    # if Psh is the real power consumed by the shunt at V = 1.0 p.u. and Qsh is the reactive power injected by
    # the shunt at V = 1.0 p.u. then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    # vector of shunt admittances
    Ysh = (buses[:, GS] + 1j * buses[:, BS]) / baseMVA

    # Line charging susceptance BR_B is also added as shunt admittance:
    Ysh[branches[:, F_BUS].astype(int) - 1] += 1j * branches[:, BR_B] / 2
    Ysh[branches[:, T_BUS].astype(int) - 1] += 1j * branches[:, BR_B] / 2


    # generators in ON status
    genson = np.flatnonzero(gens[:, GEN_STATUS] > 0)
    # connection matrix, element i, j is 1 if gen on(j) at bus i is ON
    Cgen = sparse((np.ones(ngen), (gens[genson, GEN_BUS]-1, range(ngen))), (nbus, ngen))
    # power injected by gens in p.u.
    Sgen = Cgen * (gens[genson, PG] + 1j * gens[genson, QG]) / baseMVA
    # power injected by loads in p.u.
    Sload = (buses[:, PD] + 1j * buses[:, QD]) / baseMVA


    V_iter = V[mask_root].copy() # initial voltage vector without root bus
    Iinj = np.conj((Sgen - Sload)[mask_root] / V_iter) - Ysh[mask_root] * V_iter   # Initial current injections

    n_iter = 0
    term_criterion = False
    success = 1

    while not term_criterion:
        n_iter_inner = 0
        n_iter += 1
        if n_iter > maxiter:
            success = 0
            break
            # raise ConvergenceError("\n\t PF does not converge after {0} iterations!".format(n_iter-1))

        if BCBV.shape[0] > nbus-1:  # if nbrch > nbus - 1 -> network has loops
            DLF_loop = sp.dot(BCBV, BIBC)
            A = DLF_loop[0:nbus-1, 0:nbus-1]
            M = DLF_loop[nbus-1:, 0:nbus-1]
            N = DLF_loop[nbus-1:, nbus-1:]
            #TODO: use more efficient inversion !?
            DLF = A - sp.dot(sp.dot(M.T, sp.linalg.inv(N)), M)  # Krons Reduction
        else:   # no loops -> radial network
            DLF = sp.dot(BCBV, BIBC)

        deltaV = sp.dot(DLF, Iinj)
        V_new = np.ones(nbus - 1) * V[root_bus_i] + deltaV

        # ##
        # inner loop for considering PV buses
        inner_loop_citerion = False
        V_inner = V_new.copy()

        success_inner = 1
        while not inner_loop_citerion and len(busPV) > 0:
            Vmis = (np.abs(gens[genPV_ind,VG])) **2 - (np.abs(V_inner[busPV_ind_mask])) **2
            dQ = Vmis / (2 * DLF[busPV_ind_mask,busPV_ind_mask].imag)

            gens[genPV_ind,QG] += dQ
            if (gens[genPV_ind,QG] < gens[genPV_ind,QMIN]).any():
                Qviol_ind = np.argwhere((gens[genPV_ind,QG] < gens[genPV_ind,QMIN])).flatten()
                gens[genPV_ind[Qviol_ind],QG] = gens[genPV_ind[Qviol_ind],QMIN]
            elif (gens[genPV_ind,QG] > gens[genPV_ind,QMAX]).any():
                Qviol_ind = np.argwhere((gens[genPV_ind, QG] < gens[genPV_ind, QMIN])).flatten()
                gens[genPV_ind[Qviol_ind],QG] = gens[genPV_ind[Qviol_ind],QMAX]

            Sgen = Cgen * (gens[genson, PG] + 1j * gens[genson, QG]) / baseMVA
            Iinj = np.conj((Sgen - Sload)[mask_root]/V_inner) - Ysh[mask_root] * V_inner
            deltaV = sp.dot(DLF,Iinj)
            V_inner = np.ones(nbus-1)*V[root_bus_i] + deltaV

            if n_iter_inner > 20 or np.any(np.abs(V_inner[busPV_ind_mask]) > 2):
                success_inner = 0
                break
                # raise ConvergenceError("\n\t inner iterations (PV nodes) did not converge!!!")

            n_iter_inner += 1

            if np.all(np.abs(dQ) < 1.e-2):   # inner loop termination criterion
                inner_loop_citerion = True
                V_new = V_inner.copy()

        if success_inner == 0:
            break

        # testing termination criterion -
        if np.max(np.abs(V_new - V_iter)) < epsilon:
            term_criterion = True

        V_iter = V_new.copy()   # update iterating complex voltage vector

        # updating injected currents
        Iinj = np.conj((Sgen - Sload)[mask_root]/V_iter) - Ysh[mask_root] * V_iter

    V_final = np.insert(V_iter, root_bus_i, [(1. + 0.j)])  # inserting back the root bus


    return V_final, success


def run_ddlf(ppc, epsilon = 1.e-5, maxiter = 50):
    """
    runs direct distribution load flow
    :param ppc: mat
    :return:
    """
    time_start = time()

    ppci = ppc

    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]

    branches = branch[:, F_BUS:T_BUS + 1]

    mask_root = bus[:,BUS_TYPE] == 3 # reference bus is assumed as root bus for a radial network
    root_bus = bus[mask_root, BUS_I][0]
    nbus = bus.shape[0]

    # power system graph created from list of branches as a dictionary
    system_graph = graph_dict(branches)

    # ###
    # bus ordering according to BFS search

    # TODO check if bfs bus ordering is necessary for the direct power flow
    buses_ordered_bfs, edges_ordered_bfs, branches_loops = bfs_edges(system_graph, root_bus)
    buses_bfs_dict = dict(zip(buses_ordered_bfs, range(1, nbus + 1)))   # old to new bus names
    ppc_bfs = bus_reindex(ppci, buses_bfs_dict)

    branches_loops_bfs = []
    if len(branches_loops) > 0:
        print ("  {0} loops detected\n  following branches are cut to obtain radial network: {1}".format(
            len(branches_loops), branches_loops) )
        branches_loops_bfs = [(buses_bfs_dict[fbus],buses_bfs_dict[tbus]) for fbus,tbus in branches_loops]
        branches_loops_bfs = np.sort(branches_loops_bfs, axis=1)  # sort in order to T_BUS > F_BUS
        branches_loops_bfs = list(zip(branches_loops_bfs[:,0],branches_loops_bfs[:,1]))


    # generating two matrices for direct LF - BIBC and BCBV
    BIBC, BCBV = bibc_bcbv(ppc_bfs, branches_loops_bfs)



    # ###
    # LF initialization and calculation
    V_final, success = runpf_bibc_bcbv(ppc_bfs, BIBC, BCBV, epsilon = 1.e-5, maxiter = 50)
    V_final = V_final[np.argsort(buses_ordered_bfs)]    # return bus voltages in original bus order


    # update data matrices with solution
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    ## get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V_final, ref, pv, pq)


    ppci["et"] = time() - time_start
    ppci["success"] = success

    ##-----  output results  -----
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    results = ppci

    return results, success
