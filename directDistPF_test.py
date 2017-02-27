

import numpy as np
import timeit as timeit

from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, VM, VA, REF, GS, BS
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, PF, QF, PT, QT, TAP, BR_STATUS
from pypower.idx_gen import GEN_BUS, PG, QG, PMAX, PMIN, QMAX, QMIN, GEN_STATUS, VG


from directDistPF import run_ddlf, bus_reindex


# #
# # load a network case studies

from Rede_BT_100kVA_InovGrid_moreLoads import Rede_BT_100kVA_InovGrid_moreLoads
ppc = Rede_BT_100kVA_InovGrid_moreLoads()

# # test bus ordering
# dict = dict(zip(ppc['bus'][:,0],ppc['bus'][:,0]))
# dict[5]=2
# dict[2]=5

# ppc = bus_reindex(ppc,dict)

# # test loops
nbrch = ppc['branch'].shape[0]
ppc['branch'] = np.concatenate((ppc['branch'], ppc['branch'][nbrch-1:nbrch,:]),axis=0)
ppc['branch'][-1,F_BUS] = 17
ppc['branch'][-1,T_BUS] = 29

ppc['branch'] = np.concatenate((ppc['branch'], ppc['branch'][nbrch-1:nbrch,:]),axis=0)
ppc['branch'][-1,F_BUS] = 30
ppc['branch'][-1,T_BUS] = 33

ppc['branch'] = np.concatenate((ppc['branch'], ppc['branch'][nbrch-1:nbrch,:]),axis=0)
ppc['branch'][-1,F_BUS] = 22
ppc['branch'][-1,T_BUS] = 32


# from pypower.case24_ieee_rts import case24_ieee_rts
# ppc = case24_ieee_rts()






####
## LF by pypower (Newton-Raphson) using PYPOWER
###
import pypower.api as pypow
ppopt = pypow.ppoption(VERBOSE=0, OUT_ALL=0)  #prevents results printing in each iteration

start_time_LF = timeit.default_timer()

results, success = pypow.runpf(ppc, ppopt=ppopt) #ppopt=ppopt

time_LF = timeit.default_timer() - start_time_LF

print ("\n\t N-R converged in {0} s".format(time_LF))
if not success:
    print ("\n powerflow did not converge")




# ####
# Direct load flow solution (a matrix back/fwd sweep)
# ####
start_time = timeit.default_timer()
V_DDLF = run_ddlf(ppc, epsilon=1.e-5)
time_DDLF = timeit.default_timer() - start_time
print ("\n\t Direct PF converged in {0} s".format(time_DDLF))

# print np.abs(V_DDLF)

print ("\n  maximum voltage magnitude error: {0}".format(np.max( np.abs( results['bus'][:, VM]-np.abs(V_DDLF) ) )))
print ("  maximum voltage angle error: {0}".format(np.max(np.abs( results['bus'][:, VA]-np.angle(V_DDLF, deg=True) ))))



print ("\n NR converged in\t{0} s\n DPF converged in\t{1} s".format(time_LF, time_DDLF))