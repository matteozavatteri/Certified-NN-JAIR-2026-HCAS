# ====================================================================
# Property 1 AND
# Property 2 AND
# Property 3 AND
# Property 4 AND
# Property 6 AND
# Property 7 AND
# Property 8 AND
# Property 10
# ====================================================================

import properties.P1 as P1
import properties.P2 as P2
import properties.P3 as P3
import properties.P4 as P4
import properties.P6 as P6
import properties.P7 as P7
import properties.P8 as P8
import properties.P10 as P10


import time
import random
from maraboupy import Marabou, MarabouCore
from maraboupy import MarabouUtils
import math
import gurobipy as gp
import os
import torch
import fm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

import conf

PROPERTIES = [P1,P2,P3,P4,P6,P7,P8,P10]

def soft_loss(network, N, args):
	#print("soft_loss P123467810")
	return sum(P.soft_loss(network, N, args) for P in PROPERTIES)

def opportunistic_search(network, args, N):
	#print("opportunistic_search P123467810")
	CEX = []
	CEY = []
	OpportunisticTime = 0
	for P in PROPERTIES:
		Xi, Yi, ti = P.opportunistic_search(network, args, N)
		CEX.append(Xi)
		CEY.append(Yi)
		OpportunisticTime += ti
	return CEX, CEY, OpportunisticTime

# joint repair
def repair(CEX, CEY, args):
	#print("repair P123467810")
	start = time.time()
	env = gp.Env(empty=True)
	env.setParam('OutputFlag', 0)
	env.start()

	for j in range(len(CEY)):
		REPX = CEX[j]
		REPY = CEY[j]
		PRE_MASKS  = [Pi.precondition_mask(REPX) for Pi in PROPERTIES]

		for i in range(REPY.shape[0]):
			m = gp.Model("repair",env=env)
			S = [m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for _ in range(conf.OutputSIZE)]

			for (P,M) in zip(PROPERTIES,PRE_MASKS):
				if M[i].item():
					P.MIQP_repair_constraints(m, REPX[i], REPY[i], S, args)

			m.setObjective(gp.quicksum(s ** 2 for s in S), gp.GRB.MINIMIZE)
			m.optimize()
			assert m.Status == gp.GRB.OPTIMAL

			for act in range(conf.OutputSIZE):
				REPY[i][act] += S[act].X

			m.dispose()
	RepairTIME = time.time() - start
	return CEX, CEY, RepairTIME


def sampling(REPX, REPY, args):
	#print("sampling P123467810")
	start = time.time()
	N = REPX[0].shape[0] # actual counterexamples found
	x = REPX[0].repeat_interleave(args.sampling,0)
	y = REPY[0].repeat_interleave(args.sampling,0)

	x += 0.01 * torch.randn(args.sampling * N,conf.InputSIZE)
	y += 0.01 * torch.randn(args.sampling * N,conf.OutputSIZE)

	SAT_MASKS = [~P.precondition_mask(x) | P.postcondition_mask(x, y,args) for P in PROPERTIES]
	mask = torch.all(torch.stack(tuple(SAT_MASKS)),dim=0)
	return [x[mask]], [y[mask]], time.time() - start


def formal_verification(network, args):
	#print("formal_verification P123467810")
	CEX = []
	CEY = []
	FormalVerificationTIME = 0
	for P in PROPERTIES:
		Xi, Yi, ti = P.formal_verification(network, args)
		CEX.append(Xi)
		CEY.append(Yi)
		FormalVerificationTIME += ti
	return CEX, CEY, FormalVerificationTIME