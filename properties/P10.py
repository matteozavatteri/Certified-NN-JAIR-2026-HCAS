# ====================================================================
# Property 10
# ====================================================================
# For a far away intruder, the network advises clear of conflict.
# ====================================================================
import time
import random
from maraboupy import Marabou, MarabouCore
from maraboupy import MarabouUtils
import tempfile
import gurobipy as gp

import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

import conf


def precondition_enabling_batch(N):
	#print("precondition_enabling_batch P10")
	low  = torch.tensor([0.4388, 0.1114, -0.4999])
	high = torch.tensor([0.7960, 0.4999, -0.4983])
	x = low + (high - low) * torch.rand(N, low.shape[0])
	return x

def precondition_mask(x):
	return (0.4388 <= x[:,conf.rho]) & (x[:,conf.rho] <= 0.7960) & (0.1114 <= x[:,conf.theta]) & (x[:,conf.theta] <= 0.4999) & (-0.4999 <= x[:,conf.psi]) & (x[:,conf.psi] <= -0.4983)

def postcondition_mask(x,y,args):
	return y[:, conf.coc] >= y[:, conf.ActionsDifferentFromCOC].max(dim=1).values + args.delta

def soft_loss(network, N, args):	
	#print("soft_loss P10")
	x = precondition_enabling_batch(N)
	y = network(x)
	VS = torch.relu(y[:, conf.ActionsDifferentFromCOC] -  y[:, conf.coc].unsqueeze(1) + args.delta).sum(dim=1)
	mask = VS > 0
	return VS[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)

def opportunistic_search(network, args, N):
	#print("opportunistic_search P10")
	start = time.time()
	x = precondition_enabling_batch(N)
	with torch.inference_mode():
		y = network(x)

	mask = y[:, conf.coc] <= y[:, conf.ActionsDifferentFromCOC].max(dim=1).values + (args.delta - args.epsilon)
	CEX = x[mask][:args.counterexamples]
	CEY = y[mask][:args.counterexamples]
	OpportunisticSeachTIME = time.time() - start
	return CEX, CEY, OpportunisticSeachTIME

def MIQP_repair_constraints(m, x, y, s, args):
	#print("MIQP_repair_constraints P10")
	m.addConstr(y[conf.coc].item() + s[conf.coc] >= y[conf.wl].item() + s[conf.wl] + args.delta)
	m.addConstr(y[conf.coc].item() + s[conf.coc] >= y[conf.wr].item() + s[conf.wr] + args.delta)
	m.addConstr(y[conf.coc].item() + s[conf.coc] >= y[conf.sl].item() + s[conf.sl] + args.delta)
	m.addConstr(y[conf.coc].item() + s[conf.coc] >= y[conf.sr].item() + s[conf.sr] + args.delta)

def repair(CEX, CEY, args):
	#print("repair P10")
	start = time.time()
	env = gp.Env(empty=True)
	env.setParam('OutputFlag', 0)
	env.start()

	for i in range(CEY.shape[0]):
		m = gp.Model("repair",env=env)
		S = [m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for _ in range(conf.OutputSIZE)]
		MIQP_repair_constraints(m, CEX[i], CEY[i], S, args)
		m.setObjective(gp.quicksum(s ** 2 for s in S), gp.GRB.MINIMIZE)
		m.optimize()
		assert m.Status == gp.GRB.OPTIMAL

		for act in range(conf.OutputSIZE):
			CEY[i][act] += S[act].X

		m.dispose()
	RepairTIME = time.time() - start
	return CEX, CEY, RepairTIME


def sampling(REPX, REPY, args):
	#print("sampling P10")
	start = time.time()
	N = REPX.shape[0] # actual counterexamples found
	x = REPX.repeat_interleave(args.sampling,0)
	y = REPY.repeat_interleave(args.sampling,0)

	x += 0.01 * torch.randn(args.sampling * N,conf.InputSIZE)
	y += 0.01 * torch.randn(args.sampling * N,conf.OutputSIZE)
	
	mask = ~precondition_mask(x) | postcondition_mask(x,y,args)
	return x[mask], y[mask], time.time() - start

def formal_verification(network, args):
	#print("formal_verification P10")
	start = time.time()
	Q = None
	with tempfile.TemporaryDirectory() as tmpdir:
		network.eval()
		x = torch.zeros(1, conf.InputSIZE)
		onnx_path = os.path.join(tmpdir, "model.onnx")
		with torch.inference_mode():		
			torch.onnx.export(network, x, onnx_path)
		Q = Marabou.read_onnx(onnx_path)

	assert Q is not None

	X, Y = Q.inputVars[0][0], Q.outputVars[0][0]

	Q.setLowerBound(X[conf.rho],  0.4388)
	Q.setUpperBound(X[conf.rho],  0.7960)
	Q.setLowerBound(X[conf.theta], 0.1114)
	Q.setUpperBound(X[conf.theta], 0.4999)
	Q.setLowerBound(X[conf.psi], -0.4999) 
	Q.setUpperBound(X[conf.psi], -0.4983)  

	# bigOR (COC - A <= DELTA - epsilon)
	disjunction = []
	for act in conf.ActionsDifferentFromCOC:
		eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
		eq.addAddend(1, Y[conf.coc])
		eq.addAddend(-1, Y[act])
		eq.setScalar(args.delta - args.epsilon)
		disjunction.append([eq])
	Q.addDisjunctionConstraint(disjunction)

	options = Marabou.createOptions(numWorkers=args.cores, verbosity=0)
	sat, vals, stats = Q.solve(verbose=False,options=options)
	CEX = torch.tensor([[vals[x] for x in X]]) if sat == 'sat' else torch.empty(0,conf.InputSIZE)
	CEY = torch.tensor([[vals[y] for y in Y]]) if sat == 'sat' else torch.empty(0,conf.OutputSIZE)
	FormalVerificationTIME = time.time() - start
	return CEX, CEY, FormalVerificationTIME