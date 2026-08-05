# ====================================================================
# Property 7
# ====================================================================
# If vertical separation is large, the network will never advise a strong turn
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
	#print("precondition_enabling_batch P7")
	low  = torch.tensor([-0.2039, -0.4999, -0.4999])
	high = torch.tensor([0.7960, 0.4999, 0.4999])
	x = low + (high - low) * torch.rand(N, low.shape[0])
	return x

def precondition_mask(x):
	return (x[:,conf.rho] >= -0.2039) & (x[:,conf.rho] <= 0.7960) & (x[:,conf.theta] >= -0.4999) & (x[:,conf.theta] <= 0.4999) & (x[:,conf.psi] >= -0.4999) & (x[:,conf.psi] <= 0.4999)

def postcondition_mask(x,y,args):
	return (y[:, conf.sl] - y[:, conf.ActionsDifferentFromSL_SR].max(dim=1).values <= -args.delta) & (y[:, conf.sr] - y[:, conf.ActionsDifferentFromSL_SR].max(dim=1).values <= -args.delta)

def soft_loss(network, N, args):	
	#print("soft_loss P7")
	x = precondition_enabling_batch(N)
	y = network(x)
	VS =  torch.relu(y[:, conf.sl].unsqueeze(1) - y[:, conf.ActionsDifferentFromSL_SR] + args.delta).prod(dim=1) + torch.relu(y[:, conf.sr].unsqueeze(1) - y[:, conf.ActionsDifferentFromSL_SR] + args.delta).prod(dim=1)
	mask = VS > 0
	return VS[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)

def opportunistic_search(network, args, N):
	#print("opportunistic_search P7")
	start = time.time()
	x = precondition_enabling_batch(N)
	with torch.inference_mode():
		y = network(x)

	mask = (y[:, conf.sl] - y[:, conf.ActionsDifferentFromSL_SR].max(dim=1).values >= (args.epsilon - args.delta)) | (y[:, conf.sr] - y[:, conf.ActionsDifferentFromSL_SR].max(dim=1).values >= (args.epsilon - args.delta))
	CEX = x[mask][:args.counterexamples]
	CEY = y[mask][:args.counterexamples]
	OpportunisticSeachTIME = time.time() - start
	return CEX, CEY, OpportunisticSeachTIME

def MIQP_repair_constraints(m, x, y, s, args):
	#print("MIQP_repair_constraints P7")
	a1 = m.addVar(vtype=gp.GRB.BINARY)
	a2 = m.addVar(vtype=gp.GRB.BINARY)
	a3 = m.addVar(vtype=gp.GRB.BINARY)
	m.addGenConstrIndicator(a1, True, y[conf.sr].item() + s[conf.sr] + args.delta <= y[conf.coc].item() + s[conf.coc])
	m.addGenConstrIndicator(a2, True, y[conf.sr].item() + s[conf.sr] + args.delta <= y[conf.wl].item()  + s[conf.wl])
	m.addGenConstrIndicator(a3, True, y[conf.sr].item() + s[conf.sr] + args.delta <= y[conf.wr].item()  + s[conf.wr])
	m.addConstr(a1 + a2 + a3 >= 1)	

	b1 = m.addVar(vtype=gp.GRB.BINARY)
	b2 = m.addVar(vtype=gp.GRB.BINARY)
	b3 = m.addVar(vtype=gp.GRB.BINARY)
	m.addGenConstrIndicator(b1, True, y[conf.sl].item() + s[conf.sl] + args.delta <= y[conf.coc].item() + s[conf.coc])
	m.addGenConstrIndicator(b2, True, y[conf.sl].item() + s[conf.sl] + args.delta <= y[conf.wl].item()  + s[conf.wl])
	m.addGenConstrIndicator(b3, True, y[conf.sl].item() + s[conf.sl] + args.delta <= y[conf.wr].item()  + s[conf.wr])
	m.addConstr(b1 + b2 + b3 >= 1)	

def repair(CEX, CEY, args):
	#print("repair P7")
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
	#print("sampling P7")
	start = time.time()
	N = REPX.shape[0] # actual counterexamples found
	x = REPX.repeat_interleave(args.sampling,0)
	y = REPY.repeat_interleave(args.sampling,0)

	x += 0.01 * torch.randn(args.sampling * N,conf.InputSIZE)
	y += 0.01 * torch.randn(args.sampling * N,conf.OutputSIZE)

	mask = ~precondition_mask(x) | postcondition_mask(x,y,args)
	return x[mask], y[mask], time.time() - start

def formal_verification(network, args):
	#print("formal_verification P7")
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

	Q.setLowerBound(X[conf.rho],  -0.2039)
	Q.setUpperBound(X[conf.rho],  0.7960)
	Q.setLowerBound(X[conf.theta], -0.4999)
	Q.setUpperBound(X[conf.theta], 0.4999)
	Q.setLowerBound(X[conf.psi], -0.4999) 
	Q.setUpperBound(X[conf.psi], 0.4999)  

	# [bigAND (SL - A >= epsilon - delta)] or [bigAND (SR - A >= epsilon - delta)]
	disjunction = []

	conjSL = []
	for act in conf.ActionsDifferentFromSL_SR:
		eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
		eq.addAddend(1, Y[conf.sl])
		eq.addAddend(-1, Y[act])
		eq.setScalar(args.epsilon - args.delta)
		conjSL.append(eq)
	disjunction.append(conjSL)

	conjSR = []
	for act in conf.ActionsDifferentFromSL_SR:
		eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
		eq.addAddend(1, Y[conf.sr])
		eq.addAddend(-1, Y[act])
		eq.setScalar(args.epsilon - args.delta)
		conjSR.append(eq)
	disjunction.append(conjSR)

	Q.addDisjunctionConstraint(disjunction)

	options = Marabou.createOptions(numWorkers=args.cores, verbosity=0)
	sat, vals, stats = Q.solve(verbose=False,options=options)
	CEX = torch.tensor([[vals[x] for x in X]]) if sat == 'sat' else torch.empty(0,conf.InputSIZE)
	CEY = torch.tensor([[vals[y] for y in Y]]) if sat == 'sat' else torch.empty(0,conf.OutputSIZE)
	FormalVerificationTIME = time.time() - start
	return CEX, CEY, FormalVerificationTIME