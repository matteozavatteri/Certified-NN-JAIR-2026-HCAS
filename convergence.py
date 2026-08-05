
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import argparse
import math
import conf

import experiments

import properties.P1 as P1
import properties.P2 as P2
import properties.P3 as P3
import properties.P4 as P4
import properties.P5 as P5
import properties.P6 as P6
import properties.P7 as P7
import properties.P8 as P8
import properties.P9 as P9
import properties.P10 as P10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

#plt.rcParams['text.usetex'] = True

def neural_network(hidden_layers, neurons_per_hidden_layer):

	InputNeurons  = conf.InputSIZE
	OutputNeurons = conf.OutputSIZE

	layers = []
	layers.append(nn.Linear(InputNeurons,neurons_per_hidden_layer))
	layers.append(nn.ReLU())
	
	for _ in range(hidden_layers-1):
		layers.append(nn.Linear(neurons_per_hidden_layer,neurons_per_hidden_layer))
		layers.append(nn.ReLU())

	layers.append(nn.Linear(neurons_per_hidden_layer,OutputNeurons))
	network = nn.Sequential(*layers)
	
	return network


import os 
import json
import argparse
from statistics import mean, stdev


def CounterExamples(P,network,EPSILON,DELTA,N=1000):
	
	CE = 0

	if P == 1:	
		x = P1.precondition_enabling_batch(N)
		y = network(x)
		mask = (y[:, conf.coc] + EPSILON <= -0.0433 + DELTA)
		CE = mask.sum().item()

	elif P == 2:
		x = P2.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.coc] <= y[:, conf.ActionsDifferentFromCOC].min(dim=1).values + (DELTA - EPSILON)
		CE = mask.sum().item()

	elif P == 3:
		x = P3.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.ActionsDifferentFromCOC].max(dim=1).values <= y[:, conf.coc] + (DELTA - EPSILON)
		CE = mask.sum().item()

	elif P == 4:
		x = P4.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.ActionsDifferentFromCOC].max(dim=1).values <= y[:, conf.coc] + (DELTA - EPSILON)
		CE = mask.sum().item()

	elif P == 5:
		x = P5.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.sr] <= y[:, conf.ActionsDifferentFromSR].max(dim=1).values + (DELTA - EPSILON)
		CE = mask.sum().item()

	elif P == 6:
		x = P6.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.coc] <= y[:, conf.ActionsDifferentFromCOC].max(dim=1).values + (DELTA - EPSILON)
		CE = mask.sum().item()

	elif P == 7:
		x = P7.precondition_enabling_batch(N)
		y = network(x)
		mask = (y[:, conf.sl] - y[:, conf.ActionsDifferentFromSL_SR].max(dim=1).values >= (EPSILON - DELTA)) | (y[:, conf.sr] - y[:, conf.ActionsDifferentFromSL_SR].max(dim=1).values >= (EPSILON - DELTA))
		CE = mask.sum().item()

	elif P == 8:
		x = P8.precondition_enabling_batch(N)
		y = network(x)
		mask = (y[:, conf.coc] - y[:, conf.ActionsDifferentFromCOC_WL].max(dim=1).values <= (DELTA - EPSILON)) & (y[:, conf.wl] - y[:, conf.ActionsDifferentFromCOC_WL].max(dim=1).values <= (DELTA - EPSILON))
		CE = mask.sum().item()

	elif P == 9:
		x = P9.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.sl] <= y[:, conf.ActionsDifferentFromSL].max(dim=1).values + (DELTA - EPSILON)
		CE = mask.sum().item()
	
	elif P == 10:
		x = P10.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.coc] <= y[:, conf.ActionsDifferentFromCOC].max(dim=1).values + (DELTA - EPSILON)
		CE = mask.sum().item()

	elif P == 12345610:
		ALL = [1,2,3,4,5,6,10]
		for P in ALL:
			CE += CounterExamples(P,network,EPSILON=EPSILON,DELTA=EPSILON,N=int(N/len(ALL)))
	
	elif P == 123456910:
		ALL = [1,2,3,4,5,6,9,10]
		for P in ALL:
			CE += CounterExamples(P,network,EPSILON=EPSILON,DELTA=EPSILON,N=int(N/len(ALL)))

	elif P == 123456810:
		ALL = [1,2,3,4,5,6,8,10]
		for P in ALL:
			CE += CounterExamples(P,network,EPSILON=EPSILON,DELTA=EPSILON,N=int(N/len(ALL)))

	return CE

if not os.path.exists("img"):
	os.mkdir("img")

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--epsilon-delta", default=False, action='store_true', help="compute graphs for ablation")
	
	args = parser.parse_args()

	SEEDS = [100]
	DELTAS = experiments.DELTA   if args.epsilon_delta else [0.01]
	EPSILONS = experiments.EPSILON if args.epsilon_delta else [0.0001]

	EPSILON_DELTA = [(e,d) for e in EPSILONS for d in DELTAS if d > e]
	raw_data = experiments.load_experiment_data()

	print("Computing convergence ...")
	xticks = 11
	x = [10*i for i in range(xticks)]

	DATA = {k : [] for k in raw_data if raw_data[k]['TotalTIME'] is not None}
	for key in DATA:
		P,epsilon,delta = (float(x) for x in key.split('/')[-1].split('-')[:3])
		P = int(P)
		path = 'experiments/{}'.format(key)
		INDEXES = [round(i * (raw_data[key]['CEGISiterations'] - 1) / (xticks - 1)) for i in range(xticks)]
		for index in INDEXES:
			network = neural_network(6,50)
			network.load_state_dict(torch.load('{}/network-{}.pt'.format(path,index+1),map_location=torch.device(device)))
			with torch.inference_mode():
				DATA[key].append(CounterExamples(P,network,epsilon,delta))

	print("done.")

	


	COLORS  = {
			(0.001,0.01)  : 'tab:red',
			(0.0001,0.01)  : 'tab:blue',
			(0.0001,0.001)   : 'tab:green'			}

	MARKERS  = {
			(0.001,0.01)   : 'o',
			(0.0001,0.01)  : 's',
			(0.0001,0.001) : 'X'
			}

	print("Computing plots...")


	for P in [1,2,3,4,5,6,7,8,9,10,12345610,123456810,123456910]:
		print('Property P{}'.format(P))
		fig, ax = plt.subplots()
		plt.xlabel('CEGIS iterations')
		labels = {'xticks' : x, 'xticklabels': x}
		ax.set(**labels)
		for soft in ['','-SA']:
			for (epsilon,delta) in EPSILON_DELTA:
				KEYS = [experiments.key_template.format(
					pra=pra,
					tau=tau,
					property=P, 
				    epsilon=epsilon, 
				    delta=delta,
				    learning_rate=learning_rate, 
				    layers=6,
				    neurons=50,
				    counterexamples=50, 
				    sampling=20, 
				    seed=seed, 
				    soft=soft
				    ) \
					for pra in experiments.PRA \
					for tau in experiments.TAU \
				    for learning_rate in experiments.LEARNING_RATES \
				    for seed in experiments.SEEDS]

				y = []
				for i in range(xticks):
					y.append(mean(DATA[k][i] for k in KEYS if k in DATA))

				epsilonTEXT = '10^{-2}' if epsilon == 0.01 else '10^{-3}' if epsilon == 0.001 else '10^{-4}'
				deltaTEXT   = '10^{-2}' if delta   == 0.01 else '10^{-3}' if delta   == 0.001 else '10^{-4}'
				label = '$\\Delta={}, \\varepsilon={}$'.format(deltaTEXT,epsilonTEXT) if soft == '' else None
				linestyle = 'solid' if soft == '' else 'dashed'
				#ax.set_title('$P_{{{}}}$'.format(P))
				ax.plot(x,y,marker=MARKERS[(epsilon,delta)],color=COLORS[(epsilon,delta)],linewidth=2,linestyle=linestyle,label=label)
				
		ax.xaxis.set_major_formatter(PercentFormatter(100))
		plt.legend()
		plt.grid(True)

		kind = 'baseline' if not args.epsilon_delta else 'epsilon-delta'
		name = 'img/convergence-HCAS-{}-P{}.pdf'.format(kind,P)
		print('Saving {}'.format(name))
		plt.savefig(name)

	print("done")
