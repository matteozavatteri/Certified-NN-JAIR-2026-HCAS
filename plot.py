import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import conf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

plt.rcParams['text.usetex'] = True


def neural_network(hidden_layers, neurons_per_hidden_layer):
	layers = []
	layers.append(nn.Linear(conf.InputSIZE,neurons_per_hidden_layer))
	layers.append(nn.ReLU())
	for _ in range(hidden_layers-1):
		layers.append(nn.Linear(neurons_per_hidden_layer,neurons_per_hidden_layer))
		layers.append(nn.ReLU())
	layers.append(nn.Linear(neurons_per_hidden_layer,conf.OutputSIZE))
	network = nn.Sequential(*layers)
	
	return network

def p_area(p):
	assert (p >= 1 and p <= 10) or p == 12345610
	Elements = []
	rho   = None
	theta = None
	if p == 1:
		rho   = [55947.691, 56000]
		theta = [-3.141592,3.141592]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)
	
	elif p == 2:
		rho   = [55947.691, 56000]
		theta = [-3.141592,3.141592]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)
	
	elif p == 3: # recall that psi >= 3.10
		rho   = [1500, 1800]
		theta = [-0.06,0.06]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)
	
	elif p == 4: # recall that psi == 0
		rho   = [1500, 1800]
		theta = [-0.06,0.06]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)
	
	elif p == 5: # recall that psi == 0
		rho   = [250, 400]
		theta = [0.2,0.4]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)
	
	elif p == 6: # recall that -3.141592 <= psi <= -3.1366
		rho   = [12000, 56000]
		theta = [0.7,3.141592]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)
		theta = [-3.141592,-0.7]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)

	elif p == 7:
		rho   = [0, 56000]
		theta = [-3.141592,3.141592]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)

	elif p == 8:
		rho   = [0, 56000]
		theta = [-3.141592,-0.75*3.141592]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)

	elif p == 9:
		rho   = [2000, 7000]
		theta = [-0.4,-0.14]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)
	
	elif p == 10:
		rho   = [36000,56000]
		theta = [0.7,3.141592]
		rect = plt.Rectangle((min(rho), min(theta)), max(rho)-min(rho), max(theta)-min(theta), color="red", fill=False)
		Elements.append(rect)

	elif p == 12345610:
		for i in [1,2,3,5,10]:
			Elements += p_area(i)
	
	return Elements

def psi_value(p): # normalized
	assert (p >= 1 and p <= 10) or p == 12345610
	psi = None
	
	if p == 1:
		psi = -0.5
	
	elif p == 2:
		psi = -0.5
	
	elif p == 3:
		psi = 0.4934

	elif p == 4:
		psi = 0

	elif p == 5:
		psi = -0.4999

	elif p == 6:
		psi = -0.4999

	elif p == 7:
		psi = -0.4999

	elif p == 8:
		psi = -0.0159

	elif p == 9:
		psi = -0.4999

	elif p == 10:
		psi = -0.4999

	elif p == 12345610:
		psi = 0.4934
	
	assert psi is not None
	return psi

import experiments
import properties.P7 as P7

if __name__ == '__main__':
	N=100
	rho   = torch.linspace(-0.21, 0.8, N)
	theta = torch.linspace(-0.5, 0.5, N)
	r, t = torch.meshgrid(rho, theta, indexing='ij')
	rT = r.reshape(N**2, 1)
	tT = t.reshape(N**2, 1)
	EPSILON_DELTA = [(e,d) for e in experiments.EPSILON for d in experiments.DELTA if d > e]
	raw_data = experiments.load_experiment_data()

	for soft in experiments.SOFT:
		KEYS = [experiments.key_template.format(
			pra=pra,
			tau=tau,
			property=p, 
			epsilon=epsilon,
			delta=delta, 
			learning_rate=1e-3, 
			layers=6,
			neurons=50,
			counterexamples=50, 
			sampling=20, 
			seed=100, 
			soft=soft
			) 
			for pra in experiments.PRA 
			for tau in experiments.TAU
			for p in experiments.PROPERTIES: 
			for epsilon in experiments.EPSILON 
			for delta in experiments.DELTA 
			if delta > epsilon and ( 
				(p in [1,2,3,4,5,6,10,12345610] and pra == 0 and tau == '00') or 
				(p == 7 and pra == 0 and tau == '60') or 
				(p == 8 and pra == 1 and tau == '60') or 
				(p == 9 and pra == 2 and tau == '60') or 
				(p == 123456810 and pra == 1 and tau == '60') or 
				(p == 123456910 and pra == 2 and tau == '60') 
				)]
		
		
		DATA = [raw_data[k]['TotalTIME'] is not None for k in KEYS]
		assert all(DATA)

		from collections import namedtuple
		from maraboupy import Marabou, MarabouCore
		from maraboupy import MarabouUtils
		import tempfile

		args = namedtuple('a', ['epsilon', 'delta','cores'])

		for key in KEYS:
			P,epsilon,delta = (float(x) for x in key.split('/')[-1].split('-')[:3])
			P = int(P)
			args.delta = delta
			args.epsilon = epsilon
			args.cores = 4
			path = 'experiments/{}'.format(key)
			
			INDEXES = [1,raw_data[key]['CEGISiterations']]

			suffix = 'initial'
			for index in INDEXES:
				network = neural_network(6,50)
				network.load_state_dict(torch.load('{}/network-{}.pt'.format(path,index),map_location=torch.device(device)))
				with torch.inference_mode():
					psi = torch.full((N**2, 1), psi_value(P))
					inputs = torch.cat((rT, tT, psi), dim=1)
					outputs = network(inputs)
					advisories = torch.argmax(outputs, dim=1)
					rho_denorm = (56000 * inputs[:, 0]) + 11423.75
					theta_denorm = 2 * 3.141592 * inputs[:, 1]

					# Convert to numpy ONLY for plotting
					rho_np = rho_denorm.numpy()
					theta_np = theta_denorm.numpy()
					advisories_np = advisories.numpy()

					# --- Plot ---
					fig, ax = plt.subplots()
					ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
					labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
					plt.yticks(ticks, labels)

					
					if P == 1:
						mask = outputs[:,0] >= -0.0433
						ax.scatter(rho_np[mask], theta_np[mask], c=conf.COLORS[0], label="COC $\\geq$ -1", rasterized=True)

						mask =  outputs[:,0] < -0.0433
						ax.scatter(rho_np[mask], theta_np[mask], c=conf.COLORS[1], label="COC $\\le$ -1", rasterized=True)

					else:
						ACTIONS = list(range(conf.OutputSIZE))
						for (a, c, l) in zip(ACTIONS, conf.COLORS, ['COC','WL','WR','SL','SR']):
							mask = advisories_np == a
							ax.scatter(rho_np[mask], theta_np[mask], c=c, label=l, rasterized=True)

					Elements = p_area(P)
					for element in Elements:
						ax.add_patch(element)

					plt.xlabel('$\\rho$: range to intruder (ft)')
					plt.ylabel('$\\theta$: bearing angle to intruder (rad)')
					ax.legend(loc='upper right')
					#plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


					outdir = 'img'
					if not os.path.exists(outdir):
						os.mkdir(outdir)

					outdir += '/policy'
					if not os.path.exists(outdir):
						os.mkdir(outdir)
 
					plt.savefig('{DIR}/{key}-{suffix}.pdf'.format(DIR=outdir,key=key.replace('/','-'),suffix=suffix), format="pdf")
					suffix = 'final'
					plt.close()