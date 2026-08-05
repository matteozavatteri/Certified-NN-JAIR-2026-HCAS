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


def polar_perimeter(ax, rho_min, rho_max, theta_min, theta_max, **kwargs):
	# Smooth sampling
	theta = np.linspace(theta_min, theta_max, 300)
	rho = np.linspace(rho_min, rho_max, 300)

	# --- arcs ---
	ax.plot(theta, np.full_like(theta, rho_min), **kwargs)  # inner arc
	ax.plot(theta, np.full_like(theta, rho_max), **kwargs)  # outer arc

	# --- radial edges ---
	ax.plot(np.full_like(rho, theta_min), rho, **kwargs)  # left edge
	ax.plot(np.full_like(rho, theta_max), rho, **kwargs)  # right edge



def p_area(p):
	assert p in experiments.PROPERTIES
	Elements = []
	rho   = None
	theta = None
	if p == 1:
		rho   = [55947.691, 56000]
		theta = [-3.141592,3.141592]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))
	
	elif p == 2:
		rho   = [55947.691, 56000]
		theta = [-3.141592,3.141592]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))
	
	elif p == 3: # recall that psi >= 3.10
		rho   = [1500, 1800]
		theta = [-0.06,0.06]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))
	
	elif p == 4: # recall that psi == 0
		rho   = [1500, 1800]
		theta = [-0.06,0.06]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))
	
	elif p == 5: # recall that psi == 0
		rho   = [250, 400]
		theta = [0.2,0.4]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))
	
	elif p == 6: # recall that -3.141592 <= psi <= -3.1366
		rho   = [12000, 56000]
		theta = [0.7,3.141592]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))
		theta = [-3.141592,-0.7]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))

	elif p == 7:
		rho   = [0, 56000]
		theta = [-3.141592,3.141592]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))

	elif p == 8:
		rho   = [0, 56000]
		theta = [-3.141592,-0.75*3.141592]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))

	elif p == 9:
		rho   = [2000, 7000]
		theta = [-0.4,-0.14]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))
	
	elif p == 10:
		rho   = [36000,56000]
		theta = [0.7,3.141592]
		Elements.append((min(rho),max(rho),min(theta),max(theta)))

	elif p == 12345610:
		for i in [1,2,3,4,5,6,10]:
			Elements += p_area(i)
	
	elif p == 123456810:
		for i in [1,2,3,4,5,6,8,10]:
			Elements += p_area(i)

	elif p == 123456910:
		for i in [1,2,3,4,5,6,9,10]:
			Elements += p_area(i)
	
	return Elements

def psi_value(p): # normalized
	assert p in experiments.PROPERTIES
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

	else:
		psi = 0
	
	assert psi is not None
	return psi

import experiments
import properties.P7 as P7

if __name__ == '__main__':
	N=200
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
			for p in [1,2,3,4,5,6,7,8,9,10]
			for epsilon in experiments.EPSILON 
			for delta in experiments.DELTA 
			if delta > epsilon and ( 
				(p in [1,2,3,4,5,6,10] and pra == 0 and tau == '00') or 
				(p == 7 and pra == 0 and tau == '60') or 
				(p == 8 and pra == 1 and tau == '60') or 
				(p == 9 and pra == 2 and tau == '60') or 
				(p == 123467810 and pra == 1 and tau == '60') or 
				(p == 123456810 and pra == 1 and tau == '60') or 
				(p == 123456910 and pra == 2 and tau == '60') 
				)]
		
		
		DATA = [raw_data[k]['TotalTIME'] is not None for k in KEYS]
		assert all(DATA)

		from collections import namedtuple
		from maraboupy import Marabou, MarabouCore
		from maraboupy import MarabouUtils
		import tempfile

		for key in KEYS:
			P,epsilon,delta = (float(x) for x in key.split('/')[-1].split('-')[:3])
			P = int(P)
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
					fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
					

					
					if P == 1:
						mask = outputs[:,0] >= -0.0433
						ax.scatter(theta_np[mask], rho_np[mask], c=conf.COLORS[0], label="COC $\\geq$ -1", rasterized=True)

						mask =  outputs[:,0] < -0.0433
						ax.scatter(theta_np[mask], rho_np[mask], c=conf.COLORS[1], label="COC $\\le$ -1", rasterized=True)

					else:
						ACTIONS = list(range(conf.OutputSIZE))
						for (a, c, l) in zip(ACTIONS, conf.COLORS, ['COC','WL','WR','SL','SR']):
							mask = advisories_np == a
							ax.scatter(theta_np[mask], rho_np[mask], c=c, label=l, rasterized=True)

					Elements = p_area(P)
					for (minRHO, maxRHO, minTHETA, maxTHETA) in Elements:
						polar_perimeter(
							ax,
							rho_min=minRHO,
							rho_max=maxRHO,
							theta_min=minTHETA,
							theta_max=maxTHETA,
							color='red',
							linewidth=1
						)

					ax.legend(loc='best')
					ax.set_thetagrids(
						[0, 45, 90, 135, 180, 225, 270, 315],
						labels=['$0$', '$\\frac{\\pi}{4}$', '$\\frac{\\pi}{2}$', '$\\frac{3}{4}\\pi$', '$\\pi$', '$\\frac{5}{4}\\pi$', '$\\frac{3}{2}\\pi$', '$\\frac{}{4}\\pi$']
					)
					ax.set_rticks([10000, 20000, 30000, 40000, 50000])  # Fewer radial ticks
					ax.set_yticklabels(['$10K$ ft', '$20K$ ft', '$30K$ ft', '$40K$ ft', '$50K$ ft'],color='white')
					ax.set_rlabel_position(90)
					outdir = 'img'
					if not os.path.exists(outdir):
						os.mkdir(outdir)

					outdir += '/policy'
					if not os.path.exists(outdir):
						os.mkdir(outdir)
 
					plt.savefig('{DIR}/{key}-{suffix}.pdf'.format(DIR=outdir,key=key.replace('/','-'),suffix=suffix), format="pdf")
					suffix = 'final'
					plt.close()