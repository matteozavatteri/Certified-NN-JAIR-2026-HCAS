import torch
import torch.nn as nn

import argparse
from statistics import mean

import conf
import os
import time
import shutil
import json
import sys
import random
import pandas as pd
import conf
import training
import fm
import h5py

#device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("Using device: {}".format(device))
torch.set_default_device(device)

def load_dataset(dataset):
	f = h5py.File(dataset, 'r')
	return [X.tolist() + y.tolist() for (X,y) in zip(f['X'],f['y'])]

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

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("property", type=int, default=1, help="1,2,3,4,5,6,7,8,9,10,123456810,123467810,123456910,12345610 (last four for multiple properties)")
	parser.add_argument("--learning-rate", type=float, default=0.001, help="learning rate (0.01, 0.001, 0.0001; default: 0.001)")
	parser.add_argument("--epochs", type=int, default=1000, help="number of epochs (default: 100)")
	parser.add_argument("--early-stopping", type=int, default=50, help="early stopping (default: 100)")
	parser.add_argument("--hidden-layers", type=int, default=6, help="number of hidden layers (default: 6)")
	parser.add_argument("--neurons-per-hidden-layer", type=int, default=50, help="number of neurons per hidden layer (default: 50)")
	parser.add_argument("--seed", type=int, default=1, help="random seed for shuffling dataset (default: 1)")
	parser.add_argument("--counterexamples", type=int, default=1, help="max number of counterexamples")
	parser.add_argument("--sampling", type=int, default=20, help="sample more points around the repaired counterexample")
	parser.add_argument("--retain", default=False, action='store_true', help="retain the current network (last CEGIS iteration)")
	parser.add_argument("--force", default=False, action='store_true', help="delete previous stat files and do the experiment again")
	parser.add_argument("--soft-acceleration", default=False, action='store_true', help="apply soft constraint acceleration")
	parser.add_argument("--keep", default=False, action='store_true', help="always keep the current network (last CEGIS iteration)")
	parser.add_argument("--pra", default='0', help="previous advisory (0(COC), 1(WL), 2(WR), 3(SL),4 (SR); default: 0)")
	parser.add_argument("--tau", default='00', help="time to loss of vertical separation (00,05,10,15,20,30,40,60: default: 0)")
	parser.add_argument("--delta", type=float, default=0.01, help="learning rate (0.01, 0.001, 0.0001; default: 0.01)")
	parser.add_argument("--epsilon", type=float, default=0.0001, help="learning rate (0.01, 0.001, 0.0001; default: 0.0001)")
	parser.add_argument("--cores", type=int, default=4, help="number of physical CPU cores to use where appropriate (e.g., formal verification)")
	args = parser.parse_args()

	assert args.property in [1,2,3,4,5,6,7,8,9,10,123456810,123467810,123456910,12345610]

	assert args.learning_rate >= 0.0001
	assert args.epochs >= 1
	assert args.early_stopping >= 1 and args.early_stopping <= args.epochs
	assert args.hidden_layers >= 1
	assert args.neurons_per_hidden_layer >= 1
	assert args.counterexamples > 0
	assert args.sampling >= 0
	assert args.pra in ['{}'.format(i) for i in range(5)]
	assert args.tau in ['00','05','10','15','20','30','40','60']
	assert args.delta >= 0
	assert args.epsilon >= 0
	assert args.cores >= 1

	assert args.property != 7 or args.tau == '60'
	assert args.property != 8 or (args.pra == '1' and args.tau == '60')
	assert args.property != 9 or (args.pra == '2')
	
	assert args.property != 123467810 or (args.pra == '1' and args.tau == '60')
	assert args.property != 123456810 or (args.pra == '1' and args.tau == '60')
	assert args.property != 123456910 or (args.pra == '2' and args.tau == '60') 

	if not os.path.exists('experiments'):
		os.mkdir('experiments')
	
	COMBO = 'pra{pra}-tau{tau}/'.format(
		pra=args.pra,
		tau=args.tau
		)

	if not os.path.exists('experiments/{}'.format(COMBO)):
		os.mkdir('experiments/{}'.format(COMBO))

	EXPERIMENT = '{property}-{epsilon}-{delta}-{lr}-{layers}x{neurons}-{counterexamples}-{sampling}-{seed}-{keep}{soft}'.format(
		property=args.property,
		epsilon=args.epsilon,
		delta=args.delta,
		lr=args.learning_rate,
		layers=args.hidden_layers,
		neurons=args.neurons_per_hidden_layer,
		counterexamples=args.counterexamples,
		sampling=args.sampling,
		seed=args.seed,
		keep='K' if args.keep else 'L',
		soft='-SA' if args.soft_acceleration else ''
		)
	
	if os.path.exists('experiments/{}/{}'.format(COMBO,EXPERIMENT)):
		if not args.force:
			exit('experiment exists, use --force to overwrite and do the experiment again')
		else:
			#print('overwriting experiment {}'.format(EXPERIMENT))
			shutil.rmtree('experiments/{}/{}'.format(COMBO,EXPERIMENT))
	os.mkdir('experiments/{}/{}'.format(COMBO,EXPERIMENT))
	
	START = time.time()

	dataset = load_dataset('HCAS_polar_networks/HCAS_polar_TrainingData_v6_pra{}_tau{}.h5'.format(args.pra,args.tau))

	random.seed(args.seed)
	random.shuffle(dataset)

	# partition
	Training   = dataset[:43000]
	Validation = dataset[43000:48396]
	Test       = dataset[48396:]
	
	assert (len(Training) + len(Validation) + len(Test)) == len(dataset)

	X_train = torch.tensor([i[:conf.InputSIZE] for i in Training], dtype=torch.float32)
	y_train = torch.tensor([i[conf.InputSIZE:] for i in Training], dtype=torch.float32)

	X_valid = torch.tensor([i[:conf.InputSIZE] for i in Validation], dtype=torch.float32)
	y_valid = torch.tensor([i[conf.InputSIZE:] for i in Validation], dtype=torch.float32)
	
	X_test  = torch.tensor([i[:conf.InputSIZE] for i in Test], dtype=torch.float32)
	y_test  = torch.tensor([i[conf.InputSIZE:] for i in Test], dtype=torch.float32)


	stats = {
		'OriginalInputs' : X_train.shape[0],
		'TrainingTIME' : [],
		'OpportunisticSeachTIME' : [],
		'FormalVerificationTIME' : [],
		'RepairTIME' : [],
		'SamplingTIME' : [],
		'LossOnValidation' : [],
		'LossOnTest' : [],
		'OpportunisticCEX' : [],
		'FormalVerificationCEX' : [],
		'NewPairs' : [],
		'TotalTIME' : None
	}

	loss_fn = training.custom_MSELoss
	lossT = None
	cegis_iteration = 0

	network = neural_network(args.hidden_layers, args.neurons_per_hidden_layer)
	
	while True:				
		cegis_iteration += 1	
		# print("iteration={}".format(cegis_iteration))
		if cegis_iteration > 1 and (not args.keep):
			network = neural_network(args.hidden_layers, args.neurons_per_hidden_layer) # new one
		
		soft = args.property if args.soft_acceleration else None
		
		network, lossV, TrainingTIME = training.train(network, X_train, y_train, X_valid, y_valid, args)			

		stats['TrainingTIME'].append(TrainingTIME)
		stats['LossOnValidation'].append(lossV)

		# compute loss on test
		with torch.no_grad():
			y_pred = network(X_test)
			lossT = loss_fn(y_pred, y_test).item()
			stats['LossOnTest'].append(lossT)

		stats['OpportunisticSeachTIME'].append(0)
		stats['OpportunisticCEX'].append(0)
		stats['FormalVerificationTIME'].append(0)
		stats['FormalVerificationCEX'].append(0)
		stats['RepairTIME'].append(0)
		stats['SamplingTIME'].append(0)
		stats['NewPairs'].append(0)
		stats['CEGISiterations'] = cegis_iteration

		NumberOfCounterExamples = 0
		if args.counterexamples > 0:
			CEX, CEY, OpportunisticTime = fm.opportunistic_search_proxy(network, args, N=1000)
			NumberOfCounterExamples = CEX.shape[0] if isinstance(CEX, torch.Tensor) else sum(cex.shape[0] for cex in CEX)
			stats['OpportunisticSeachTIME'][-1] = OpportunisticTime
			stats['OpportunisticCEX'][-1] = NumberOfCounterExamples

			if NumberOfCounterExamples == 0:
				CEX, CEY, FormalVerificationTIME = fm.formal_verification_proxy(network, args)
				NumberOfCounterExamples = CEX.shape[0] if isinstance(CEX, torch.Tensor) else sum(cex.shape[0] for cex in CEX)
				stats['FormalVerificationTIME'][-1] = FormalVerificationTIME
				stats['FormalVerificationCEX'][-1] =  NumberOfCounterExamples
				
			if NumberOfCounterExamples > 0:
				REPX, REPY, RepairTIME = fm.repair_proxy(CEX, CEY, args)
				SX, SY, SamplingTIME = fm.sampling_proxy(REPX, REPY, args)
				X_train = torch.cat((X_train, REPX, SX), dim=0) if isinstance(REPX, torch.Tensor) else torch.cat(tuple([X_train] + REPX + SX), dim=0)
				y_train = torch.cat((y_train, REPY, SY), dim=0) if isinstance(REPX, torch.Tensor) else torch.cat(tuple([y_train] + REPY + SY), dim=0)
				stats['RepairTIME'][-1] = RepairTIME
				stats['NewPairs'][-1] = CEX.shape[0] + SX.shape[0] if isinstance(CEX, torch.Tensor) else sum(cex.shape[0] for cex in CEX) + sum(sx.shape[0] for sx in SX)
				stats['SamplingTIME'][-1] = SamplingTIME

		# statistics
		with open('experiments/{}/{}/stats.json'.format(COMBO,EXPERIMENT), 'w') as f:
			json.dump(stats, f)

		torch.save(network.state_dict(), 'experiments/{}/{}/network-{}.pt'.format(COMBO,EXPERIMENT,cegis_iteration))
		
		if NumberOfCounterExamples == 0:
			#print("No counterexamples found")
			break

	stats['TotalTIME'] = time.time() - START
	
	# final statistics
	with open('experiments/{}/{}/stats.json'.format(COMBO,EXPERIMENT), 'w') as f:
		json.dump(stats, f)