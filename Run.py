import subprocess

import experiments

# in seconds
TIMEOUT=3600*24

E = 1000 # epochs
S = 50  # early stopping

# seeds
SEEDS = [100]

SOFT = ['--soft-acceleration','']
# command
c = (
	'python HCASCertifier.py {property} '
	'--pra {pra} '
	'--tau {tau} '
	'--learning-rate {learning_rate} '
	'--epochs {epochs} '
	'--early-stopping {earlystopping} '
	'--hidden-layers {layers} '
	'--neurons-per-hidden-layer {neurons} '
	'--counterexamples {counterexamples} '
	'--sampling {sampling} '
	'--seed {seed} '
	'{soft} ' 
	'--epsilon {epsilon} '
	'--delta {delta}'
	)

# on all networks
cmds = [c.format(
	property=p, 
	pra=pra, 
	tau=tau, 
	learning_rate=learning_rate, 
	epochs=E, 
	earlystopping=S, 
	layers=layers, 
	neurons=neurons,
	counterexamples=counterexamples,
	sampling=sampling, 
	seed=seed, 
	soft=soft, 
	epsilon=epsilon, 
	delta=delta) 
		for p in experiments.PROPERTIES
		for pra in experiments.PRA 
		for tau in experiments.TAU 
		for learning_rate in experiments.LEARNING_RATES 
		for (layers,neurons) in experiments.BRAINS
		for seed in SEEDS 
		for (counterexamples,sampling) in experiments.STRATEGIES
		for soft in SOFT  
		for epsilon in experiments.EPSILON 
		for delta in experiments.DELTA 
		if delta > epsilon and (
			(p in [1,2,3,4,5,6,10,12345610]) or 
			(p == 7 and tau == '60') or 
			(p == 8 and pra == 1 and tau == '60') or 
			(p == 9 and pra == 2) or
			(p == 123467810 and pra == 1 and tau == '60') or
			(p == 123456810 and pra == 1 and tau == '60') or
			(p == 123456910 and pra == 2 and tau == '60') 
		)]


for cmd in cmds:
	#print("Running: {}".format(cmd))

	try:
	    p = subprocess.run(cmd.split(), stdout=subprocess.PIPE, timeout=TIMEOUT)
	except subprocess.CalledProcessError as e:
	    print("{}".format(e))
	except:
	    print('timeout\n')
