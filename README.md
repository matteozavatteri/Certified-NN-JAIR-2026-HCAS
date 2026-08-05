# Paper

This repository contains the supplementary material of the paper:

	Matteo Zavatteri, Davide Bresolin, Nicolò Navarin. 
	Certified Neural Networks: from Verification to Synthesis.
	Journal of Artificial Intelligence Research, 2026.

# Dataset

This code works on the original dataset of the paper:
	
	Julian, Kyle D and Kochenderfer, Mykel J
	Guaranteeing safety for neural network-based aircraft collision avoidance systems
	Digital Avionics Systems Conference (DASC), 2019.

The dataset we provide here is computed by using the generator and the original data here: [https://github.com/sisl/HorizontalCAS](https://github.com/sisl/HorizontalCAS)

To run all the experiments of the paper run:

	$ python3 Run.py

# Usage

To run a single experiment run 

	$ python3 HCASCertifier.py -h 

to see the help screen (see also inside Run.py to have example of possible commands).

# Data
The original data of the experimental evaluation discussed in the paper in contained in the folder `experiments`.
