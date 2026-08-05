import torch
import torch.optim as optim
import torch.nn as nn
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

import copy
import conf
import fm
    
def custom_MSELoss(y_pred, y_true):

    difference = y_true - y_pred
    # the index of the highest value in y_true is the optimal advisory
    advisory_true = torch.argmax(y_true, dim=1)

    # matrix where there's a 1 in the index of the optimal advisory, 0 for the other 4 suboptimal advisories
    advisory_onehot = nn.functional.one_hot(advisory_true, conf.OutputSIZE)

    # matrix where there's a 0 in the index of the optimal advisory, -1 for the other 4 suboptimal advisories
    others_onehot = advisory_onehot - 1

    # matrix that contains the differences between y_true and y_pred only for the optimal advisory
    d_optimal = difference * advisory_onehot
    # matrix that contains the differences between y_true and y_pred for the suboptimal advisories
    d_suboptimal = difference * others_onehot

    # penalized loss to be applied when the optimal advisory is underestimated
    penalized_optimal_loss = conf.AMPLIFIER * (conf.OutputSIZE - 1) * (torch.square(d_optimal) + torch.abs(d_optimal))
    # normal loss to be applied when the optimal advisory is overestimated
    optimal_loss = torch.square(d_optimal)

    # penalized loss to be applied when a suboptimal advisory is overestimated
    penalized_suboptimal_loss = conf.AMPLIFIER * (torch.square(d_suboptimal) + torch.abs(d_suboptimal))
    # normal loss to be applied when a suboptimal advisory is underestimated
    suboptimal_loss = torch.square(d_suboptimal)

    # apply the losses
    optimal_advisory_loss = torch.where(d_optimal > 0, penalized_optimal_loss, optimal_loss)
    suboptimal_advisory_loss = torch.where(d_suboptimal > 0, penalized_suboptimal_loss, suboptimal_loss)

    # average out the combined losses
    mean_loss = torch.mean(optimal_advisory_loss + suboptimal_advisory_loss)
    return mean_loss

def train(network, X_train, y_train, X_valid, y_valid, args):
	start = time.time()
	optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)
	batch_size = 512

	loss_fn = custom_MSELoss
	best_state = copy.deepcopy(network.state_dict())
	best_lossV  = 1000000000
	best_lossS  = 1000000000 if args.soft_acceleration else 0
	
	#print("Initial network: loss(validation) = {}".format(best_lossV))

	stop = 0

	train = torch.utils.data.TensorDataset(X_train, y_train)
	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device.type))

	for epoch in range(args.epochs):
		lossS = 0
		#print("epoch = {}".format(epoch))
		for data in train_loader:
			Xbatch, ybatch = data
			y_pred = network(Xbatch)
			loss = loss_fn(y_pred, ybatch)
			# soft constraint loss
			if args.soft_acceleration:
				lossS = fm.soft_loss_proxy(network, batch_size, args) # random soft batch
				#print(lossS)
				loss += lossS
								
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		# compute loss on valid
		lossS = 0
		with torch.no_grad():
			y_pred = network(X_valid)
			lossV = loss_fn(y_pred, y_valid).item()
			if args.soft_acceleration:
				lossS = fm.soft_loss_proxy(network, X_valid.shape[0], args).item()
				
		if (lossV + lossS) < (best_lossV + best_lossS): # see above
			best_state = copy.deepcopy(network.state_dict())
			best_lossV  = lossV
			best_lossS  = lossS
			#print("Update after epoch {}: lossV = {}, lossS = {}".format(epoch, best_lossV, best_lossS))
			stop = 0
		else:
			stop += 1

		if stop > args.early_stopping:
			break
			
	network.load_state_dict(best_state)
	return network, best_lossV, time.time() - start