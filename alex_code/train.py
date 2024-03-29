import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import time

from models import *
from utils import *

def train_model(config, checkpoint_dir = None):

	AX = config["AX"]
	Y = config["Y"]

	method = config["method"]

	trial_no = config["trial"]
	lr = config["lr"]
	lam = config["lam"]
	lam_ridge = config["lam_ridge"]
	penalty = config["penalty"]
	lag = config["lag"]
	hidden = config["hidden"]
	max_iter = config["max_iter"]
	device = config["device"]
	lookback = config["lookback"]
	check_every = config["check_every"]
	verbose = config["verbose"]
	tol = config['tol']
	dynamics = config['dynamics']
		
	gc_dir = config['gc_dir']
	dir_name = config['dir_name']

	np.random.seed(trial_no)
	torch.manual_seed(trial_no)


	file_name = '{}.trial{}.lam{}.ridge{}.h{}.{}.lag{}.{}'.format(method,trial_no,lam,
				lam_ridge,hidden[0],penalty,lag,dynamics)
	gc_path1 = os.path.join(gc_dir,dir_name,file_name + '.pt')
	gc_path2 = os.path.join(gc_dir,dir_name,file_name + '.ignore_lag.pt')

	if not os.path.exists(gc_path1) and not os.path.exists(gc_path2):
		
		num_regs = AX.shape[-1]
		num_targets = Y.shape[1]

		cmlp = cMLP(num_targets, num_regs,lag=lag, hidden=hidden, device=device, activation='relu')
		cmlp.to(device)

		AX = AX.to(device)
		Y = Y.to(device)

		#if torch.cuda.device_count() > 1:
		#    cmlp = nn.DataParallel(cmlp)
			
		'''Train model with ISTA.'''
		lag = cmlp.lag
		# p = Y.shape[1]
		loss_fn = nn.MSELoss(reduction='none')
		train_loss_list = []

		# For early stopping.
		best_it = None
		best_loss = np.inf
		best_model = None

		# Calculate smooth error.
		loss = loss_fn(cmlp(AX),Y).mean(0).sum()
		# loss = sum([loss_fn(cmlp.networks[i](AX), Y[:,i]) for i in range(Y.shape[1])])

		ridge = torch.sum(torch.stack([ridge_regularize(net, lam_ridge) for net in cmlp.networks]))
		smooth = loss + ridge

		variable_usage_list = []
		loss_list = []

		# For early stopping.
		train_loss_list = []
		best_it = None
		best_loss = np.inf
		best_model = None

		for it in range(max_iter):

			start = time.time()
			# Take gradient step.
			smooth.backward()
			for param in cmlp.parameters():
				param.data = param - lr * param.grad

			# Take prox step.
			if lam > 0:
				for net in cmlp.networks:
					prox_update(net, lam, lr, penalty)

			cmlp.zero_grad()

			# Calculate loss for next iteration.
			loss = loss_fn(cmlp(AX),Y).mean(0).sum()
			# loss = sum([loss_fn(cmlp.networks[i](AX), Y[:,i]) for i in range(Y.shape[1])])
			ridge = torch.sum(torch.stack([ridge_regularize(net, lam_ridge) for net in cmlp.networks]))
			smooth = loss + ridge

			# Check progress.
			if (it + 1) % check_every == 0:

				nonsmooth = torch.sum(torch.stack([regularize(net, lam, penalty) for net in cmlp.networks]))
				mean_loss = (smooth + nonsmooth).detach()/Y.shape[1]

				variable_usage = torch.mean(cmlp.GC(ignore_lag=False).float())
				variable_usage_list.append(variable_usage)
				loss_list.append(mean_loss)

				# Check for early stopping.
				if mean_loss < best_loss:
					best_loss = mean_loss
					best_it = it
					best_model = deepcopy(cmlp)

					if verbose:
						print('Lam={}: Iter {}, {} sec'.format(lam,it+1,np.round(time.time()-start,2)),
							  '-----','Loss: %.2f' % mean_loss,', Variable usage = %.2f%%' % (100 * variable_usage)) # ,
							  # '|||','%.3f' % loss_crit,'%.3f' % variable_usage_crit)

				elif (it - best_it) == lookback * check_every:
					if verbose:
						print('EARLY STOP: Lam={}, Iter {}'.format(lam,it + 1))
					break

		# Restore best model.
		restore_parameters(cmlp, best_model)


		if not os.path.exists(gc_dir):
			os.mkdir(gc_dir)
		if not os.path.exists(os.path.join(gc_dir,dir_name)):
			os.mkdir(os.path.join(gc_dir,dir_name))

		file_name = '{}.trial{}.lam{}.ridge{}.h{}.{}.lag{}.{}'.format(method,trial_no,lam,
					lam_ridge,hidden[0],penalty,lag,dynamics)
		GC_lag = cmlp.GC(threshold=False, ignore_lag=False).cpu()
		torch.save(GC_lag, os.path.join(gc_dir,dir_name,file_name + '.pt'))

		GC_lag = cmlp.GC(threshold=False, ignore_lag=True).cpu()
		torch.save(GC_lag, os.path.join(gc_dir,dir_name,file_name + '.ignore_lag.pt'))


	
	
def train_model_batch(config, checkpoint_dir = None):

	AX = config["AX"]
	Y = config["Y"]

	method = config["method"]

	trial_no = config["trial"]
	lr = config["lr"]
	lam = config["lam"]
	lam_ridge = config["lam_ridge"]
	penalty = config["penalty"]
	lag = config["lag"]
	hidden = config["hidden"]
	max_iter = config["max_iter"]
	device = config["device"]
	lookback = config["lookback"]
	check_every = config["check_every"]
	verbose = config["verbose"]
	tol = config['tol']
	dynamics = config['dynamics']
		
	gc_dir = config['gc_dir']
	dir_name = config['dir_name']

	np.random.seed(trial_no)
	torch.manual_seed(trial_no)


	file_name = '{}.trial{}.lam{}.ridge{}.h{}.{}.lag{}.{}'.format(method,trial_no,lam,
				lam_ridge,hidden[0],penalty,lag,dynamics)
	gc_path1 = os.path.join(gc_dir,dir_name,file_name + '.pt')
	gc_path2 = os.path.join(gc_dir,dir_name,file_name + '.ignore_lag.pt')

	if not os.path.exists(gc_path1) and not os.path.exists(gc_path2):

		num_regs = AX.shape[-1]
		num_targets = Y.shape[1]

		cmlp = cMLP(num_targets, num_regs,lag=lag, hidden=hidden, device=device, activation='relu')
		cmlp.to(device)

		# AX = AX.to(device)
		# Y = Y.to(device)

		#if torch.cuda.device_count() > 1:
		#    cmlp = nn.DataParallel(cmlp)
			
		'''Train model with ISTA.'''
		lag = cmlp.lag
		# p = Y.shape[1]
		loss_fn = nn.MSELoss(reduction='none')
		train_loss_list = []

		# For early stopping.
		best_it = None
		best_loss = np.inf
		best_model = None

		batch_size = 512

		# Calculate smooth error.
		loss = sum(sum([loss_fn(cmlp(AX[:,i:i+batch_size,:].to(device)),Y[i:i+batch_size].to(device)).mean(0) for i in range(0,AX.shape[1],batch_size)])) # .sum()
		# loss = sum([loss_fn(cmlp.networks[i](AX), Y[:,i]) for i in range(Y.shape[1])])

		ridge = torch.sum(torch.stack([ridge_regularize(net, lam_ridge) for net in cmlp.networks]))
		smooth = loss + ridge

		variable_usage_list = []
		loss_list = []

		# For early stopping.
		train_loss_list = []
		best_it = None
		best_loss = np.inf
		best_model = None

		for it in range(max_iter):

			start = time.time()
			# Take gradient step.
			smooth.backward()
			for param in cmlp.parameters():
				param.data = param - lr * param.grad

			# Take prox step.
			if lam > 0:
				for net in cmlp.networks:
					prox_update(net, lam, lr, penalty)

			cmlp.zero_grad()

			# Calculate loss for next iteration.
			loss = sum(sum([loss_fn(cmlp(AX[:,i:i+batch_size,:].to(device)),Y[i:i+batch_size].to(device)).mean(0) for i in range(0,AX.shape[1],batch_size)])) # .sum()
			# loss = sum([loss_fn(cmlp.networks[i](AX), Y[:,i]) for i in range(Y.shape[1])])
			ridge = torch.sum(torch.stack([ridge_regularize(net, lam_ridge) for net in cmlp.networks]))
			smooth = loss + ridge

			# Check progress.
			if (it + 1) % check_every == 0:

				nonsmooth = torch.sum(torch.stack([regularize(net, lam, penalty) for net in cmlp.networks]))
				mean_loss = (smooth + nonsmooth).detach()/Y.shape[1]

				variable_usage = torch.mean(cmlp.GC(ignore_lag=False).float())
				variable_usage_list.append(variable_usage)
				loss_list.append(mean_loss)

				# Check for early stopping.
				if mean_loss < best_loss:
					best_loss = mean_loss
					best_it = it
					best_model = deepcopy(cmlp)

					if verbose:
						print('Lam={}: Iter {}, {} sec'.format(lam,it+1,np.round(time.time()-start,2)),
							  '-----','Loss: %.2f' % mean_loss,', Variable usage = %.2f%%' % (100 * variable_usage)) # ,
							  # '|||','%.3f' % loss_crit,'%.3f' % variable_usage_crit)

				elif (it - best_it) == lookback * check_every:
					if verbose:
						print('EARLY STOP: Lam={}, Iter {}'.format(lam,it + 1))
					break

		# Restore best model.
		restore_parameters(cmlp, best_model)


		if not os.path.exists(gc_dir):
			os.mkdir(gc_dir)
		if not os.path.exists(os.path.join(gc_dir,dir_name)):
			os.mkdir(os.path.join(gc_dir,dir_name))

		file_name = '{}.trial{}.lam{}.ridge{}.h{}.{}.lag{}.{}'.format(method,trial_no,lam,
					lam_ridge,hidden[0],penalty,lag,dynamics)
		GC_lag = cmlp.GC(threshold=False, ignore_lag=False).cpu()
		torch.save(GC_lag, os.path.join(gc_dir,dir_name,file_name + '.pt'))

		GC_lag = cmlp.GC(threshold=False, ignore_lag=True).cpu()
		torch.save(GC_lag, os.path.join(gc_dir,dir_name,file_name + '.ignore_lag.pt'))

	else:
		print('Already trained...DONE')

	
	

