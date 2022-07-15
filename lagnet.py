'''
Training LagNet:
	A: adjacency matrix (N x N)
	X: expression matrix (N x g)
	L: number of lags to look back
	K: number of layers in the model
	d: number of nodes in hidden layers
	target: the gene id that the model predicts.
	lam: lamda value, the coefficient for regularization
	alpha: coefficient in mixed regularization
'''

import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import os

from models import LagNet
from utils import construct_S0_S1

def run(A,X,L,K,d,target,final_activation=None,lam=794e-6,alpha=0.5,seed=1,optim='adam',
		initial_learning_rate=0.001,beta_1=0.9,beta_2=0.999,epochs=200,
		save_dir='./results',save_name='lagnet'):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(seed)
	np.random.seed(seed)

	start = time.time()
	
	g = X.size(dim=1)

	A = A.float()
	X = X.float()

	#S_0, _ = construct_S0_S1(A)

	#S_0 = S_0.to(device)
	A = A.to(device)
	X = X.to(device)

	model = LagNet(A,X,L,K,d,final_activation)
	model.to(device)

	criterion = nn.MSELoss()  # reduction = sum?

	if optim == 'sgd':
		optimizer = torch.optim.SGD(params=model.parameters(), 
			lr=initial_learning_rate)
	elif optim == 'adam':
		optimizer = torch.optim.Adam(params=model.parameters(), 
			lr=initial_learning_rate, betas=(beta_1, beta_2))

	for epoch in range(epochs):
		ep_start = time.time()
			
		preds = model()
		targets = X[:, target]
		targets = torch.unsqueeze(targets, 1)

		reg = 0

		params = list(model.parameters())
		weights = []
		for i in range(1, L + 1):
			weights.append(params[i])

		weights = torch.stack(weights, dim=0)
		weights = [weights[:,:,i] for i in range(g)]

		for i in range(g):
			reg = reg + alpha * torch.linalg.matrix_norm(weights[i])
			for j in range(L):
				reg = reg + (1 - alpha) * torch.linalg.norm(weights[i][j])
		reg = reg * lam

		loss = criterion(preds, targets) + reg

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		ep_end = time.time()

		mode = 'train'
		print('Epoch {} ({:.2f} seconds): {} loss {:.2f}'.format(epoch,ep_end-ep_start,mode,loss))


	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	model.to('cpu')
	torch.save(model.state_dict(),os.path.join(save_dir,'{}.model_weights.pth'.format(
		save_name)))

	print('Total Time: {} seconds'.format(time.time()-start))

	with torch.no_grad():
		params = list(model.parameters())
		weights = []
		for i in range(1, L + 1):
			weights.append(params[i])

		weights = torch.stack(weights, dim=0)
		weights = [weights[:,:,i] for i in range(g)]

		return weights