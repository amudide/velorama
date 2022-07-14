import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import os

from models import LagNet

def run(A,X,L,K,d,target,final_activation=None,seed=1,optim='adam',
		initial_learning_rate=0.001,beta_1=0.9,beta_2=0.999,epochs=20,
		save_dir='./results',save_name='lagnet'):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(seed)
	np.random.seed(seed)

	start = time.time()

	A = A.float()
	X = X.float()

	A = A.to(device)
	X = X.to(device)

	model = LagNet(A,X,L,K,d,final_activation)
	model.to(device)

	criterion = nn.MSELoss()

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

		loss = criterion(preds, targets)

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

	return model.parameters()