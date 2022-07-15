'''
Model architecture for LagNet:
	A: adjacency matrix (N x N)
	X: expression matrix (N x g)
	L: number of lags to look back
	K: number of layers in the model
	d: number of nodes in hidden layers
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class LagNet(nn.Module):
	def __init__(self,A,X,L,K,d,final_activation=None):
		super(LagNet, self).__init__()

		self.A = A
		self.X = X
		self.L = L
		self.K = K
		self.d = d
		self.final_activation = final_activation

		self.N = self.X.size(dim=0)
		self.g = self.X.size(dim=1)

		cur = torch.clone(self.X)
		for i in range(1, self.L + 1):
			cur = torch.matmul(self.A, cur)
			setattr(self,'ax{}'.format(i),cur)

		for i in range(1, self.L + 1):
			setattr(self,'fc1_{}'.format(i),nn.Linear(self.g, self.d, bias=False))

		self.b1 = nn.Parameter(torch.zeros((1, self.d)),requires_grad=True) # initialization?
		
		for i in range(2, self.K):
			setattr(self,'fc{}'.format(i),nn.Linear(self.d, self.d))

		setattr(self,'fc{}'.format(self.K),nn.Linear(self.d, 1))


	def forward(self):
		ret = torch.zeros(self.N, self.d)  # set requires_grad to True?

		for i in range(1, self.L + 1):
			ret = ret + getattr(self,'fc1_{}'.format(i))(getattr(self,'ax{}'.format(i)))
		ret = ret + self.b1
		ret = F.relu(ret)

		for i in range(2, self.K):
			ret = getattr(self,'fc{}'.format(i))(ret)
			ret = F.relu(ret)

		ret = getattr(self,'fc{}'.format(self.K))(ret)

		if self.final_activation == 'exp':
			ret = torch.exp(ret)

		return ret