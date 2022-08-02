import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from models.utils import construct_S, seq2dag

def activation_helper(activation, dim=None):
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: %s' % activation)
    return act

class lagnet(nn.Module):
    ax = []

    def __init__(self, T, num_series, lag, hidden, device, activation):
        super(lagnet, self).__init__()
        self.activation = activation_helper(activation)
        self.hidden = hidden
        self.lag = lag
        self.T = T
        self.device = device
        self.dropout = nn.Dropout(0.25)

        # Set up network.
        layer = nn.Conv1d(num_series, hidden[0], lag)
        modules = [layer]

        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            layer = nn.Conv1d(d_in, d_out, 1)
            modules.append(layer)

        # Register parameters.
        self.layers = nn.ModuleList(modules)

    def forward(self):
        ret = torch.zeros(self.T, self.hidden[0], device=self.device)
        for i in range(self.lag):
            ret = ret + torch.matmul(lagnet.ax[i], self.layers[0].weight[:, :, self.lag - 1 - i].T)
        ret = ret + self.layers[0].bias

        ret = ret.T
        for i, fc in enumerate(self.layers):
            if i == 0:
                continue
            ret = self.activation(ret)
            ret = fc(ret)

        ret = ret.T
        # exponential function?

        ret = torch.unsqueeze(ret, 0)
        return ret

class clagnet(nn.Module):
    def __init__(self, A, X, num_series, lag, hidden, device, activation='relu'):
        '''
        clagnet model with one lagnet per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(clagnet, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        if A == "linear":
            A = seq2dag(X.shape[1])
        S = construct_S(A)
        S = S.to(device)

        ax = []
        cur = torch.clone(X[0])
        for _ in range(lag):
            cur = torch.matmul(S, cur)
            ax.append(cur)
        
        #print("AX matrices:")
        #print(torch.stack(ax))

        lagnet.ax = torch.stack(ax)
        lagnet.ax = lagnet.ax.to(device)

        # Set up networks.
        self.networks = nn.ModuleList([
            lagnet(X.shape[1], num_series, lag, hidden, device, activation)
            for _ in range(num_series)])

    def forward(self):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([network() for network in self.networks], dim=2)

    def GC(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if ignore_lag:
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


def prox_update(network, lam, lr, penalty):
    '''
    Perform in place proximal update on first layer weight matrix.

    Args:
      network: lagnet network.
      lam: regularization parameter.
      lr: learning rate.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        for i in range(lag):
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            W.data[:, :, :(i+1)] = (
                (W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def regularize(network, lam, penalty):
    '''
    Calculate regularization term for first layer weight matrix.

    Args:
      network: lagnet network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params

def train_model_ista(clagnet, X, lam, lam_ridge, lr, penalty = "H", max_iter=50000, 
                    check_every=100, lookback=5, verbose = 1):

    '''Train model with ISTA.'''
    lag = clagnet.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    # Calculate smooth error.
    loss = sum([loss_fn(clagnet.networks[i](), X[:, :, i:i+1])
                for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in clagnet.networks])
    smooth = loss + ridge

    for it in range(max_iter):
        # Take gradient step.
        smooth.backward()
        for param in clagnet.parameters():
            param.data = param - lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in clagnet.networks:
                prox_update(net, lam, lr, penalty)

        clagnet.zero_grad()

        # Calculate loss for next iteration.
        loss = sum([loss_fn(clagnet.networks[i](), X[:, :, i:i+1])
                    for i in range(p)])
        ridge = sum([ridge_regularize(net, lam_ridge) for net in clagnet.networks])
        smooth = loss + ridge

        # Check progress.
        if (it + 1) % check_every == 0:
            # Add nonsmooth penalty.
            nonsmooth = sum([regularize(net, lam, penalty)
                             for net in clagnet.networks])
            mean_loss = (smooth + nonsmooth) / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(clagnet.GC().float())))

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(clagnet)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(clagnet, best_model)

    return train_loss_list

