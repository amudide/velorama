import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from copy import deepcopy
from models.model_helper import activation_helper
from models.utils import construct_S, seq2dag, construct_dag
import matplotlib.pyplot as plt
from sklearn import metrics
from synthetic import simulate_var, simulate_lorenz_96
import scvelo as scv
import cellrank as cr
from cellrank.tl.kernels import VelocityKernel


class MLP(nn.Module):
    ax = []

    def __init__(self, T, num_series, lag, hidden, device, activation):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)
        self.hidden = hidden
        self.lag = lag
        self.T = T
        self.device = device

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
            ret = ret + torch.matmul(MLP.ax[i], self.layers[0].weight[:, :, self.lag - 1 - i].T)
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

class cMLP(nn.Module):
    def __init__(self, A, X, num_series, lag, hidden, device, activation='relu'):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        if A == "linear":
            A = seq2dag(X.shape[1])
        S = construct_S(A)
        S = S.to(device)

        ax = []
        cur = S
        for _ in range(lag):
            ax.append(torch.matmul(cur, X[0]))
            cur = torch.matmul(S, cur)
            for i in range(len(cur)):
                cur[i][i] = 0
        
        #print("AX matrices:")
        #print(torch.stack(ax))

        MLP.ax = torch.stack(ax)
        MLP.ax = MLP.ax.to(device)

        # Set up networks.
        self.networks = nn.ModuleList([
            MLP(X.shape[1], num_series, lag, hidden, device, activation)
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


class cMLPSparse(nn.Module):
    def __init__(self, num_series, sparsity, lag, hidden, activation='relu'):
        '''
        cMLP model that only uses specified interactions.

        Args:
          num_series: dimensionality of multivariate time series.
          sparsity: torch byte tensor indicating Granger causality, with size
            (num_series, num_series).
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLPSparse, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)
        self.sparsity = sparsity

        # Set up networks.
        self.networks = []
        for i in range(num_series):
            num_inputs = int(torch.sum(sparsity[i].int()))
            self.networks.append(MLP(num_inputs, lag, hidden, activation))

        # Register parameters.
        param_list = []
        for i in range(num_series):
            param_list += list(self.networks[i].parameters())
        self.param_list = nn.ParameterList(param_list)

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([self.networks[i](X[:, :, self.sparsity[i]])
                          for i in range(self.p)], dim=2)


def prox_update(network, lam, lr, penalty):
    '''
    Perform in place proximal update on first layer weight matrix.

    Args:
      network: MLP network.
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
      network: MLP network.
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


def train_model_gista(config, checkpoint_dir = None):
    '''
    Train cMLP model with GISTA.

    Args:
      cmlp: cmlp model.
      X: tensor of data, shape (batch, T, p).
      lam: parameter for nonsmooth regularization.
      lam_ridge: parameter for ridge regularization on output layer.
      lr: learning rate.
      penalty: type of nonsmooth regularization.
      max_iter: max number of GISTA iterations.
      check_every: how frequently to record loss.
      r: for line search.
      lr_min: for line search.
      sigma: for line search.
      monotone: for line search.
      m: for line search.
      lr_decay: for adjusting initial learning rate of line search.
      begin_line_search: whether to begin with line search.
      switch_tol: tolerance for switching to line search.
      verbose: level of verbosity (0, 1, 2).
    '''
    
    A = config["A"]
    X = config["X"]
    trial = config["trial"]
    lr = config["lr"]
    lam = config["lam"]
    lam_ridge = config["lam_ridge"]
    penalty = config["penalty"]
    lag = config["lag"]
    hidden = config["hidden"]
    max_iter = config["max_iter"]
    GC = config["GC"]
    device = config["device"]
    lookback = config["lookback"]
    check_every = config["check_every"]
    verbose = config["verbose"]
    r=0.8
    lr_min=1e-8
    sigma=0.5
    monotone=False
    m=10
    lr_decay=0.5
    begin_line_search=True
    switch_tol=1e-3
    
    if X is None:
        exp = trial.split('-')[0]
        des = int(trial.split('-')[1])
        ln = int(trial.split('-')[2])
        sed = int(trial.split('-')[3])
        
        if exp == "var":
            X_np, beta, GC = simulate_var(p=20, T=ln, lag=des, seed=sed)
            X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
            
        if exp == "lorenz":
            X_np, GC = simulate_lorenz_96(p=20, F=des, T=ln, seed=sed)
            X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
    
    cmlp = cMLP(A, X, X.shape[-1], lag=lag, hidden=hidden, device=device)
    cmlp.to(device)
    
    p = cmlp.p
    lag = cmlp.lag
    cmlp_copy = deepcopy(cmlp)
    loss_fn = nn.MSELoss(reduction='mean')
    lr_list = [lr for _ in range(p)]

    # Calculate full loss.
    mse_list = []
    smooth_list = []
    loss_list = []
    for i in range(p):
        net = cmlp.networks[i]
        mse = loss_fn(net(), X[:, :, i:i+1])
        ridge = ridge_regularize(net, lam_ridge)
        smooth = mse + ridge
        mse_list.append(mse)
        smooth_list.append(smooth)
        with torch.no_grad():
            nonsmooth = regularize(net, lam, penalty)
            loss = smooth + nonsmooth
            loss_list.append(loss)

    # Set up lists for loss and mse.
    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p
    train_loss_list = [loss_mean]
    train_mse_list = [mse_mean]

    # For switching to line search.
    line_search = begin_line_search

    # For line search criterion.
    done = [False for _ in range(p)]
    assert 0 < sigma <= 1
    assert m > 0
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    for it in range(max_iter):
        # Backpropagate errors.
        sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

        # For next iteration.
        new_mse_list = []
        new_smooth_list = []
        new_loss_list = []

        # Perform GISTA step for each network.
        for i in range(p):
            # Skip if network converged.
            if done[i]:
                new_mse_list.append(mse_list[i])
                new_smooth_list.append(smooth_list[i])
                new_loss_list.append(loss_list[i])
                continue

            # Prepare for line search.
            step = False
            lr_it = lr_list[i]
            net = cmlp.networks[i]
            net_copy = cmlp_copy.networks[i]

            while not step:
                # Perform tentative ISTA step.
                for param, temp_param in zip(net.parameters(),
                                             net_copy.parameters()):
                    temp_param.data = param - lr_it * param.grad

                # Proximal update.
                prox_update(net_copy, lam, lr_it, penalty)

                # Check line search criterion.
                mse = loss_fn(net_copy(), X[:, :, i:i+1])
                ridge = ridge_regularize(net_copy, lam_ridge)
                smooth = mse + ridge
                with torch.no_grad():
                    nonsmooth = regularize(net_copy, lam, penalty)
                    loss = smooth + nonsmooth
                    tol = (0.5 * sigma / lr_it) * sum(
                        [torch.sum((param - temp_param) ** 2)
                         for param, temp_param in
                         zip(net.parameters(), net_copy.parameters())])

                comp = loss_list[i] if monotone else max(last_losses[i])
                if not line_search or (comp - loss) > tol:
                    step = True
                    if verbose > 1:
                        print('Taking step, network i = %d, lr = %f'
                              % (i, lr_it))
                        print('Gap = %f, tol = %f' % (comp - loss, tol))

                    # For next iteration.
                    new_mse_list.append(mse)
                    new_smooth_list.append(smooth)
                    new_loss_list.append(loss)

                    # Adjust initial learning rate.
                    lr_list[i] = (
                        (lr_list[i] ** (1 - lr_decay)) * (lr_it ** lr_decay))

                    if not monotone:
                        if len(last_losses[i]) == m:
                            last_losses[i].pop(0)
                        last_losses[i].append(loss)
                else:
                    # Reduce learning rate.
                    lr_it *= r
                    if lr_it < lr_min:
                        done[i] = True
                        new_mse_list.append(mse_list[i])
                        new_smooth_list.append(smooth_list[i])
                        new_loss_list.append(loss_list[i])
                        if verbose > 0:
                            print('Network %d converged' % (i + 1))
                        break

            # Clean up.
            net.zero_grad()

            if step:
                # Swap network parameters.
                cmlp.networks[i], cmlp_copy.networks[i] = net_copy, net

        # For next iteration.
        mse_list = new_mse_list
        smooth_list = new_smooth_list
        loss_list = new_loss_list

        # Check if all networks have converged.
        if sum(done) == p:
            if verbose > 0:
                print('Done at iteration = %d' % (it + 1))
            break

        # Check progress.
        if (it + 1) % check_every == 0:
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
                ridge_mean = (sum(smooth_list) - sum(mse_list)) / p
                nonsmooth_mean = (sum(loss_list) - sum(smooth_list)) / p

            train_loss_list.append(loss_mean)
            train_mse_list.append(mse_mean)

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Total loss = %f' % loss_mean)
                print('MSE = %f, Ridge = %f, Nonsmooth = %f'
                      % (mse_mean, ridge_mean, nonsmooth_mean))
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cmlp.GC().float())))

            # Check whether loss has increased.
            if not line_search:
                if train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                    line_search = True
                    if verbose > 0:
                        print('Switching to line search')

    GC_est = cmlp.GC(threshold=False).cpu().data.numpy()

    # Make figures
    y_true = flatten(GC)
    y_probas = flatten(GC_est)
    
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_probas)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.savefig('/afs/csail.mit.edu/u/a/amudide/gc/img/' + trial + '-' + str(lag) + '-' + str(lam) + '-' + str(penalty) + '-gista-roc.png', bbox_inches='tight')
    plt.show()
    
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    axarr[0].imshow(GC, cmap='Blues')
    axarr[0].set_title('GC actual')
    axarr[0].set_ylabel('Affected series')
    axarr[0].set_xlabel('Causal series')
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])

    axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
    axarr[1].set_title('GC estimated')
    axarr[1].set_ylabel('Affected series')
    axarr[1].set_xlabel('Causal series')
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])
    
    plt.savefig('/afs/csail.mit.edu/u/a/amudide/gc/img/' + trial + '-' + str(lag) + '-' + str(lam) + '-' + str(penalty) + '-gista-gc.png', bbox_inches='tight')
    plt.show()
    
    tune.report(score=roc_auc)


def train_model_adam(config, checkpoint_dir = None):
    
    A = config["A"]
    X = config["X"]
    trial = config["trial"]
    lr = config["lr"]
    lam = config["lam"]
    lam_ridge = config["lam_ridge"]
    penalty = config["penalty"]
    lag = config["lag"]
    hidden = config["hidden"]
    max_iter = config["max_iter"]
    GC = config["GC"]
    device = config["device"]
    lookback = config["lookback"]
    check_every = config["check_every"]
    verbose = config["verbose"]
    
    if X is None:
        exp = trial.split('-')[0]
        des = int(trial.split('-')[1])
        ln = int(trial.split('-')[2])
        sed = int(trial.split('-')[3])
        
        if exp == "var":
            X_np, beta, GC = simulate_var(p=20, T=ln, lag=des, seed=sed)
            X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
            
        if exp == "lorenz":
            X_np, GC = simulate_lorenz_96(p=20, F=des, T=ln, seed=sed)
            X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
    
    cmlp = cMLP(A, X, X.shape[-1], lag=lag, hidden=hidden, device=device)
    cmlp.to(device)
    
    
    '''Train model with Adam.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        loss = sum([loss_fn(cmlp.networks[i](), X[:, :, i:i+1])
                for i in range(p)])

        # Add penalty terms.
        if lam > 0:
            loss = loss + sum([regularize(net, lam, penalty)
                               for net in cmlp.networks])
        if lam_ridge > 0:
            loss = loss + sum([ridge_regularize(net, lam_ridge)
                               for net in cmlp.networks])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        cmlp.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    GC_est = cmlp.GC(threshold=False).cpu().data.numpy()

    # Make figures
    y_true = flatten(GC)
    y_probas = flatten(GC_est)
    
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_probas)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.savefig('/afs/csail.mit.edu/u/a/amudide/gc/img/' + trial + '-' + str(lag) + '-' + str(lam) + '-' + str(penalty) + '-adam-roc.png', bbox_inches='tight')
    plt.show()
    
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    axarr[0].imshow(GC, cmap='Blues')
    axarr[0].set_title('GC actual')
    axarr[0].set_ylabel('Affected series')
    axarr[0].set_xlabel('Causal series')
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])

    axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
    axarr[1].set_title('GC estimated')
    axarr[1].set_ylabel('Affected series')
    axarr[1].set_xlabel('Causal series')
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])
    
    plt.savefig('/afs/csail.mit.edu/u/a/amudide/gc/img/' + trial + '-' + str(lag) + '-' + str(lam) + '-' + str(penalty) + '-adam-gc.png', bbox_inches='tight')
    plt.show()
    
    tune.report(score=roc_auc)

def flatten(xss):
    return [x for xs in xss for x in xs]

def train_model_ista(config, checkpoint_dir = None):
    A = config["A"]
    X = config["X"]
    trial = config["trial"]
    lr = config["lr"]
    lam = config["lam"]
    lam_ridge = config["lam_ridge"]
    penalty = config["penalty"]
    lag = config["lag"]
    hidden = config["hidden"]
    max_iter = config["max_iter"]
    GC = config["GC"]
    device = config["device"]
    lookback = config["lookback"]
    check_every = config["check_every"]
    verbose = config["verbose"]
    
    if A == "linear":
        exp = trial.split('-')[0]
        des = int(trial.split('-')[1])
        ln = int(trial.split('-')[2])
        sed = int(trial.split('-')[3])
        
        if exp == "var":
            X_np, beta, GC = simulate_var(p=20, T=ln, lag=des, seed=sed)
            X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
            
        if exp == "lorenz":
            X_np, GC = simulate_lorenz_96(p=20, F=des, T=ln, seed=sed)
            X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
    else:
        X = pd.read_csv('/afs/csail.mit.edu/u/a/amudide/gc/data_sets/' + trial + '/T.csv', index_col=0)
            
        X = X.to_numpy()
        X = np.transpose(X)
        adata = AnnData(X, dtype=np.float32)
        
        Y = pd.read_csv('/afs/csail.mit.edu/u/a/amudide/gc/data_sets/' + trial + '/U.csv', index_col=0)
        Y = Y.to_numpy()
        Y = np.transpose(Y)

        Z = pd.read_csv('/afs/csail.mit.edu/u/a/amudide/gc/data_sets/' + trial + '/S.csv', index_col=0)
        Z = Z.to_numpy()
        Z = np.transpose(Z)
        
        adata.layers["unspliced"] = Y
        adata.layers["spliced"] = Z
        
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        
        #scv.tl.recover_dynamics(adata, n_jobs=8)
        #scv.tl.velocity(adata, mode="dynamical")
        
        scv.tl.velocity(adata)
        
        scv.tl.velocity_graph(adata)
        
        vk = VelocityKernel(adata).compute_transition_matrix()

        A = vk.transition_matrix
        
        #A = adata.uns['velocity_graph']
        A = A.toarray()
        
        '''
        for i in range(len(A)):
            for j in range(len(A)):
                if A[i][j] < 0.1:
                    A[i][j] = 0
                else:
                    A[i][j] = 1
        '''
        
        for i in range(len(A)):
            A[i][i] = 0
        
        #sc.tl.pca(adata, svd_solver='arpack')
        #A = construct_dag(adata.obsm['X_pca'], 0)
        #A = A.T
        
        A = torch.from_numpy(A)
        X = torch.from_numpy(X)
        X = torch.unsqueeze(X, 0)

        A = A.float()
        X = X.float()

        A = A.to(device)
        X = X.to(device)
        
        gclist = pd.read_csv('/afs/csail.mit.edu/u/a/amudide/gc/data_sets/' + trial + '/gt_GRN.csv', header=None)
        GC = np.zeros([X.shape[2], X.shape[2]])
        for index, row in gclist.iterrows():
            GC[row[1]][row[0]] = 1
        
    
    cmlp = cMLP(A, X, X.shape[-1], lag=lag, hidden=hidden, device=device)
    cmlp.to(device)
    
    #if torch.cuda.device_count() > 1:
    #    cmlp = nn.DataParallel(cmlp)
        
    '''Train model with ISTA.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    # Calculate smooth error.
    loss = sum([loss_fn(cmlp.networks[i](), X[:, :, i:i+1])
                for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
    smooth = loss + ridge

    for it in range(max_iter):
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
        loss = sum([loss_fn(cmlp.networks[i](), X[:, :, i:i+1])
                    for i in range(p)])
        ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
        smooth = loss + ridge

        # Check progress.
        if (it + 1) % check_every == 0:
            # Add nonsmooth penalty.
            nonsmooth = sum([regularize(net, lam, penalty)
                             for net in cmlp.networks])
            mean_loss = (smooth + nonsmooth) / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cmlp.GC().float())))

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    GC_est = cmlp.GC(threshold=False).cpu().data.numpy()
    
    GC_lag = cmlp.GC(threshold=False, ignore_lag=False).cpu()
    torch.save(GC_lag, '/afs/csail.mit.edu/u/a/amudide/gc/img/cr-' + str(max_iter) + '-' + trial + '-' + str(penalty) + '-' + str(lam) + '-' + str(lr) + '-ista-lag.pt')
    
    for i in range(len(GC)):
        GC[i][i] = 0
        GC_est[i][i] = 0
        
    np.savetxt('/afs/csail.mit.edu/u/a/amudide/gc/img/cr-' + str(max_iter) + '-' + trial + '-' + str(penalty) + '-' + str(lam) + '-' + str(lr) + '-ista-gc.csv', GC_est, delimiter=",")

    # Make figures
    y_true = flatten(GC)
    y_probas = flatten(GC_est)
    
    ydf = pd.DataFrame(list(zip(y_true, y_probas)),
               columns =['y_true', 'y_probas'])
    ydf.to_csv('/afs/csail.mit.edu/u/a/amudide/gc/img/cr-' + str(max_iter) + '-' + trial + '-' + str(penalty) + '-' + str(lam) + '-' + str(lr) + '-ista-preds.csv')
    
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_probas)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.savefig('/afs/csail.mit.edu/u/a/amudide/gc/img/cr-' + str(max_iter) + '-' + trial + '-' + str(penalty) + '-' + str(lam) + '-' + str(lr) + '-ista-roc.png', bbox_inches='tight')
    plt.show()
    
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    axarr[0].imshow(GC, cmap='Blues')
    axarr[0].set_title('GC actual')
    axarr[0].set_ylabel('Affected series')
    axarr[0].set_xlabel('Causal series')
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])

    axarr[1].imshow(GC_est, cmap='Blues', vmin=0.7, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
    axarr[1].set_title('GC estimated')
    axarr[1].set_ylabel('Affected series')
    axarr[1].set_xlabel('Causal series')
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])
    
    plt.savefig('/afs/csail.mit.edu/u/a/amudide/gc/img/cr-' + str(max_iter) + '-' + trial + '-' + str(penalty) + '-' + str(lam) + '-' + str(lr) + '-ista-gc.png', bbox_inches='tight')
    plt.show()
    
    tune.report(score=roc_auc)

def train_unregularized(cmlp, X, lr, max_iter, lookback=5, check_every=100,
                        verbose=1):
    '''Train model with Adam and no regularization.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        pred = cmlp()
        loss = sum([loss_fn(pred[:, :, i], X[:, :, i]) for i in range(p)])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        cmlp.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    return train_loss_list
