import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.cmlp import cMLP, train_model_ista
from models.utils import construct_dag
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os
from anndata import AnnData
import scanpy as sc
import scvelo as scv
import cellrank as cr
from cellrank.tl.kernels import VelocityKernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config={"velo": False,
        "proba": False, ## if velo is True and proba is False, it won't use the probabilistic transition matrix
        "dyna": False,
        "log": False,
        "gstd": False,
        "A": None,
        "X": None,
        "trial": tune.grid_search(['De-noised_100G_3T_300cPerT_dynamics_9_DS4', 'De-noised_100G_4T_300cPerT_dynamics_10_DS5', 'De-noised_100G_6T_300cPerT_dynamics_7_DS6', 'De-noised_100G_7T_300cPerT_dynamics_11_DS7']),
        "lr": tune.grid_search([0.0001, 0.01]),
        "lam": tune.grid_search(np.logspace(-2.0, 1.0, num=19).tolist()),
        "lam_ridge": 0,
        "penalty": 'H', ## options are 'H', 'GSGL' and 'GL'
        "lag": 5,
        "hidden": [32],
        "max_iter": 10000,
        "GC": None,
        "device": device,
        "lookback": 5,
        "check_every": 100,
        "verbose": 1}

analysis = tune.run(
    train_model_ista,
    resources_per_trial={"cpu": 1, "gpu": 0.1},
    config=config)

