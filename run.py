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

config={"velo": True,
        "A": None,
        "X": None,
        "trial": 'De-noised_100G_6T_300cPerT_dynamics_7_DS6',
        "lr": 0.0001,
        "lam": tune.grid_search(np.logspace(-3.0, 3.0, num=39).tolist()),
        "lam_ridge": 0,
        "penalty": 'H', ## other options are 'GSGL' and 'GL'
        "lag": 5,
        "hidden": [100], ## [100, 100] would correspond to two hidden layers, each with 100 nodes
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

