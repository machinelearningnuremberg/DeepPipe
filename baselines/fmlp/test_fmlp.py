from fmlp import MLPEnsemble
import numpy as np
import torch 
import json
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as OHE
from torch import autograd as ag
from scipy.stats import norm as norm
import argparse
import copy
from fmlp_loaders import PMFMetaDatasetLoader, TOboeMetaDatasetLoader

def EI(mean, sigma, best_f, epsilon = 0):    
    with np.errstate(divide='warn'):
        imp = mean -best_f - epsilon
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def BO(model, Lambdas, response, x_observed, x_pending, bo_iter):

    y = response[x_observed]
    q = Lambdas[x_observed]

    best = max(response)
    regret = [(best-max(y[0:(i+1)])).item() for i in range(0,len(y)) ]
    for i in range(bo_iter):

        if max(y) == best:
            regret+=[0]*(bo_iter-i)
            break
        model.fit(q, y)
        x_query = Lambdas[x_pending]
        mu, std = model.predict(x_query)
        ei = EI(mu, std, max(y))
        next_q = x_pending[np.argmax(ei)]
        x_observed.append(next_q)
        x_pending.remove(next_q)
        y = response[x_observed]
        q = Lambdas[x_observed]
        regret.append((best-max(y)).item())

    return regret

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--bo_iter', default=105, type=int)
    parser.add_argument('--experiment_name', default="exp01", type=str)
    parser.add_argument('--task', default="37", type=str)
    parser.add_argument('--test_id', default="test0", type=str)
    parser.add_argument('--fold', default="0", type=int)
    parser.add_argument('--dataset', default="oboe-dataset", type=str)
    parser.add_argument('--n_models', default=20, type=int)


    args = parser.parse_args()
    print(args)

    #hyperparaters
    experiment_name = args.experiment_name
    epochs = args.epochs
    task = args.task
    bo_iter = args.bo_iter
    test_id = args.test_id
    fold = args.fold
    dataset = args.dataset
    n_models = args.n_models
    n_val_iters = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##loading data
    rootdir     = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(rootdir, "..","data")

    if dataset=="oboe-dataset":
        metadataset = TOboeMetaDatasetLoader(path, test=True, fold = fold)
    else:
        metadataset = PMFMetaDatasetLoader(path, test=True,fold=fold)
   
    model = MLPEnsemble(path=rootdir, experiment_name=experiment_name, n_models=n_models, finetuning_epochs = epochs, device=device)

    Lambdas, response, x_pending, x_observed = metadataset.get_initial_data(task)
    regret = BO(model, Lambdas, response, x_observed, x_pending, bo_iter)
    path = os.path.join(rootdir, "..",  "results", "fmlp", experiment_name, test_id)

    os.makedirs(path,exist_ok=True)
    with open(path+"/"+task+".json", "w") as f:
        json.dump({task:regret}, f)