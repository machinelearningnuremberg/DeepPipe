from fmlp import MLPEnsemble, MLP
import numpy as np
import torch 
import json
import os
import pandas as pd
from torch import autograd as ag
import argparse
from fmlp_loaders import PMFMetaDatasetLoader, TOboeMetaDatasetLoader


#fmlp.to(device)
def train(batch_sampler, model, model_name,
             lr = 0.0001, epochs=10000, device="cpu", 
             n_val_iters = 100, batch_size=64, print_interval = 1000):

    optimizer = torch.optim.Adam(model.parameters(), lr= lr)
    mse = torch.nn.MSELoss()       
    loss_hist = []
    best_p = np.inf
    for epoch in range(epochs):

        optimizer.zero_grad()
        X, y = batch_sampler(batch_size)
        X, y = totorch(X, device), totorch(y, device)

        pred = model(X).reshape(y.shape)
        loss = mse(pred, y)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.detach().cpu().numpy().item())

        #compute validation error
        with torch.no_grad():
            total_error = 0

            for val_iter in range(n_val_iters):
                X, y = batch_sampler(batch_size=batch_size, val=True)
                X, y = totorch(X, device), totorch(y, device)
                pred = model(X).reshape(y.shape)
                loss = mse(pred, y)
                total_error+=loss.detach().cpu().numpy().item()

            total_val_error = total_error/n_val_iters
            

        if total_val_error<best_p:
            best_p = total_val_error
            torch.save(model, model_name)


        if (epoch % print_interval) == 0:
            print("Val:", total_val_error)

    return total_val_error

def totorch(x,device):
    if type(x) is tuple:
        return tuple([ag.Variable(torch.Tensor(e)).to(device) for e in x])
    return ag.Variable(torch.Tensor(x)).to(device)    


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_hidden', default=5, type=int, help="hidden neurons")
    parser.add_argument('--n_layers', default=0, type=int, help="additional layers to the factorized layer")
    parser.add_argument('--factorized', default=1, type=int)
    parser.add_argument('--n_val_iters', default=100, type=int)
    parser.add_argument('--experiment_name', default="exp01", type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--surrogate_index', default=1, type=int)
    parser.add_argument('--print_interval', default=1000, type=int)
    parser.add_argument('--dataset', default="oboe-dataset", type=str)

    args = parser.parse_args()
    print(args)

    #hyperparaters
    batch_size = args.batch_size
    n_hidden = args.n_hidden
    n_layers = args.n_layers
    factorized = args.factorized
    epochs = args.epochs
    n_val_iters = args.n_val_iters
    experiment_name = args.experiment_name
    learning_rate = args.learning_rate
    surrogate_index = args.surrogate_index
    print_interval = args.print_interval
    dataset = args.dataset

    n_output = 1


    ##loading data
    rootdir     = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(rootdir,"..", "data")

    #prepare conf
    models_dir = os.path.join(rootdir, "..", "models", "fmlp", experiment_name)+"/"
    os.makedirs(models_dir,exist_ok=True)
    model_name =  os.path.join(models_dir,str(surrogate_index)+".pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    #model_creationg
    if dataset=="oboe-dataset":
        metadataset = TOboeMetaDatasetLoader(path)
    else:
        metadataset = PMFMetaDatasetLoader(path)
    
    fmlp_model = MLP(metadataset.get_total_dim(), n_hidden, n_layers, n_output, factorized=factorized).to(device)

    train(    batch_sampler=  metadataset.batch_sampler, 
                model = fmlp_model, 
                model_name = model_name,
                batch_size = batch_size,
                lr = learning_rate, 
                epochs=epochs, 
                device=device, 
                print_interval=print_interval,
                n_val_iters = n_val_iters)

