import torch
import torch.nn as nn
import copy
from torch import autograd as ag
import numpy as np
import os

def totorch(x,device):
    if type(x) is tuple:
        return tuple([ag.Variable(torch.Tensor(e)).to(device) for e in x])
    return ag.Variable(torch.Tensor(x)).to(device)    



class FactorizedLayer(nn.Module):

    def __init__(self, n_input, n_hidden, n_latent = 8 ):
        super(FactorizedLayer, self).__init__()

        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.beta = nn.Parameter(torch.rand((n_hidden,n_latent, n_input)), requires_grad=True)
        self.linear = nn.Linear(n_input, n_hidden)

    def forward(self, x):

        beta1=self.beta.view(self.n_hidden,self.n_latent, self.n_input)
        beta2 = self.beta.transpose(1,2)
        c = torch.matmul(beta2,beta1)
        x_tiled = torch.tile(x.unsqueeze(-1),(1,1,self.n_input))
        mask = torch.triu(torch.ones(self.n_input, self.n_input), diagonal=1).to(x_tiled.device)
        factorized_interactions = torch.einsum("ijk,mjk,ikj,jk->im",x_tiled,c,x_tiled,mask)
        linear_output = self.linear(x)

        output = factorized_interactions + linear_output
        
        return output




class MLPEnsemble:

    def __init__(self, path, experiment_name, n_models=100, finetuning_epochs = 1000, finetuning_lr = 0.0001, device="cpu"):

        self.path = os.path.join(path, "..", "models", "fmlp",experiment_name)
        self.experiment_name = experiment_name
        self.n_models = n_models
        self.model_name_list = os.listdir(self.path)
        self.model_name_list = self.model_name_list[:min(len(self.model_name_list), n_models)]
        self.device = device
        self.finetuning_epochs = finetuning_epochs
        self.finetuning_lr = finetuning_lr

        self._load_models()


    def fit_model(self, model, X,y):
        
        temp_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(temp_model.parameters(), lr= self.finetuning_lr)
        mse = torch.nn.MSELoss()

        X, y = totorch(X, self.device), totorch(y, self.device)

        loss_hist = []

        for _ in range(self.finetuning_epochs):

            optimizer.zero_grad()
            pred = temp_model(X).reshape(y.shape)
            loss = mse(pred, y)
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.detach().cpu().numpy().item())

        return temp_model

    def predict(self,X):

        X = totorch(X, self.device)
        y = []
        for model in self.fitted_models:
            model.eval()
            y.append(model(X).detach().cpu().numpy().tolist())

        y = np.array(y)
        return y.mean(axis=0), y.std(axis=0)

    def fit(self, q, y):
        
        self.fitted_models = [self.fit_model(model, q, y) for model in self.models]


    def _load_models(self):

        try:
            self.models = [torch.load(self.path+"/"+model_name) for model_name in self.model_name_list]
            self.n_models = len(self.models)
            print(f"Loaded {len(self.models)} models")
        except Exception as e:

            print(e)
            model1 = torch.load("models/fmlp/fmlp_A01.pt")
            model2 = torch.load("models/fmlp01.pt")
            self.models = [model1, model2]
            self.n_models = 2


class MLP(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, n_output=1, factorized=True):

        super(MLP, self).__init__()

        #assert n_layers>1 , "number of layers should be greater than 2"

        self.n_input = n_input
        self.n_hidden = n_hidden

        if factorized:
            self.fl = FactorizedLayer(n_input, n_hidden)
        else:
            self.fl = nn.Linear(n_input, n_hidden)
        self.ll = nn.ModuleList([nn.Linear(n_hidden,n_hidden) for _ in range(1,n_layers+1)])
        self.output_layer = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fl(x)
        x = self.relu(x)
        
        for layer in self.ll:
            x = layer(x)
            x = self.relu(x)

        x = self.output_layer(x)

        return x


if __name__=="__main__":

    n_input = 2
    n_hidden = 3
    x = torch.rand((4,2))

    fmlp_layer = FactorizedLayer( n_input, n_hidden)
    fmlp = MLP(n_input, n_hidden, 3)
    output = fmlp(x)

    print(output.shape)
    print(output)