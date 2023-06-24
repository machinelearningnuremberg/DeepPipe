from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.surrogates.RandomForest import RandomForest
from pyGPGO.covfunc import matern32
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from pybnn.bohamiann import Bohamiann
from pybnn import DNGO
from sklearn.preprocessing import MinMaxScaler
import torch
from rgpe import get_fitted_model, compute_rank_weights, RGPE
import copy
import os
from data_loader import data_loader
from rgpe import train as train_rgpe
from skl import StructuredKernelLearning

class BaseSurrogate:

    def __init__(self):

        self.model = None
    
    def fit(self, X,y):

        assert self.model is not None, "Model not implemented"
        self.model.fit(X,y)

    def predict(self, X):

        assert self.model is not None, "Model not implemented"
        pred_mean, pred_std = self.model.predict(X)

        return pred_mean, pred_std

class RandomForestWithUncertainty(BaseSurrogate):

    def __init__(self,**args):
        super(RandomForestWithUncertainty, self).__init__()

        self.model = RandomForestRegressor(**args)
    
    def predict(self, X):

        pred = np.array([tree.predict(X) for tree in self.model]).T
        pred_mean = np.mean(pred, axis=1)
        pred_var = (pred - pred_mean.reshape(-1,1))**2
        pred_std = np.sqrt(np.mean(pred_var, axis=1))

        return pred_mean, pred_std


class DNGOSurrogate:

    def __init__(self, **args):

        self.model = DNGO(do_mcmc=False)

    def fit(self, X, y):

        dummy_X = np.ones((1,X.shape[1]))*-1
        dummy_y = np.array([-1])
        X = np.concatenate((X, dummy_X), axis=0)
        y = np.concatenate((y, dummy_y), axis=0)

        self.model.train(X,y, do_optimize=True)

    def predict(self,X):

        pred_mean, pred_var = self.model.predict(X)
        return pred_mean, np.sqrt(pred_var)


class BohamiannSurrogate:

    def __init__(self,  **args):

        super(BohamiannSurrogate, self).__init__()

        self.num_steps = args.get("num_steps",20000)
        self.num_burn_in_steps = args.get("num_burn_in_steps", 2000)
        self.keep_every = args.get("keep_every", 50)
        self.lr = args.get("lr", 1e-2)
        self.verbose = args.get("verbose", True)
        self.print_every_n_steps = args.get("print_every_n_steps", 1000)

        self.model = Bohamiann(print_every_n_steps = self.print_every_n_steps)

    def fit(self, X,y):

        dummy_X = np.ones((1,X.shape[1]))*-1
        dummy_y = np.array([-1])
        X = np.concatenate((X, dummy_X), axis=0)
        y = np.concatenate((y, dummy_y), axis=0)

        self.model.train(X, y.reshape(-1,1), num_steps = self.num_steps,
                                    num_burn_in_steps = self.num_burn_in_steps,
                                    keep_every = self.keep_every,
                                    lr = self.lr,
                                    verbose = self.verbose)
    

    def predict(self, X):

        pred_mean, pred_var = self.model.predict(X)
        return pred_mean, np.sqrt(pred_var)


class GaussianProcessSurrogate(BaseSurrogate):

    def __init__(self, kernel = "matern",  **args):

        super(GaussianProcessSurrogate, self).__init__()

        if kernel == "rbf":
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0)
        elif kernel == "matern":
            Matern(length_scale=1.0, nu=1.5)
        else:
            print("Not specified kernel")
            kernel = None

        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=500)
        

    def predict(self, X):

        pred_mean, pred_std = self.model.predict(X, return_std = True)
        return pred_mean, pred_std


class RGPESurrogate:

    def __init__(self, **args):
        self.experiment_id = args.get("experiment_id","RGPE1")
        self.num_posterior_samples = args.get("num_posterior_samples", 256)
        #self.device = args.get("device", "cpu")
        self.fold = args.get("fold", 0)
        self.dataset_name = args.get("dataset_name", "tensor-oboe")
        self.noise_std = args.get("noise_std", 0.05)
        self.meta_train = args.get("meta_train", 1)
        self.debug = args.get("debug", 0)
        self.n_tasks = args.get("n_tasks", None)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



        assert self.experiment_id is not None, print("Please specify experiment id")

        self.rootdir     = os.path.dirname(os.path.realpath(__file__))
        self.checkpoint_path = os.path.join(self.rootdir,"..","models", "RGPE", self.experiment_id)

        if not os.path.exists(self.checkpoint_path+"/model.pt"):
            self._meta_train()

        self.base_models_state_dict = list(torch.load(self.checkpoint_path+"/model.pt"))
        self.data_by_task = torch.load(self.checkpoint_path+"/data_by_task.pt")

        if self.n_tasks is None:
            self.n_tasks = len(self.data_by_task.keys())
        
        tasks = list(self.data_by_task.keys())[:self.n_tasks]

        self.base_model_list = [get_fitted_model(self.data_by_task[task]["train_x"].to(self.device),
                                                self.data_by_task[task]["train_y"].to(self.device), 
                                                self.data_by_task[task]["train_yvar"].to(self.device), 
                                                state_dict).to(self.device) 
                                                for task,state_dict 
                                                in zip(tasks, self.base_models_state_dict[:self.n_tasks])]

    def fit(self, X, y):
        
        X = torch.DoubleTensor(X).to(self.device)
        y = MinMaxScaler().fit_transform(y.reshape(-1,1))
        y = torch.DoubleTensor(y).to(self.device).reshape(-1,1)
        #if y.std() == 0:
        y += torch.rand(y.shape).to(y.device)*0.00001
        yvar = torch.full_like(y, self.noise_std**2).to(self.device)
        target_model = get_fitted_model(X,y,yvar)
        #base_model_list = torch.ModuleList([copy.deepcopy(target_model) for _ in range(self.n_base_models)])
        model_list = self.base_model_list + [target_model]
        rank_weights = compute_rank_weights(
            X, 
            y, 
            self.base_model_list, 
            target_model, 
            self.num_posterior_samples,
            yvar
        )
        self.model = RGPE(model_list, rank_weights, device=self.device)

    def predict(self, X):
        
        X = torch.DoubleTensor(X).to(self.device)
        output = self.model(X)

        return output.mean.detach().cpu().numpy(), output.stddev.detach().cpu().numpy()

    def _meta_train(self):

        pipelines, perf_matrix, train_index, _, _, _ = data_loader(self.dataset_name, self.rootdir, fold=self.fold)
        
        if self.debug:
            train_index = train_index[:5]
        train_rgpe(pipelines, perf_matrix, train_index, self.experiment_id)



def create_surrogate(model_name, **args):

    if model_name == "GP":
        model = GaussianProcessSurrogate(**args)

    elif model_name == "RFPY":
        model = RandomForest(**args)

    elif model_name == "RFSK":
        model = RandomForestWithUncertainty(**args)

    elif model_name == "BOHAMIANN":
        model = BohamiannSurrogate(**args)

    elif model_name == "DNGO":
        model = DNGOSurrogate(**args)

    elif model_name == "RGPE":
        model = RGPESurrogate(**args)

    elif model_name == "SKL":
        model = StructuredKernelLearning(**args)

    else:
        model = None
        print("No surrogate")

    return model
