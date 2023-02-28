import torch
import numpy as np
import argparse
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import GP
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import PsdSumLazyTensor
from gpytorch.likelihoods import LikelihoodList
from torch.nn import ModuleList
from botorch.sampling.samplers import SobolQMCNormalSampler
from data_loader import data_loader, get_initial_configurations
import os
from botorch.utils.sampling import draw_sobol_samples
import json
from sklearn.preprocessing import MinMaxScaler

def compute_rank_weights(train_x,train_y, base_models, target_model, num_samples, train_yvar):
    """
    Compute ranking weights for each base model and the target model (using 
        LOOCV for the target model). Note: This implementation does not currently 
        address weight dilution, since we only have a small number of base models.
    
    Args:
        train_x: `n x d` tensor of training points (for target task)
        train_y: `n` tensor of training targets (for target task)
        base_models: list of base models
        target_model: target model
        num_samples: number of mc samples
    
    Returns:
        Tensor: `n_t`-dim tensor with the ranking weight for each model
    """
    ranking_losses = []
    # compute ranking loss for each base model
    for task in range(len(base_models)):
        model = base_models[task]
        # compute posterior over training points for target task
        posterior = model.posterior(train_x)
        sampler = SobolQMCNormalSampler(num_samples=num_samples)
        base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
        # compute and save ranking loss
        ranking_losses.append(compute_ranking_loss(base_f_samps, train_y))
    # compute ranking loss for target model using LOOCV
    # f_samps
    target_f_samps = get_target_model_loocv_sample_preds(
        train_x, train_y, train_yvar, target_model, num_samples,
    )
    ranking_losses.append(compute_ranking_loss(target_f_samps, train_y))
    ranking_loss_tensor = torch.stack(ranking_losses)
    # compute best model (minimum ranking loss) for each sample
    best_models = torch.argmin(ranking_loss_tensor, dim=0)
    # compute proportion of samples for which each model is best
    rank_weights = best_models.bincount(minlength=len(ranking_losses)).type_as(train_x) / num_samples
    return rank_weights


def roll_col(X, shift):  
    """
    Rotate columns to right by shift.
    """
    return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)


def compute_ranking_loss(f_samps, target_y):
    """
    Compute ranking loss for each sample from the posterior over target points.
    
    Args:
        f_samps: `n_samples x (n) x n`-dim tensor of samples
        target_y: `n x 1`-dim tensor of targets
    Returns:
        Tensor: `n_samples`-dim tensor containing the ranking loss across each sample
    """
    n = target_y.shape[0]
    if f_samps.ndim == 3:
        # Compute ranking loss for target model
        # take cartesian product of target_y
        cartesian_y = torch.cartesian_prod(
            target_y.squeeze(-1), 
            target_y.squeeze(-1),
        ).view(n, n, 2)
        # the diagonal of f_samps are the out-of-sample predictions
        # for each LOO model, compare the out of sample predictions to each in-sample prediction
        rank_loss = (
            (f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps) ^
            (cartesian_y[..., 0] < cartesian_y[..., 1])
        ).sum(dim=-1).sum(dim=-1)
    else:
        rank_loss = torch.zeros(f_samps.shape[0], dtype=torch.long, device=target_y.device)
        y_stack = target_y.squeeze(-1).expand(f_samps.shape)
        for i in range(1,target_y.shape[0]):
            rank_loss += (
                (roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack)
            ).sum(dim=-1) 
    return rank_loss

def get_fitted_model(train_X, train_Y, train_Yvar, state_dict=None):
    """
    Get a single task GP. The model will be fit unless a state_dict with model 
        hyperparameters is provided.
    """
    Y_mean = train_Y.mean(dim=-2, keepdim=True)
    Y_std = train_Y.std(dim=-2, keepdim=True)
    model = FixedNoiseGP(train_X, (train_Y - Y_mean)/Y_std, train_Yvar)
    model.Y_mean = Y_mean
    model.Y_std = Y_std
    if state_dict is None:
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_model(mll)
    else:
        model.load_state_dict(state_dict)
    return model

def get_target_model_loocv_sample_preds(train_x, train_y, train_yvar, target_model, num_samples):
    """
    Create a batch-mode LOOCV GP and draw a joint sample across all points from the target task.
    
    Args:
        train_x: `n x d` tensor of training points
        train_y: `n x 1` tensor of training targets
        target_model: fitted target model
        num_samples: number of mc samples to draw
    
    Return: `num_samples x n x n`-dim tensor of samples, where dim=1 represents the `n` LOO models,
        and dim=2 represents the `n` training points.
    """
    batch_size = len(train_x)
    device = train_x.device
    masks = torch.eye(len(train_x), dtype=torch.uint8, device=device).bool()
    train_x_cv = torch.stack([train_x[~m] for m in masks])
    train_y_cv = torch.stack([train_y[~m] for m in masks])
    train_yvar_cv = torch.stack([train_yvar[~m] for m in masks])
    state_dict = target_model.state_dict()
    # expand to batch size of batch_mode LOOCV model
    state_dict_expanded = {
        name: t.expand(batch_size, *[-1 for _ in range(t.ndim)])
        for name, t in state_dict.items()
    }
    model = get_fitted_model(train_x_cv, train_y_cv, train_yvar_cv, state_dict=state_dict_expanded)
    with torch.no_grad():
        posterior = model.posterior(train_x)
        # Since we have a batch mode gp and model.posterior always returns an output dimension,
        # the output from `posterior.sample()` here `num_samples x n x n x 1`, so let's squeeze
        # the last dimension.
        sampler = SobolQMCNormalSampler(num_samples=num_samples)
        return sampler(posterior).squeeze(-1)

class RGPE(GP, GPyTorchModel):
    """
    Rank-weighted GP ensemble. Note: this class inherits from GPyTorchModel which provides an 
        interface for GPyTorch models in botorch.
    """
    
    _num_outputs = 1  # metadata for botorch
    
    def __init__(self, models, weights, device):
        super().__init__()
        self.models = ModuleList(models).to(device)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "RGPE currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models]).to(device)
        self.weights = weights
        self.to(weights)
        
    def forward(self, x):
        weighted_means = []
        weighted_covars = []
        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights**2 > 0).nonzero()
        non_zero_weights = self.weights[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()
        
        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            model = self.models[raw_idx]
            posterior = model.posterior(x)
            # unstandardize predictions
            posterior_mean = posterior.mean.squeeze(-1)*model.Y_std + model.Y_mean
            posterior_cov = posterior.mvn.lazy_covariance_matrix * model.Y_std.pow(2)
            # apply weight
            weight = non_zero_weights[non_zero_weight_idx]
            weighted_means.append(weight * posterior_mean)
            weighted_covars.append(posterior_cov * weight**2)
        # set mean and covariance to be the rank-weighted sum the means and covariances of the
        # base models and target model
        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLazyTensor(*weighted_covars)
        return MultivariateNormal(mean_x, covar_x)


def load_config (rootdir, experiment_id):

    try:
        with open(rootdir+"/../configurations/"+experiment_id+".json") as f:
            config = json.load(f)
    except:
        config = {"noise_std": 0.05, "num_training_points":1000, "num_posterior_points":256}

    return config


def train(pipelines, perf_matrix, train_index, experiment_id):
        
    #dtype = torch.double
    #dim, n_pipelines = pipelines.shape
    rootdir     = os.path.dirname(os.path.realpath(__file__))
    

    config = load_config(rootdir, experiment_id)
    num_training_points = config["num_training_points"]
    noise_std = config["noise_std"]

    np.random.seed(0)
    checkpoint_path = rootdir+"/../models/RGPE/"+experiment_id+"/"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    data_by_task = {}
    for task in train_index:
        sample_ids = np.where(~np.isnan(perf_matrix[task,:]))[0]

        if len(sample_ids) >0:
            np.random.shuffle(sample_ids)
            sample_ids = sample_ids[:num_training_points]
            train_x = torch.DoubleTensor(pipelines[sample_ids])
            train_y = torch.DoubleTensor(MinMaxScaler().fit_transform(perf_matrix[task,sample_ids].reshape(-1,1)))
            train_yvar = torch.full_like(train_y, noise_std**2)
            data_by_task[str(task)] = {
                # scale x to [0, 1]
                'train_x': train_x,
                'train_y': train_y,
                'train_yvar': train_yvar,
            }         

    base_model_list = []
    for task in train_index:
        task = str(task)

        try:
            print(f"Fitting base model {task}")
            model = get_fitted_model(
                data_by_task[task]['train_x'], 
                data_by_task[task]['train_y'], 
                data_by_task[task]['train_yvar'],
            )
            base_model_list.append(model)  
        except Exception as e:

            print(e)
            del data_by_task[task]


    checkpoint_path = os.path.join(rootdir,"..","models", "RGPE", experiment_id)
    os.makedirs(checkpoint_path,exist_ok=True)

    base_models_state_dict = [base_model.state_dict() for base_model in base_model_list]
    torch.save(base_models_state_dict, checkpoint_path+"/model.pt")
    torch.save(data_by_task, checkpoint_path+"/data_by_task.pt")
    
    #config= {"noise_std": noise_std, "num_posterior_samples": num_posterior_samples, "checkpoint_path":  checkpoint_path}
    
    #with open(rootdir+"/../configurations/"+experiment_id+".json", "w") as f:
    #    json.dump(config, f)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--experiment_id', type=str, default="RGPE1")
    parser.add_argument('--dataset_name', type=str, default="tensor-oboe")
    parser.add_argument('--debug', type=int, default=0)
    

    args = parser.parse_args()
    print(args)

    fold = args.fold
    debug = args.debug
    experiment_id = args.experiment_id
    dataset_name = args.dataset_name

    rootdir     = os.path.dirname(os.path.realpath(__file__))
    pipelines, perf_matrix, train_index, test_index, init_ids = data_loader(dataset_name, rootdir, fold=fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if debug:
        train_index = train_index[:5]

    train(pipelines, perf_matrix, train_index, experiment_id)
   