from torch.nn.utils.prune import custom_from_mask 
from torch import nn
import torch
import numpy as np
import time
#from AutoMLDeepKernelGPModules import ReptileModel
from torch.utils.tensorboard import SummaryWriter
import gpytorch
import logging
import os
import copy


class MaskedMLP (nn.Module):

    def __init__(self, masks, list_for_concatenator, concat_matrix, hidden_dim, algorithm_domain, hidden_dim_per_stage = None,  
                n_non_masked_layers = 2, out_features = 1, omit_estimator = -1, device="cpu"):
        super(MaskedMLP, self).__init__()

        masked_layers = []
        non_masked_layers = []
        n_list_for_concatenator = len(list_for_concatenator)

        for mask in masks:
            hidden_dim1, hidden_dim2 = mask.shape
            linear = nn.Linear(hidden_dim2, hidden_dim1).to(device)
            linear_masked = custom_from_mask(linear, name='weight', mask=mask)
            masked_layers.append(linear_masked) #encoders layers
            
        
        self.hidden_dim_masked = concat_matrix.shape[1]

        non_masked_layers.append(nn.Linear(self.hidden_dim_masked,hidden_dim).to(device))
        for i in range(1, n_non_masked_layers-1):
            non_masked_layers.append(nn.Linear(hidden_dim,hidden_dim).to(device))
        non_masked_layers.append(nn.Linear(hidden_dim,out_features).to(device))  #aggregator layers

        self.masked_layers = nn.ModuleList(masked_layers)
        self.non_masked_layers = nn.ModuleList(non_masked_layers)
        self.select_masks = [torch.eye(len(i), device=device) for i in list_for_concatenator]
        self.hidden_dim_per_stage = [hidden_dim for _ in range(len(list_for_concatenator))] if hidden_dim_per_stage is None else hidden_dim_per_stage
        self.concat_matrix =  concat_matrix.double()
        self.relu = nn.ReLU()
        self.device = device
        self.masks = masks
        self.list_for_concatenator = list_for_concatenator
        self.n_non_masked_layers = n_non_masked_layers
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.n_list_for_concatenator = n_list_for_concatenator
        self.algorithm_domain = algorithm_domain
        self.omit_estimator = omit_estimator
        
        self.hidden = torch.ones(1,self.hidden_dim_masked,device=device)

        
    def init_weights(self):

        for layer in self.masked_layers:

            layer.weight_orig.data.fill_(1)
            layer.bias.data.fill_(1)

        for layer in self.non_masked_layers:

            layer.weight.data.fill_(1)
            layer.bias.data.fill_(1)

    def precompute_mask(self,select_mask_id):
 
        out = []
        for i, mask in enumerate(self.select_masks):
            out.append(torch.kron(mask[select_mask_id[:,i],:], torch.ones(1, self.hidden_dim_per_stage[i], device=self.device)))

        self.out = torch.cat(out, axis=1)

    def initialize_encoders_with_high_std(self):

        for i, mask in enumerate(self.masks):
            for name, param in self.masked_layers[i].named_parameters():

                if "weight" in name:
                    param.data.normal_(0.0,5)

    def freeze_masked_encoder_layer(self, high_std = True):
 
        ix_estimator = self.list_for_concatenator[-1][int(self.omit_estimator)]
        ix1 = self.algorithm_domain[ix_estimator]
        ix2 = self.algorithm_domain[ix_estimator+1]

        for i, mask in enumerate(self.masks):

            submask= mask[:,ix1:ix2]
            ix3, ix4 = np.where(submask[:,0].cpu()==1)[0][[0,-1]]
            temp_mask = torch.zeros(mask.shape).to(self.device)
            temp_mask[ix3:ix4, ix1:ix2] = 1

            for name, param in self.masked_layers[i].named_parameters():

                if "weight" in name:
                    if high_std:
                        param.data[ix3:ix4, ix1:ix2].data.normal_(0.0,3)
                    param.grad = torch.multiply(temp_mask, param.grad)
                else:
                    param.grad = torch.multiply(temp_mask[:,ix1], param.grad)
                   
            ix1, ix2 = ix3, ix4


    def forward(self,x, select_mask_id = None):

        for layer in self.masked_layers:
            x = self.relu(layer(x))

        if len(self.masks)>0:
            if select_mask_id is not None:
                out = []
                for i, mask in enumerate(self.select_masks):
                    out.append(torch.kron(mask[select_mask_id[:,i],:], torch.ones(1, self.hidden_dim_per_stage[i], device=self.device)))

                out = torch.cat(out, axis=1)
            else:
                out = self.out

            x = torch.mul(out, x) #masks the activations before the concatenation
            x = torch.matmul( x, self.concat_matrix) # performs the concatenation

        for layer in self.non_masked_layers[:-1]:
            x = self.relu(layer(x))
        x = self.non_masked_layers[-1](x)

        return x



class Encoder(nn.Module):

    def __init__(self, n_input, hidden_dim, n_layers,  device="cuda"):
        super(Encoder, self).__init__()
        layer_size = [n_input]+[hidden_dim for i in range(n_layers)]

        self.layers = nn.ModuleList([nn.Linear(layer_size[i], layer_size[i+1]).to(device) for i in range(n_layers)])
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x



class DeepPipe(nn.Module):
    def __init__(self, training_tasks, validation_tasks, pipelines, error_matrix, feature_extractor, list_for_selector, 
                        checkpoint_path, batch_size = 1000, test_batch_size = 100, kernel = "matern", nu= 2.5 , ard = 1, device="cpu"):

        super(DeepPipe, self).__init__()

        self.training_tasks = training_tasks
        self.validation_tasks = validation_tasks
        self.performance_matrix = 1-error_matrix
        self.pipelines = pipelines
        self.n_training_tasks = len(training_tasks)
        self.n_validation_tasks = len(validation_tasks)
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.kernel = kernel
        self.nu = nu
        self.ard = ard

        assert self.checkpoint_path!=None, "Provide a configuration path"
        
        os.makedirs(self.checkpoint_path,exist_ok=True)
        self.device = device
        logging.basicConfig(filename=os.path.join(self.checkpoint_path,"log.txt"), level=logging.DEBUG)
        self.feature_extractor = feature_extractor.double()
        self.get_model_likelihood_mll(self.batch_size)
        self.mse        = nn.MSELoss()
        self.curr_valid_loss = np.inf
        self.setup_writers()
        self.list_for_selector = torch.LongTensor(list_for_selector).to(self.device)
    
        self.train_metrics = Metric()
        self.valid_metrics = Metric(prefix="valid: ")
        print(self)


    def setup_writers(self,):

        train_log_dir = os.path.join(self.checkpoint_path,"train")
        os.makedirs(train_log_dir,exist_ok=True)
        self.train_summary_writer = SummaryWriter(train_log_dir)
        
        valid_log_dir = os.path.join(self.checkpoint_path,"valid")
        os.makedirs(valid_log_dir,exist_ok=True)
        self.valid_summary_writer = SummaryWriter(valid_log_dir)        
    

    def get_model_likelihood_mll(self, train_size):
        
        train_x=torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y=torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = ExactGPLayer(train_x=train_x, 
                                    train_y=train_y, 
                                    likelihood=likelihood, 
                                    kernel = self.kernel, 
                                    ard = self.ard,
                                    nu = self.nu,
                                    dims = self.feature_extractor.out_features)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device).double()


    def train(self, epoch, optimizer, verbose = False):

        self.model.train()
        self.feature_extractor.train()
        self.likelihood.train()

        for task in self.training_tasks:
            #task =np.random.choice(self.training_tasks, size=1).item()
            evaluated_pipelines = torch.where(~torch.isnan(self.performance_matrix[task,:]))

            while len(evaluated_pipelines[0])==0:
                task =np.random.choice(self.training_tasks, size=1).item()
                evaluated_pipelines = torch.where(~torch.isnan(self.performance_matrix[task,:])) 
                               
            batch_size = self.batch_size
            sampled_pipelines = np.random.choice(evaluated_pipelines[0].cpu().numpy(),batch_size)
            labels = self.performance_matrix[task, sampled_pipelines].to(self.device)
            inputs = self.pipelines[sampled_pipelines,:].to(self.device)
            select_mask_id = self.list_for_selector[sampled_pipelines].to(self.device)

            try:
                optimizer.zero_grad()

                z = self.feature_extractor(inputs, select_mask_id)
                self.model.set_train_data(inputs=z, targets=labels, strict=False)
                predictions = self.model(z)
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()

                optimizer.step()
                mse = self.mse(predictions.mean, labels)

                if verbose:
                    print(loss)
                self.train_metrics.update(loss,self.model.likelihood.noise,mse)
                
            except Exception as e:
                print(e)
                self.train_metrics.update(np.array(np.nan),np.array(np.nan),np.array(np.nan))


        training_results = self.train_metrics.get()
        for k,v in training_results.items():
            self.train_summary_writer.add_scalar(k, v, epoch)
        for task in self.validation_tasks:
            try:
                mse,loss, loss2 = self.test(task)
                loss = loss.detach().cpu().numpy()
                loss2 = loss2.detach().cpu().numpy()

                if np.isnan(loss):
                    print("NaN loss")
            except Exception as e:
                mse = np.array(np.nan)
                loss = np.array(np.nan)
                loss2 = np.array(np.nan)
                print(e)
            self.valid_metrics.update(loss,loss2,mse,)
            
        logging.info(self.train_metrics.report() + " " + self.valid_metrics.report())
        validation_results = self.valid_metrics.get()
        for k,v in validation_results.items():
            self.valid_summary_writer.add_scalar(k, v, epoch)
        self.feature_extractor.train()
        self.likelihood.train()
        self.model.train()
        
        if validation_results["loss"] < self.curr_valid_loss:
            self.save_checkpoint(os.path.join(self.checkpoint_path,"weights"))
            self.curr_valid_loss = validation_results["loss"]
        self.valid_metrics.reset()       
        self.train_metrics.reset()

    def sample_pipelines(self, task, evaluated_pipelines, batch_size):
        sampled_pipelines = np.random.choice(evaluated_pipelines[0].cpu().numpy(),batch_size)
        labels = self.performance_matrix[task, sampled_pipelines].to(self.device)
        inputs = self.pipelines[sampled_pipelines,:].to(self.device)
        select_mask_id = self.list_for_selector[sampled_pipelines].to(self.device)

        return labels, inputs, select_mask_id


    def test(self, task): 
        evaluated_pipelines = torch.where(~torch.isnan(self.performance_matrix[task,:]))[0].cpu().numpy()
        support_idx = np.random.choice(evaluated_pipelines,
                                            replace=False,size=min(self.batch_size, 50))
        query_idx = np.random.choice(
        np.setdiff1d(evaluated_pipelines,support_idx),replace=False,size=self.test_batch_size)  
        x_support = self.pipelines[support_idx,]
        x_query = self.pipelines[query_idx,]
        y_support = self.performance_matrix[task, support_idx]
        y_query = self.performance_matrix[task, query_idx]
        support_mask_id = self.list_for_selector[support_idx,]
        query_mask_id = self.list_for_selector[query_idx,]

        z_support = self.feature_extractor(x_support, support_mask_id).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)
        self.model.eval()        
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():

            predictions = self.model(z_support)
            loss2 = -self.mll(predictions, self.model.train_targets)
            

            z_query = self.feature_extractor(x_query, query_mask_id).detach()
            pred    = self.likelihood(self.model(z_query))
            loss = -self.mll(pred, y_query)

        mse = self.mse(pred.mean, y_query)
        return mse,loss, loss2

        
    def save_checkpoint(self, checkpoint):
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])




class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, ard , nu, dims ):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)

        self.mean_module  = gpytorch.means.ConstantMean()
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        self.dims = dims
        self.kernel = kernel
        self.ard = ard
        self.nu = nu
        
        ## RBF kernel
        if(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dims if ard else None))
        elif(kernel=='matern'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu,ard_num_dims=dims if ard else None))
        ## Spectral kernel
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")
            
    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


class DeepKernelGP(nn.Module):

    def __init__(self, X, Y, kernel, nu, feature_extractor, support,log_dir, list_for_selector, ard = 1, patience = 16, device = "cpu"):

        super(DeepKernelGP, self).__init__()
        ## GP parameters
        self.device = device
        self.X,self.Y  = X,Y
        self.feature_extractor = feature_extractor.to(self.device)
        self.list_for_selector = list_for_selector
        self.kernel = kernel
        self.ard = ard
        self.nu = nu
        self.patience = patience
        self.get_model_likelihood_mll(len(support))
        logging.basicConfig(filename=log_dir, level=logging.DEBUG)

    def get_model_likelihood_mll(self, train_size):
        train_x=torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y=torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = ExactGPLayer(train_x=train_x, 
                                    train_y=train_y, 
                                    likelihood=likelihood, 
                                    kernel = self.kernel, 
                                    ard = self.ard, 
                                    nu = self.nu,
                                    dims = self.feature_extractor.out_features)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device).double()


    def train(self, support, load_model,optimizer, checkpoint=None, epochs=1000, verbose = False, tol = 0.0001, freeze_backbone = False):
        if load_model:
            assert(checkpoint is not None)
            if verbose:
                print("KEYS MATCHED")
            self.load_checkpoint(os.path.join(checkpoint,"weights"))

        if freeze_backbone:
        
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False   

        losses = [np.inf]
        best_loss = np.inf
        
        initial_weights = copy.deepcopy(self.state_dict())
        weights = copy.deepcopy(self.state_dict())
        epoch = 0
        patience=0
        max_patience = self.patience
        select_mask_id = self.list_for_selector[support]
        inputs = self.X[support]
        labels = self.Y[support]   

        self.feature_extractor.precompute_mask(select_mask_id)

        self.feature_extractor.train()
        self.model.train()
        self.likelihood.train()

        starttime = time.time()
        for epoch in range(epochs):

            optimizer.zero_grad()
            z = self.feature_extractor(inputs)
            self.model.set_train_data(inputs=z, targets=labels)
            predictions = self.model(z)
            try:
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                        
                if self.feature_extractor.omit_estimator != -1:
                    self.feature_extractor.freeze_masked_encoder_layer(False) 

                optimizer.step()
            except Exception as ada:
                logging.info(f"Exception {ada}")
                break

            if verbose:
                print("Iter {iter}/{epochs} - Loss: {loss:.5f}   noise: {noise:.5f} ".format(
                    iter=epoch+1,epochs=epochs,loss=loss.item(),noise=self.likelihood.noise.item()))                
            

            losses.append(loss.detach().to("cpu").item())
            if best_loss>losses[-1]:
                best_loss = losses[-1]
                weights = copy.deepcopy(self.state_dict())

            if np.allclose(losses[-1],losses[-2],atol=tol):
                patience+=1
                if patience>max_patience:
                    break
            else:
                patience=0
        endtime = time.time()
        self.load_state_dict(weights)
        logging.info(f"Current Iteration: {len(support)} | Incumbent {max(self.Y[support])} | Duration {np.round(endtime-starttime,2)} | Epochs {epoch} | Noise {self.likelihood.noise.item()}")
        return losses,weights,initial_weights
    
    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint,map_location=torch.device(self.device))
        self.model.load_state_dict(ckpt['gp'],strict=False)
        self.likelihood.load_state_dict(ckpt['likelihood'],strict=False)

        self.feature_extractor.load_state_dict(ckpt['net'],strict=False)

    def predict(self,support, query):
        x_support,y_support = self.X[support].to(self.device), self.Y[support].to(self.device)
        x_query = self.X[query].to(self.device)
        spt_mask_id = self.list_for_selector[support]
        qry_mask_id = self.list_for_selector[query]

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()        

        z_support = self.feature_extractor(x_support, spt_mask_id).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

        with torch.no_grad():
            z_query = self.feature_extractor(x_query, qry_mask_id).detach()
            pred    = self.likelihood(self.model(z_query))

        mu    = pred.mean.detach().to("cpu").numpy().reshape(-1,)
        stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1,)
        
        return mu,stddev
    


    
class Metric(object):
    def __init__(self, prefix='train: ', n=16):

        if prefix=="train: ":
            self.message=prefix + "loss: {loss:.2f} - noise: {log_var:.2f} - mse: {mse:.4f}"
        else:
            self.message=prefix + "loss spt.: {loss:.2f} - loss qry.: {log_var:.2f} - mse: {mse:.4f}"
        
        self.n = n
        self.reset()
        
    def update(self,loss,noise,mse):
        self.loss.append(np.asscalar(loss))
        self.noise.append(np.asscalar(noise))
        self.mse.append(np.asscalar(mse))
    
    def reset(self,):
        self.loss = []
        self.noise = []
        self.mse = []
    
    def report(self):
        return self.message.format(loss=np.nanmean(self.loss),
                            log_var=np.nanmean(self.noise),
                            mse=np.nanmean(self.mse))
    
    def get(self):
        return {"loss":np.nanmean(self.loss),
                "noise":np.nanmean(self.noise),
                "mse":np.nanmean(self.mse)}

