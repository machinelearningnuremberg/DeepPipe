import numpy as np
import pandas as pd
import json
import openml
import sys
import os
import copy
import deeppipe_api.experiments.baselines.oboe.pipeline as pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import VotingClassifier
from scipy import stats
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")

import argparse
import torch
from deeppipe_api.modules import MaskedMLP, DeepKernelGP
import numpy as np
import time
from deeppipe_api.utils import get_pipelines_settings_on_dataset, get_response, generate_settings, load_data, get_scores
from deeppipe_api.utils import get_matrix_and_masks
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

class DeepPipe:

    def __init__(self, lr = 0.0001, 
                        experiment_id = "DeepPipeTOboe2", 
                        freeze_backbone=True, 
                        time_limit=600,
                        epochs=0,
                        n_iters = 100,
                        meta_trained = 1,
                        device = None,
                        verbose = False,
                        create_ensemble = False,
                        ensemble_size = 10,
                        apply_cv = False,
                        get_dummies = True):
        rootdir     = os.path.dirname(os.path.realpath(__file__))
        
        if experiment_id is None:
            #generate a random id
            experiment_id = str(int(time.time()))

        with open(os.path.join(rootdir, "experiments", "data", "classification.json")) as f:
            configs = json.load(f)
        with open(os.path.join(rootdir, "experiments", "data", "tensor_oboe_meta_dataset.json")) as f:
            data = json.load(f)
        with open(os.path.join(rootdir,"experiments","configurations","DeepPipeTOboe.json")) as f:
            configuration = json.load(f)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.time_limit = time_limit
        self.n_iters = n_iters
        self.meta_trained = meta_trained
        self.epochs = epochs
        self.freeze_backbone = freeze_backbone
        self.verbose = verbose
        self.lr = lr
        self.create_ensemble = create_ensemble
        self.ensemble_size = ensemble_size
        self.apply_cv = apply_cv
        self.hidden_dim_factor = configuration["hidden_dim_factor"]
        self.hidden_dim = configuration["hidden_dim"]
        self.encoder_layers = configuration["encoder_layers"]
        self.aggregated_layers = configuration["aggregated_layers"]
        self.out_features = configuration["out_features"]
        self.kernel = configuration["kernel"]
        self.get_dummies = get_dummies

        if self.kernel == "matern":
            self.nu = 2.5
        else:
            self.nu = None
        self.original_pipelines = torch.FloatTensor(data["pipelines"]).to(self.device)
        self.list_for_concatenator = data["list_for_concatenator"]
        self.algorithm_domain = data["algorithm_domain"]
        self.list_for_selector = torch.LongTensor(data["list_for_selector"]).to(self.device)
        folds_index = data["tasks_index"]
        self.init_ids = data["init_ids"]
        self.settings = generate_settings(configs)

        log_dir     = os.path.join(rootdir,"logs", experiment_id)
        os.makedirs(log_dir,exist_ok=True)
        self.logger = os.path.join(log_dir,f"{all}.txt")
        self.checkpoint_path = os.path.join(rootdir,"experiments", "checkpoints",experiment_id)
        os.makedirs(self.checkpoint_path,exist_ok=True)

    def preprocess(self, X, y=None):
        if self.get_dummies:
            X = pd.get_dummies(X, drop_first=True)
            if y is not None:
                y = np.array(y).reshape(-1)
                return X, y
        return X

    def fit(self, X, y):
        X, y = self.preprocess(X, y)
        num_features_tr = X.shape[1]
        num_points_tr = X.shape[0]
        pipelines_settings = self.get_pipelines_settings_on_dataset(self.settings, num_points_tr, num_features_tr)
        
        #split train and test
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        categorical = np.array(X_train.dtypes=="category")
        output = get_matrix_and_masks(self.hidden_dim, self.original_pipelines, self.list_for_concatenator, self.algorithm_domain, 
                                      self.list_for_selector, self.encoder_layers, self.hidden_dim_factor, self.device)
        
        concat_matrix, masks, hidden_dim, hidden_dim_per_stage, pipelines = output
        self.hidden_dim = hidden_dim
        if np.sum(categorical)>0:

            valid_pipelines_id = []
            valid_pipelines = []

            for i, setting in enumerate(pipelines_settings):
                if (setting["imputer"]["hyperparameters"]["strategy"] in ["most_frequent", "constant"]) and (setting["encoder"]["algorithm"] is not None):
                    valid_pipelines_id.append(i)
                    valid_pipelines.append(setting)
                
        else:
            valid_pipelines_id = np.arange(len(pipelines_settings)).tolist()
            valid_pipelines = pipelines_settings

        Lambda = pipelines.to(self.device).double()[valid_pipelines_id]
        original_x = np.array(valid_pipelines_id)[[7208, 3429, 13062, 5740, 22397]]
        x= []
        response = torch.zeros((Lambda.shape[0],1)).double().to(self.device)
        on_time = True
        history_list = []
        start_time = time.time()

        if self.create_ensemble:
            fitted_pipelines = []

        print("Observing initial recommended pipelines...")
        for pipeline_id in original_x:
            try:
                if on_time:
                    new_idx = valid_pipelines_id.index(pipeline_id)
                    data = (X_train, y_train, X_val, y_val)

                    if self.create_ensemble:
                        _ , temp_model = get_response(data, pipeline_id,  valid_pipelines,  categorical, apply_cv=False)
                        fitted_pipelines.append(temp_model)
                    temp_response, _ = get_response(data, new_idx, valid_pipelines, categorical, self.apply_cv)
                    #current_time = time.time()
                    spent_time = time.time()-start_time
                    if temp_response!=0.5:

                        if self.verbose:
                            print(temp_response.item())
                        response[new_idx] = temp_response
                        x.append(new_idx)
                    info_tuple =  (spent_time, temp_response, torch.max(response).detach().cpu().numpy(), new_idx)
                    history_list.append(info_tuple)
                else:
                    break
                
                if spent_time>self.time_limit: 
                    on_time=False
                    break
            except Exception as e:
                print(e)
                temp_response = 0.0

        q = Lambda[x].double()
        y = response

        print("Exploring new pipelines...")
        if on_time: 
            pbar = tqdm(range(self.n_iters))
            for bo_iter in pbar:
                q = Lambda[x].double()
                y = response
                feature_extractor = MaskedMLP(masks, self.list_for_concatenator, 
                                                    hidden_dim=self.hidden_dim, 
                                                    concat_matrix=concat_matrix, 
                                                    hidden_dim_per_stage=hidden_dim_per_stage,
                                                    n_non_masked_layers = self.aggregated_layers, 
                                                    out_features = self.out_features,
                                                    algorithm_domain = self.algorithm_domain,
                                                    device=self.device)
                model     = DeepKernelGP(X = Lambda,
                                        Y = response.reshape(-1,),
                                        log_dir=self.logger,
                                        kernel=self.kernel, 
                                        nu = self.nu,
                                        support=x,
                                        feature_extractor = feature_extractor, 
                                        list_for_selector = self.list_for_selector,
                                        device = self.device).double()
                optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': self.lr},
                                            {'params': model.feature_extractor.parameters(), 'lr': self.lr}])


                _ = model.train(support = np.array(x),
                                                            load_model=self.meta_trained,
                                                            checkpoint=self.checkpoint_path,
                                                            epochs=self.epochs,
                                                            optimizer=optimizer,
                                                            verbose=self.verbose, 
                                                            freeze_backbone = self.freeze_backbone)

                scores = get_scores (model, x, Lambda, y)

                pipeline_id = np.argmax(scores)

                if self.create_ensemble:
                    _ , temp_model = get_response(data, pipeline_id,  valid_pipelines,  categorical, apply_cv=False)
                    fitted_pipelines.append(temp_model)
                temp_y, _ = get_response(data, pipeline_id,  valid_pipelines,  categorical, apply_cv=self.apply_cv)
                spent_time = time.time()-start_time
                if temp_y == 0.0:
                    print("Invalid")
                x.append(pipeline_id)
                response[pipeline_id] =  temp_y

                best_val_acc = max(response).item()

                pbar.set_description("Best val. acc.: %s" % str(np.round(best_val_acc,4)))

                info_tuple = (spent_time, temp_y, np.nanmax(response.detach().cpu().numpy()), pipeline_id)
                history_list.append(info_tuple)

                if spent_time>self.time_limit:
                    break
        
        self.history_list = df = pd.DataFrame(history_list, columns = ["Time", "ValidationPerformance", "Incumbent", "PipelineID"])
        best_model_ix = int(df[df.Time<self.time_limit]["ValidationPerformance"].values.argmax())
        self.selected_pipeline = int(df.loc[best_model_ix, "PipelineID"])
        self.X_train = X_train
        self.y_train = y_train
        self.valid_pipelines = valid_pipelines
        self.categorical = categorical


        if self.create_ensemble:
            self.ensemble = self.create_ensemble_from_models(fitted_pipelines, X_val, y_val) 
            self.model = VotingClassifier(estimators=[(str(i), model) for i, model in enumerate(self.ensemble)], voting='hard')
            self.model.fit(X_train, y_train)
        else:
            self.model = self.get_model(X_train, y_train, self.selected_pipeline, valid_pipelines, categorical, apply_cv=False)
            self.model.fit(X_train, y_train)
    
    def create_ensemble_from_models(self, models, X_val, y_val):   
        ensemble =[]
        for i in range(self.ensemble_size):
            max_score = 0
            max_score_index = 0
            for j, model in enumerate(models):
                score = self.ensemble_score(ensemble+[model], X_val, y_val)
                if score>max_score:
                    max_score = score
                    max_score_index = j
            model = models[max_score_index]
            ensemble.append(model)
            models.remove(model)
        return ensemble

    def ensemble_score(self, ensemble, X, y):
        predictions = []
        for model in ensemble:
            predictions.append(model.predict(X).tolist())
        predictions = np.array(predictions).T
        predictions = [stats.mode(row)[0][0] for row in predictions]
        return balanced_accuracy_score(y, predictions)
    
    def predict(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)

    def get_model(self, X_train,  y_train, pipeline_id, pipelines_settings, categorical, apply_cv=True, n_folds=5):
        setting = pipelines_settings[pipeline_id]

        #print(setting)
        try:
            oboe_pipeline = pipeline.PipelineObject(
                p_type="classification", config=setting, 
                index=pipeline_id, verbose=True)
            columns_to_keep = np.where(~np.all(pd.isna(X_train).values, axis=0))[0]
            model = oboe_pipeline._instantiate(categorical, columns_to_keep)

        except Exception as e:
            print(e)
            
        return model
    
    def score(self, X, y):
        X,y = self.preprocess(X, y)
        return self.model.score(X, y)

    def get_pipelines_settings_on_dataset(self, settings, num_points_tr, num_features_tr):
        def p2f(x):
            return float(x.strip('%'))/100

        # pipeline settings on dataset        
        pipeline_settings_on_dataset = []
        for item in settings:
            item_copy = copy.deepcopy(item)
            if item_copy['dim_reducer']['algorithm'] == 'PCA':
                item_copy['dim_reducer']['hyperparameters']['n_components'] = int(min( num_points_tr, p2f(item_copy['dim_reducer']['hyperparameters']['n_components']) * num_features_tr))
            elif item_copy['dim_reducer']['algorithm'] == 'SelectKBest':
                item_copy['dim_reducer']['hyperparameters']['k'] = int(p2f(item_copy['dim_reducer']['hyperparameters']['k']) * num_features_tr)

            pipeline_settings_on_dataset.append(item_copy)   
        
        return pipeline_settings_on_dataset

if __name__=="__main__":

    task_id = 37
    fold_id = 0
    n_iters = 50
    time_limit = 3600
    ensemble_size = 1
    create_ensemble = True
    apply_cv = True
    task = openml.tasks.get_task(task_id)
    X_train, X_test, y_train, y_test = load_data(task, fold=fold_id)
    deep_pipe = DeepPipe(n_iters = n_iters, 
                            create_ensemble=create_ensemble, 
                        ensemble_size=ensemble_size,
                            time_limit=time_limit,
                            apply_cv=apply_cv)
    deep_pipe.fit(X_train, y_train)
    y_pred = deep_pipe.predict(X_test)
    score = deep_pipe.score(X_test, y_test)
    print("Test acc.:", score) 
