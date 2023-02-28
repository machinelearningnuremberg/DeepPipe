import numpy as np
import pandas as pd
import json
import openml
import sys
import os

#to do implement time out: https://stackoverflow.com/questions/492519/timeout-on-a-function-call
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")

import argparse

import torch
from modules import MaskedMLP
import numpy as np
from modules import DeepKernelGP
import time
from utils import get_pipelines_settings_on_dataset, get_response, generate_settings, load_data, get_scores
from utils import get_matrix_and_masks
"""
def get_matrix_and_masks(hidden_dim, pipelines, list_for_concatenator, algorithm_domain, list_for_selector, device ):
    
    idx = 0; dim_per_algorithm = [] ; hidden_dim_per_stage = []
    for i,stage in enumerate(list_for_concatenator):
        temp_dim_per_algorithm = []
        for j, algorithm in enumerate(stage):
            temp_dim_per_algorithm.append(algorithm_domain[idx+1]-algorithm_domain[idx])       
            idx+=1 
        hidden_dim_per_stage.append(max(temp_dim_per_algorithm)*hidden_dim_factor)
        dim_per_algorithm.append(temp_dim_per_algorithm)


    hidden_size = [hidden_dim_per_stage[i] for i in range(n_stages) for j in list_for_concatenator[i]]
    n2 = [sum(hidden_size[:i]) for i in range(len(hidden_size))]
    n2.append(n2[-1]+hidden_size[-1])

    n = [algorithm_domain]

    for _ in range(encoder_layers):
        n.append(n2)

    masks = generate_mask(n, device=device)
    concat_matrix = generate_concat_matrix(list_for_concatenator, hidden_dim_per_stage=hidden_dim_per_stage, device=device)
    hidden_dim = max(sum(hidden_dim_per_stage), hidden_dim)

    if encoder_layers>0:
        concat_matrix = generate_concat_matrix(list_for_concatenator, hidden_dim_per_stage=hidden_dim_per_stage, device=device)
    else:
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(handle_unknown='ignore')
        additional_hps = enc.fit_transform(list_for_selector.cpu().numpy()).toarray()[:,3:] #assuming that the first two dimensions
        additional_hps = torch.FloatTensor(additional_hps).to(device)
        pipelines = torch.cat((pipelines, additional_hps), axis=1).to(device)
        concat_matrix = torch.eye(pipelines.shape[1]).double()
    return concat_matrix, masks, hidden_dim, hidden_dim_per_stage, pipelines
"""
if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', default="DeepPipeTOboe", type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--freeze_backbone', default=1, type=int)
    parser.add_argument('--test_id', default="bench", type=str)
    parser.add_argument('--task_id', default=7592, type=int)
    parser.add_argument('--time_limit', help = "time limit in seconds",  default=600, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--n_iters', default=10000, type=int)
    parser.add_argument('--pretrain_seed_id', default=0, type=int)
    parser.add_argument('--get_dummies', default=1, type=int)


    args = parser.parse_args()
    print(args)

    experiment_id = args.experiment_id
    lr = args.lr
    test_id = args.test_id
    freeze_backbone = args.freeze_backbone
    task_id = args.task_id
    time_limit = args.time_limit
    epochs = args.epochs
    n_iters = args.n_iters
    pretrain_seed_id = args.pretrain_seed_id
    get_dummies = args.get_dummies

    meta_trained = 1
    verbose = 0
    

    rootdir     = os.path.dirname(os.path.realpath(__file__))

    with open(rootdir+"/data/classification.json") as f:
        configs = json.load(f)
    with open(rootdir+"/data/tensor_oboe_meta_dataset.json") as f:
        data = json.load(f)

    with open(rootdir+"/configurations/"+experiment_id+".json") as f:
        configuration = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    hidden_dim_factor = configuration["hidden_dim_factor"]
    hidden_dim = configuration["hidden_dim"]
    encoder_layers = configuration["encoder_layers"]
    aggregated_layers = configuration["aggregated_layers"]
    out_features = configuration["out_features"]
    kernel = configuration["kernel"]

    if kernel == "matern":
        nu = 2.5
    else:
        nu = None
    error_matrix = torch.FloatTensor(data["error"]).to(device)
    original_pipelines = torch.FloatTensor(data["pipelines"]).to(device)
    list_for_concatenator = data["list_for_concatenator"]
    algorithm_domain = data["algorithm_domain"]
    list_for_selector = torch.LongTensor(data["list_for_selector"]).to(device)
    error_matrix[error_matrix<0] = float("nan")
    folds_index = data["tasks_index"]
    init_ids = data["init_ids"]

    n_algorithms = list_for_concatenator[-1][-1]+1
    n_stages = len(list_for_concatenator)

    settings = generate_settings(configs)
    task = openml.tasks.get_task(task_id)
    n_repeats, n_folds, n_samples = task.get_split_dimensions()

    checkpoint_path = rootdir+"/checkpoints/"+experiment_id

    log_dir     = os.path.join(rootdir,"logs",experiment_id,str(pretrain_seed_id), test_id)
    os.makedirs(log_dir,exist_ok=True)
    logger = os.path.join(log_dir,f"{all}.txt")
    checkpoint_path = os.path.join(rootdir,"checkpoints",experiment_id)
    savedir     = os.path.join(rootdir,"results", experiment_id,str(pretrain_seed_id), test_id)

    for fold_id in range(n_folds):

        X_train, X_test, y_train, y_test, num_points_tr, num_features_tr = load_data(task, fold=fold_id)

        if get_dummies:
            X_train = pd.get_dummies(X_train, drop_first=True)
            y_train = np.array(y_train).reshape(-1)
            X_test = pd.get_dummies(X_test, drop_first=True)
            y_test = np.array(y_test).reshape(-1)

                

        pipelines_settings = get_pipelines_settings_on_dataset(settings, n_folds, num_points_tr, num_features_tr)
        categorical = np.array(X_train.dtypes=="category")
        concat_matrix, masks, hidden_dim, hidden_dim_per_stage, pipelines = get_matrix_and_masks(hidden_dim, original_pipelines, list_for_concatenator, algorithm_domain, list_for_selector, encoder_layers, hidden_dim_factor, device)

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

        Lambda = pipelines.to(device).double()[valid_pipelines_id]
        n_pipelines = Lambda.shape[0]

        original_x = init_ids[(1+pretrain_seed_id)%5][str(10)]

        x= []
        response = torch.zeros((Lambda.shape[0],1)).double().to(device)
        on_time = True
        history_list = []
        start_time = time.time()

        for pipeline_id in original_x:
            try:
                if on_time:
                    new_idx = valid_pipelines_id.index(pipeline_id)
                    data = (X_train, y_train, X_test, y_test)
                    temp_response = get_response(data, new_idx, valid_pipelines, categorical)
                    #current_time = time.time()
                    spent_time = time.time()-start_time
                    if temp_response!=0.5:
                        print(temp_response)
                        response[new_idx] = temp_response
                        x.append(new_idx)
                    info_tuple =  (spent_time, temp_response, torch.max(response).detach().cpu().numpy(), new_idx)
                    history_list.append(info_tuple)
                else:
                    break
                
                if spent_time>time_limit: 
                    on_time=False
                    break
            except Exception as e:
                print(e)
                temp_response = 0.0

        q = Lambda[x].double()
        y = response

       

        if on_time: 

            for bo_iter in range(n_iters):
                q = Lambda[x].double()
                y = response
                feature_extractor = MaskedMLP(masks, list_for_concatenator, hidden_dim=hidden_dim, 
                                                                    concat_matrix=concat_matrix, 
                                                                    hidden_dim_per_stage=hidden_dim_per_stage,
                                                                    n_non_masked_layers = aggregated_layers, 
                                                                    out_features = out_features,
                                                                    algorithm_domain = algorithm_domain,
                                                                device=device)
                model     = DeepKernelGP(X = Lambda,
                                        Y = response.reshape(-1,),
                                        log_dir=logger,
                                        kernel=kernel, 
                                        nu = nu,
                                        support=x,
                                        feature_extractor = feature_extractor, 
                                        list_for_selector = list_for_selector,
                                        device = device).double()
                optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': lr},
                                            {'params': model.feature_extractor.parameters(), 'lr': lr}])


                losses,weights,initial_weights = model.train(support = np.array(x),
                                                            load_model=meta_trained,
                                                            checkpoint=checkpoint_path,
                                                            epochs=epochs,
                                                            optimizer=optimizer,
                                                            verbose=verbose, 
                                                            freeze_backbone = freeze_backbone)

                scores = get_scores (model, x, Lambda, y)

                pipeline_id = np.argmax(scores)
                temp_y = get_response(data, pipeline_id,  valid_pipelines,  categorical)
                spent_time = time.time()-start_time
                if temp_y == 0.0:
                    print("Invalid")
                x.append(pipeline_id)
                response[pipeline_id] =  temp_y
                
                info_tuple = (spent_time, temp_y, np.nanmax(response.detach().cpu().numpy()), pipeline_id)
                history_list.append(info_tuple)

                if spent_time>time_limit:
                    break
        

        os.makedirs(savedir,exist_ok=True)
        df = pd.DataFrame(history_list, columns = ["Time", "ValidationPerformance", "Incumbent", "PipelineID"])
        
        best_model_ix = int(df[df.Time<time_limit]["ValidationPerformance"].values.argmax())
        selected_pipeline = int(df.loc[best_model_ix, "PipelineID"])
        response = get_response(data, selected_pipeline, valid_pipelines, categorical, apply_cv=False)

        df["BestPipelineTestPerformance"] = response
        df.to_csv(os.path.join(savedir,str(task_id)+"_"+str(fold_id)+".csv"))

