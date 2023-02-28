from logging import raiseExceptions
import numpy as np
import pandas as pd
import os
import json
import argparse
from bo import observe_and_suggest, create_BO, random_suggest
from data_loader import data_loader, get_initial_configurations
from surrogates import create_surrogate
from sklearn.ensemble import RandomForestRegressor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--bo_iter', help="final rank, number of trials", type=int, default=101)
    parser.add_argument('--dataset_name', help="meta dataset", type=str, default="tensor-oboe")
    parser.add_argument('--fold', help="fold", type=int, default=0)
    parser.add_argument('--experiment_id', help="experiment id", type=str, default="01")
    parser.add_argument('--surrogate_name', help="surrogate name", type=str, default="GP")
    parser.add_argument('--task', help="task", type=int, default=None)
    parser.add_argument('--n_init_omit_estimator', default=1, type=int)
    parser.add_argument('--omit_estimator', default=-1, type=int)

    args = parser.parse_args()
    print(args)

    bo_iter = args.bo_iter
    dataset_name = args.dataset_name
    fold = args.fold
    experiment_id = args.experiment_id
    surrogate_name = args.surrogate_name
    task = args.task
    n_init_omit_estimator = args.n_init_omit_estimator
    omit_estimator = args.omit_estimator

    rootdir     = os.path.dirname(os.path.realpath(__file__))
    path_data = os.path.join(rootdir, "..", "data")
    path_results = os.path.join(rootdir, "..","results", surrogate_name, experiment_id)
    os.makedirs(path_results, exist_ok=True)


    experiment_conf_dir = os.path.join(rootdir, "..", "configurations")
    os.makedirs(experiment_conf_dir,exist_ok=True)
    with open(experiment_conf_dir+"/"+experiment_id+"_base.json", "w") as f:
        json.dump(args.__dict__, f)


    try:
        with open(rootdir+"/../configurations/"+experiment_id+".json") as f:
            config = json.load(f)
    except:
        config = {}

    pipelines, perf_matrix, train_index, test_index, init_ids, select_list = data_loader(dataset_name, rootdir, fold=fold)


    #random = np.random.RandomState(301)
    #randomInitializer = np.random.RandomState(0)

    if task is None:
        task_list = test_index
    else:
        task_list = [task]

    for task in task_list:

        try:

            Lambda, response, ix_observed, ix_pending = get_initial_configurations(pipelines = pipelines, 
                                                                                    perf_matrix = perf_matrix, 
                                                                                    task = task, 
                                                                                    init_ids = init_ids, 
                                                                                    select_list = select_list,
                                                                                    omit_estimator = omit_estimator,
                                                                                    n_init_omit_estimator = n_init_omit_estimator)

            surrogate = create_surrogate(model_name=surrogate_name, **config)

            if surrogate_name=="random":
                BO = create_BO(random_suggest)
            else:
                BO = create_BO(observe_and_suggest)

            if omit_estimator == -1:

                regret = BO(surrogate, Lambda, response, ix_observed, ix_pending, bo_iter=bo_iter, eval_in_batches=True)
                temp_path_results = path_results+"/"+str(task)+".json"
                with open(temp_path_results, "w") as f:
                    json.dump(regret, f)
                
            else:

                regret, ix_observed = BO(surrogate, Lambda, response, ix_observed, ix_pending, bo_iter=bo_iter, eval_in_batches=True, return_observed = True)
                
                temp_path_results = path_results+"/"+str(task)+".json"
                with open(temp_path_results, "w") as f:
                    json.dump(regret, f)

                estimator_count = np.cumsum(select_list[ix_observed,-1]==omit_estimator).tolist()      
                
                temp_path_results = path_results+"/ec_"+str(task)+".json"
                with open(temp_path_results, "w") as f:
                    json.dump(estimator_count, f)
          
        
        except Exception as e:

            print(e)





