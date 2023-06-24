import openml
import numpy as np
import argparse
import pandas as pd
import os
from sklearn.impute import SimpleImputer
import os
import time
import sys
import json

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../baselines/baselines/")


from bench_utils import get_pipelines_settings_on_dataset, get_response, generate_settings, load_data
from baselines.bo import observe_and_suggest
from baselines.data_loader import data_loader, get_initial_configurations
from baselines.surrogates import create_surrogate

def create_BO (observe_and_suggest):

    def BO(model, data, Lambdas, valid_pipelines, valid_pipelines_id, ix_observed, bo_iter, eval_in_batches = True, return_observed = False):

        response = np.zeros((max(valid_pipelines_id)+1,1))
        ix_pending = valid_pipelines_id.copy()
        start = time.time()
        spent_time = lambda _ : time.time()-start
        
        time_list = []
        ybest_list = []
        ix_init = ix_observed.copy()
        ix_observed = []

        for original_ix in ix_init:
            try:
                ix = valid_pipelines_id.index(original_ix)
                response[ix] = get_response(data, ix, valid_pipelines, categorical)
                successful = True
            except Exception as e:
                print(e)
                successful = False
                #ytest[ix] = np.nan
            

            if successful:
                ix_observed.append(original_ix)
                ix_pending.remove(original_ix)
                ybest_list.append(max(response).item())
                time_list.append(spent_time(0))
            
                if time_list[-1] > time_limit: break

        if time_list[-1] > time_limit: return ybest_list, time_limit

        for _ in range(bo_iter):

            next_q = observe_and_suggest(model, Lambdas[ix_observed], Lambdas[ix_pending], response[ix_observed], eval_in_batches)
            original_ix = ix_pending[next_q]
            ix_observed.append(original_ix)
            ix_pending.remove(original_ix)
            response[original_ix] = get_response(data, next_q, valid_pipelines, categorical)


            best_f = max(response).item()
            ybest_list.append(best_f)
            
            #check time
            time_list.append(spent_time(0))
            if time_list[-1] > time_limit: return ybest_list, time_list

        return ybest_list, time_list

    return BO

def load_data(task, fold=0):


    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0,)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    num_points_tr, num_features_tr = X_train.shape

    return X_train, X_test, y_train, y_test, num_points_tr, num_features_tr

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', default=7592, type=int)
parser.add_argument('--time_limit', default=30, type=int)
parser.add_argument('--experiment_id', default="RFSK", type=str)
parser.add_argument('--test_id', default="test1", type=str)
parser.add_argument('--method', default="Oboe", type=str)# 'Oboe' or 'TensorOboe'
parser.add_argument('--surrogate_name', default="RFSK", type=str)# 'Oboe' or 'TensorOboe'

args = parser.parse_args()
print(args)

#dummy_test()
task_id = args.task_id
time_limit = args.time_limit
experiment_id = args.experiment_id
test_id = args.test_id
method = args.method
surrogate_name = args.surrogate_name

rootdir     = os.path.dirname(os.path.realpath(__file__))
problem_type = 'classification'


savedir     = os.path.join(rootdir,"..", "results","bench", experiment_id, test_id)

task = openml.tasks.get_task(task_id)
n_repeats, n_folds, n_samples = task.get_split_dimensions()
perf_list = []

with open(rootdir+"/../data/classification.json") as f:
    configs = json.load(f)

try:
    with open(rootdir+"/../configurations/"+experiment_id+".json") as f:
        surrogate_config = json.load(f)
except:
    surrogate_config = {}

settings = generate_settings(configs)
task = openml.tasks.get_task(task_id)
n_repeats, n_folds, n_samples = task.get_split_dimensions()

for fold_id in range(n_folds):

    task = openml.tasks.get_task(task_id)
    X_train, X_test, y_train, y_test, num_points_tr, num_features_tr = load_data(task, fold=fold_id)
    
    x_train = np.array(pd.get_dummies(X_train, drop_first=True))
    #y_train = np.array(pd.get_dummies(y_train, drop_first=True)).reshape(-1)
    y_train = np.array(y_train.cat.codes).reshape(-1)
    x_test = np.array(pd.get_dummies(X_test, drop_first=True))
    #y_test= np.array(pd.get_dummies(y_test, drop_first=True)).reshape(-1)
    y_test = np.array(y_test.cat.codes).reshape(-1)

    if np.isnan(x_train[:]).sum() > 0:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        x_train = imp_mean.fit_transform(x_train)
        x_test = imp_mean.transform(x_test)

             
            
    pipelines_settings = get_pipelines_settings_on_dataset(settings, n_folds, num_points_tr, num_features_tr)
    categorical = np.array(X_train.dtypes=="category")

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


    pipelines, perf_matrix, train_index, test_index, init_ids, select_list = data_loader("tensor-oboe", rootdir, fold=0)

    Lambda, response, ix_observed, ix_pending = get_initial_configurations(pipelines = pipelines, 
                                                                            perf_matrix = perf_matrix, 
                                                                            task = 10, #using default task 
                                                                            init_ids = init_ids, 
                                                                            select_list = select_list)


    surrogate = create_surrogate(model_name=surrogate_name, **surrogate_config)
    BO = create_BO(observe_and_suggest)
    data = (X_train, y_train, X_test, y_test)
    ybest, time_list = BO(surrogate, data, Lambda, valid_pipelines, valid_pipelines_id,  ix_observed, bo_iter=10000, eval_in_batches=True)
    results = pd.DataFrame({"response": ybest, "time": time_list})

    savedir = rootdir+"/../results/bench/"+experiment_id + "/" + test_id + "/"
    os.makedirs(savedir,exist_ok=True)

    results.to_csv(os.path.join(savedir,str(task_id)+"_"+str(fold_id)+".csv"))

