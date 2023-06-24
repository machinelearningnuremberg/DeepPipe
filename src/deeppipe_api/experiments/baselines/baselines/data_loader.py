import json
from logging import raiseExceptions
import numpy as np
from sklearn.preprocessing import OneHotEncoder as OHE
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def data_loader (dataset_name, rootdir, **args):

    if dataset_name=="tensor-oboe":

        with open(rootdir+"/../data/oboe_data_v3.json") as f:
            data = json.load(f)

    elif dataset_name=="pmf": 

        with open(rootdir+"/../data/pmf_data_v1.json") as f:
            data = json.load(f)
    
    else:
        raise ValueError

    fold = args.get("fold",0)
    task = args.get("task", None)


    pipelines =  np.array(data["pipelines"])
    folds_index = data["folds_index"]   
    init_ids = data["init_ids"][(1+fold)%5]
    perf_matrix = 1-np.array(data["error"])
    perf_matrix[perf_matrix<0.0] = np.nan

    select_list = np.array(data["select_list"])
    folds = list(range(len(folds_index)))
    folds.remove((1+fold)%5)
    folds.remove(fold)

    vt = VarianceThreshold(threshold=1e-5)
    select_list = vt.fit_transform(select_list)
    algorithms_ohe = OHE().fit_transform(select_list).toarray()

    train_index = np.array([x for f in folds for x in folds_index[f]])
    pipelines = np.concatenate((pipelines, algorithms_ohe), axis=1)


    if task is None:
        test_index = folds_index[(fold+1)%5]
    else:
        task = [int (task)]

    return pipelines, perf_matrix, train_index, test_index, init_ids, select_list



def get_initial_configurations(pipelines, perf_matrix, task, init_ids, select_list, omit_estimator = -1, n_init_omit_estimator = 1):

    random_state =np.random.RandomState(seed=123)

    evaluated_pipelines = np.where(~np.isnan(perf_matrix[task,:]))[0] 
    Lambda = pipelines[evaluated_pipelines]
    response = perf_matrix[task, evaluated_pipelines]

    original_x = init_ids[str(task)]

    ix_pending = list(range(evaluated_pipelines.shape[0]))

    #ix_observed = [np.where(evaluated_pipelines==x_i)[0].item() for x_i in original_x]
    ix_observed = []
    for x_i in original_x:
        try:
            ix = np.where(evaluated_pipelines==x_i)[0].item()
            ix_observed.append(ix)
            ix_pending.remove(ix)
        except:
            print("Not found init-id")
            #ix_observed.append(-1)
            #ix = random_state.choice(ix_pending,1).item()
            #ix_observed.append(ix)
            #ix_pending.remove(ix)     
            #ix_observed.append(np.nan)  

    if omit_estimator!=-1:
        omit_estimator_rs = np.random.RandomState(123)
        x_estimator = np.where(select_list[evaluated_pipelines,-1]==omit_estimator)[0]
        
        if len(x_estimator)>0:
            new_x = omit_estimator_rs.choice(x_estimator,n_init_omit_estimator)
            ix_observed += new_x.tolist()
            [ix_pending.remove(x) for x in new_x]

    return Lambda, response, ix_observed, ix_pending
