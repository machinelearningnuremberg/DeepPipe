import itertools
from sklearn.model_selection import cross_val_score
import deeppipe_api.experiments.baselines.oboe.pipeline as pipeline
import numpy as np
from math import isclose
import copy
import pandas as pd
import torch
from sklearn.preprocessing import  OneHotEncoder
from scipy.stats import norm


def torch_scaler(x):

    return (x-torch.min(x))/(torch.max(x)-torch.min(x))


def generate_settings(configs):
    """Generate pipeline annotations of the error tensor.
    Args:
        configs (dict):         A dictionary of possible pipeline configurations.
        
    Returns:
        list: List of nested dictionaries, one entry for each model setting.
              (e.g. [{'algorithm': 'KNN',  'hyperparameters': {'n_neighbors': 1, 'p': 1}},
                     {'algorithm': 'lSVM', 'hyperparameters': {'C': 1.0}}])
    """
    
    
    pipeline_steps = ['imputer', 'encoder', 'standardizer', 'dim_reducer', 'estimator']
    assert set(configs.keys()) == set(pipeline_steps), "Pipeline steps not correct!"
    
    settings_step_aggregated = {}
    for step in pipeline_steps:
        settings_step = []
        configs_single_step = configs[step]
        for alg in configs_single_step['algorithms']:
            if alg is not None:
                hyperparams = configs_single_step['hyperparameters'][alg]
                for values in itertools.product(*hyperparams.values()):
                    configs_single_alg = dict(zip(hyperparams.keys(), list(values)))
                    for key, val in configs_single_alg.items():
                        if isinstance(val, (int, float)):
                            if isclose(val, round(val)):
                                configs_single_step[key] = int(round(val))
                    settings_step.append({'algorithm': alg, 'hyperparameters': configs_single_alg})
            else:
                settings_step.append({'algorithm': None})
                
        settings_step_aggregated[step] = settings_step
    
    settings = []
    for i, values in enumerate(itertools.product(*settings_step_aggregated.values())):
        settings.append(dict(zip(settings_step_aggregated.keys(), list(values))))
    
    return settings

def load_data(task, fold=0):
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0,)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test

def get_pipelines_settings_on_dataset(settings, n_folds, num_points_tr, num_features_tr):

    def p2f(x):
        return float(x.strip('%'))/100

    # pipeline settings on dataset        
    pipeline_settings_on_dataset = []
    for item in settings:
        item_copy = copy.deepcopy(item)
        if item_copy['dim_reducer']['algorithm'] == 'PCA':
            item_copy['dim_reducer']['hyperparameters']['n_components'] = int(min((n_folds - 1) * num_points_tr/n_folds, p2f(item_copy['dim_reducer']['hyperparameters']['n_components']) * num_features_tr))
        elif item_copy['dim_reducer']['algorithm'] == 'SelectKBest':
            item_copy['dim_reducer']['hyperparameters']['k'] = int(p2f(item_copy['dim_reducer']['hyperparameters']['k']) * num_features_tr)

        pipeline_settings_on_dataset.append(item_copy)   
    
    return pipeline_settings_on_dataset

def get_response(data, pipeline_id, pipelines_settings, categorical, apply_cv=True, n_folds=5, return_model=False):
    setting = pipelines_settings[pipeline_id]

    X_train, y_train, X_test, y_test = data
    #print(setting)
    try:

        oboe_pipeline = pipeline.PipelineObject(
            p_type="classification", config=setting, 
            index=pipeline_id, verbose=True)
        columns_to_keep = np.where(~np.all(pd.isna(X_train).values, axis=0))[0]
        model = oboe_pipeline._instantiate(categorical, columns_to_keep)

        if apply_cv:
            X_train = np.vstack((X_train, X_test))
            y_train = np.hstack((y_train, y_test))
            scores = cross_val_score(model, X_train, y_train, cv=n_folds)
            response = np.mean(scores)

        else:
            return_model = True
            model.fit(X_train, y_train)
            response = model.score(X_test, y_test)

    except Exception as e:
        print(e)
        response = 0.0

    if return_model:
        return response, model  
    
    return response, None


def generate_mask(n, device="cpu"):

    """Generate masks for the first layers"""

    masks = []
    for i in range(len(n)-1):
        a = np.array(n[i])
        b = np.array(n[i+1])
        mask =  torch.zeros(b.max(), a.max(), device=device)

        for j in range(len(a)-1):
            mask[ b[j]:b[j+1], a[j]:a[j+1]]=1  
        masks.append(mask)

    return masks

def generate_concat_matrix(list_for_concatenator, n_hidden_per_stage, device="cpu"):

    """Generate matrix for concatenation"""

    n_cols = sum([n_hidden_per_stage[i]  for i, stage in enumerate(list_for_concatenator)])
    n_rows = sum([n_hidden_per_stage[i]  for i, stage in enumerate(list_for_concatenator) for j, algorithm in enumerate(stage)])
    matrix = torch.zeros(n_rows, n_cols)
    idx = 0
    idy = 0
    for i, stage in enumerate(list_for_concatenator):


        temp_matrix = torch.ones(len(stage), 1)
        d = n_hidden_per_stage[i]
        temp_matrix = torch.kron(temp_matrix, torch.eye(d))
        matrix[idx:idx+n_hidden_per_stage[i]*len(stage),idy:idy+n_hidden_per_stage[i] ] = temp_matrix
        idx+=n_hidden_per_stage[i]*len(stage)
        idy+= n_hidden_per_stage[i]

    return matrix.to(device)



def get_matrix_and_masks(hidden_dim, pipelines, list_for_concatenator, algorithm_domains, list_for_selector, n_layers_encoder, n_hidden_factor, device, omit_estimator = -1):

    n_algorithms = list_for_concatenator[-1][-1]+1
    n_stages = len(list_for_concatenator)
    #n_layers_encoder has the same semantics as n_masked_layers
    idx = 0; dim_per_algorithm = [] ; n_hidden_per_stage = []
    for i,stage in enumerate(list_for_concatenator):
        temp_dim_per_algorithm = []
        for j, algorithm in enumerate(stage):
            temp_dim_per_algorithm.append(algorithm_domains[idx+1]-algorithm_domains[idx])       
            idx+=1 
        n_hidden_per_stage.append(max(temp_dim_per_algorithm)*n_hidden_factor)
        dim_per_algorithm.append(temp_dim_per_algorithm)

    hidden_size = [n_hidden_per_stage[i] for i in range(n_stages) for j in list_for_concatenator[i]]
    n2 = [sum(hidden_size[:i]) for i in range(len(hidden_size))]
    n2.append(n2[-1]+hidden_size[-1])

    n = [algorithm_domains]
    for _ in range(n_layers_encoder):
        n.append(n2)

    hidden_dim = max(sum(n_hidden_per_stage), hidden_dim)
    #select_mask_id = torch.LongTensor(list_for_selector).to(device)
    masks = generate_mask(n, device=device)

    if n_layers_encoder>0:
        concat_matrix = generate_concat_matrix(list_for_concatenator, n_hidden_per_stage=n_hidden_per_stage, device=device)
    else:
        
        enc = OneHotEncoder(handle_unknown='ignore')

        if len(list_for_selector[0])==2: 
            print(list_for_selector)
            init_point = 0 #Specific init point for PMF Dataset
        else:
            init_point = 3 #Specific init point for TensorOboe

        additional_hps = enc.fit_transform(list_for_selector.cpu()).toarray()[:,init_point:]

        if (omit_estimator!=-1) and (omit_estimator in enc.categories_[-1].tolist()):
            additional_hps = np.delete(additional_hps, omit_estimator, 1)

        additional_hps = torch.DoubleTensor(additional_hps).to(device)
        pipelines = torch.cat((pipelines, additional_hps), axis=1)
        #concat_matrix = np.eye(pipelines.shape[1])
        concat_matrix = torch.eye(pipelines.shape[1]).double()
        

    return concat_matrix, masks, hidden_dim, n_hidden_per_stage, pipelines


def add_encoder_for_algorithms(algorithm_domains, list_for_selector, list_for_concatenator, new_encoder_input_dim):

    algorithm_domains.append(algorithm_domains[-1]+new_encoder_input_dim)
    list_for_concatenator.append([list_for_concatenator[-1][-1]+1])
    list_for_selector = list_for_selector.tolist()
    [select_item.extend([0]) for select_item in list_for_selector]

    return algorithm_domains, list_for_concatenator, list_for_selector

def regret(output,response):
    incumbent   = output[0]
    best_output = []
    for _ in output:
        incumbent = _ if _ > incumbent else incumbent
        best_output.append(incumbent)
    opt       = max(response)
    orde      = list(np.sort(np.unique(response))[::-1])
    tmp       = pd.DataFrame(best_output,columns=['regret_validation'])
    
    tmp['rank_valid']        = tmp['regret_validation'].map(lambda x : orde.index(x))
    tmp['regret_validation'] = opt - tmp['regret_validation']
    return tmp

def EI(incumbent, model_fn,support,queries,return_variance, return_score=False):
    mu, stddev     = model_fn(queries)
    mu             = mu.reshape(-1,)
    stddev         = stddev.reshape(-1,)
    if return_variance:
        stddev         = np.sqrt(stddev)
    with np.errstate(divide='warn'):
        imp = mu - incumbent
        Z = imp / stddev
        score = imp * norm.cdf(Z) + stddev * norm.pdf(Z)
    if not return_score:
        score[support] = 0
        return np.argmax(score)
    else:
        return score


   
def get_scores(model, x, Lambda, y, eval_batch=1000):
    c = len(Lambda)
    scores = []
    for i in range(eval_batch, c+eval_batch, eval_batch):

        x_qry = list(range(i-eval_batch,min(i,c)))

        predict_fn = lambda queries: model.predict(support = x, query= x_qry)
        score   =   EI(max(np.array(y.cpu())),predict_fn,support=np.where(np.array(x)<i)[0].tolist(),queries=Lambda,return_variance=False, return_score=True)
        scores += score.tolist()
    scores = np.array(scores)
    scores[x] = -1
    
    return scores