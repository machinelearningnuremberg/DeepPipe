import numpy as np
import pandas as pd
import sys
import re
import os
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
import scipy.stats as st
import multiprocessing as mp
from scipy.optimize import minimize
import json
import linalg
import convex_opt
import argparse
import tensorly as tl
import scipy as sp
from util import extract_columns, check_dataframes, tucker_on_error_tensor, generate_settings, error


parser = argparse.ArgumentParser()
parser.add_argument('--task', help="Task", type=str, default=None)
parser.add_argument('--final_rank', help="final rank, number of trials", type=int, default=200)
parser.add_argument('--type', help="Computation type", type=str, default="ED")
parser.add_argument('--dataset_name', help="meta dataset", type=str, default="pmf")
parser.add_argument('--fold', help="fold", type=int, default="0")
parser.add_argument('--experiment_id', help="experiment id", type=str, default="01")
parser.add_argument('--impute_value', help="impute value", type=float, default=1)
parser.add_argument('--threshold', help="impute value", type=float, default=0.03) #or 0.04
parser.add_argument('--k', help="k datasets for factorization", type=int, default=150)


args = parser.parse_args()
print(args)

task = args.task
final_rank = args.final_rank
computation_type = args.type
dataset_name = args.dataset_name
fold = args.fold
experiment_id = args.experiment_id
threshold = args.threshold
impute_value = args.impute_value
k = args.k
verbose = False

assert task != None, "Specify a task"

def indloc(indices, ind):
    return np.where(np.array(indices)==ind)[0][0]

def number_of_entries_solve(N, Y, scalarization='D'):
    n = Y.shape[1]
    # It is observed the scipy.optimize solver in this problem usually converges within 50 iterations. Thus a maximum of 50 step is set as limit.
    if scalarization == 'D':
        def objective(v):
            sign, log_det = np.linalg.slogdet(Y @ np.diag(v) @ Y.T)
            return -1 * sign * log_det
    elif scalarization == 'A':
        def objective(v):
            return np.trace(np.linalg.pinv(Y @ np.diag(v) @ Y.T))
    elif scalarization == 'E':
        def objective(v):
            return np.linalg.norm(np.linalg.pinv(Y @ np.diag(v) @ Y.T), ord=2)
    def constraint(v):
        return N - np.sum(v)
    v0 = np.full((n, ), 0.5)
    constraints = {'type': 'ineq', 'fun': constraint}
    v_opt = minimize(objective, v0, method='SLSQP', bounds=[(0, 1)] * n, options={'maxiter': 50},
                     constraints=constraints)
    return v_opt.x



loo = LeaveOneOut()

rootdir     = os.path.dirname(os.path.realpath(__file__))
path_data = os.path.join(rootdir, "..", "data")
path_results = os.path.join(rootdir, "..","results", "oboe", computation_type+experiment_id)

os.makedirs(path_results,exist_ok=True)


if computation_type == "tucker":

    error_tensor = np.float64(np.load(os.path.join(path_data, 'error_tensor_imputed_20-4-2-2-8-20_f16_compressed.npz'))['a'])    

    rank_for_imputation = (20, 4, 2, 2, 8, 20)
    #_, _, error_tensor_imputed, _ = tucker_on_error_tensor(error_tensor, rank_for_imputation, save_results=False, verbose=verbose)

    k_dataset_for_factorization = k
    k_estimator_for_factorization = 30
    core_tr, factors_tr = tl.decomposition.tucker(error_tensor, rank=(k_dataset_for_factorization, 4, 2, 2, 8, k_estimator_for_factorization))
    pipeline_latent_factors = tl.unfold(tl.tenalg.multi_mode_dot(core_tr, factors_tr[1:], modes=[1, 2, 3, 4, 5]), mode=0)
    U_t, S_t, Vt_t = sp.linalg.svd(pipeline_latent_factors, full_matrices=False)   
    errmtx_train = Vt_t

    if dataset_name == "pmf":
        with open(path_data+"/pmf_data_v1.json") as f:
                data = json.load(f)
    else:
        with open(path_data+"/oboe_data_v2.json") as f:
                data = json.load(f)

    errmtx_common =  np.array(data["error"])
    folds_index = data["folds_index"]   
    init_ids = data["init_ids"]
    folds = list(range(len(folds_index)))
    folds.remove((1+fold)%5)
    folds.remove(fold)

    train_index = np.array([x for f in folds for x in folds_index[f]])
    test_index = int(task)
    ix_init = init_ids[(1+fold)%5][str(task)]



else:

    if dataset_name == "oboe":
        errmtx_df = pd.read_csv(os.path.join( path_data+'/error_matrix.csv'), index_col=0, header=0)
        errmtx = errmtx_df.values
        ind_errmtx = errmtx_df.index.astype(int)


        ind_metafeatures = pd.read_csv(path_data+'/metafeatures_oboe.csv', index_col=0, header=0).index
        ind_common = list(set(ind_errmtx).intersection(set(ind_metafeatures)))
        errmtx_common_df = errmtx_df.loc[ind_common]
        errmtx_common = errmtx_common_df.values


        #they seem not tu substract the test column for the pivoting!
    elif dataset_name == "pmf":

        metafeatures_df = pd.read_csv(path_data+"/metafeatures_oboe.csv", index_col=0, header=0)
        initialization_ids = os.path.join( path_data+"/init_ids2.json")
        with open(initialization_ids, "rb") as f:
            init_ids = json.load(f)

        train_ids = list(pd.read_csv(path_data+"/ids_train.csv").iloc[:,0])
        test_ids = list(pd.read_csv(path_data+"/ids_test.csv").iloc[:,0])

        errmtx_df = pd.read_csv(os.path.join( path_data+'/all_normalized_accuracy_with_pipelineID.csv'), index_col=0, header=0)
        errmtx_common = errmtx_df.values.T

        errmtx_common[np.isnan(errmtx_common)] = 0
        errmtx_common = 1-errmtx_common
        columns = list(errmtx_df.columns)

        train_index = [columns.index(str(x)) for x in train_ids]
        columns_train = [x for x in columns if int(x) in train_ids]
        errmtx_train = errmtx_common[train_index,:]

        ix_init = init_ids[str(task)]
        test_index = columns.index(task)

        print(errmtx_common.shape)

    elif dataset_name == "tensor-oboe":

        with open(rootdir+"/../data/oboe_data_v2.json") as f:
            data = json.load(f)

        errmtx_common =  np.array(data["error"])
        #errmtx_common[np.isnan(errmtx_common)] = 0
        #errmtx_common = 1-errmtx_common

        folds_index = data["folds_index"]   
        init_ids = data["init_ids"]
        folds = list(range(5))
        folds.remove((1+fold)%5)
        folds.remove(fold)

        train_index = np.array([x for f in folds for x in folds_index[f]])
        test_index = int(task)
        ix_init = init_ids[(1+fold)%5][str(task)]

        errmtx_train = errmtx_common[train_index,:]
        errmtx_train[np.isnan(errmtx_train)] = impute_value
        errmtx_train[errmtx_train <0.0] = impute_value


    else:
        raise NameError("Dataset name not implemented")

errmtx_pred = np.zeros(errmtx_common.shape)
errmtx_pred2 = np.zeros(errmtx_common.shape)

#X_pca, Y_pca, _ = linalg.pca(errmtx_common, threshold=threshold)


n_init = 5
initial_rank = 5
n_pipelines = errmtx_common.shape[1]

regret_all_with_mf = pd.DataFrame(columns=['rank {}'.format(rank) for rank in range(initial_rank, final_rank+1)])
errors_encountered_all = []


regret_all = pd.DataFrame(columns=['rank {}'.format(rank) for rank in range(initial_rank, final_rank+1)])
errors_encountered_all = []


regret = []
regret_per_epoch = []
regret_per_epoch2 = []

new_row = np.zeros((1, errmtx_common.shape[1]))
new_row2 = np.zeros((1, errmtx_common.shape[1]))

response = errmtx_common[test_index,:]
invalid_pipelines = (errmtx_common[test_index,:]<0.0) | (np.isnan(errmtx_common[test_index,:]))

errmtx_common[test_index,response <0.0] = np.nan
invalid_response = np.where(np.isnan(response))[0]

# true best
y_best_true = np.nanmin(errmtx_common[test_index, :]).item()        

# predicted best
y_best_pred = np.nanmin(errmtx_common[test_index, ix_init]).item()

regret = [np.nanmin(errmtx_common[test_index, ix_init[0:i+1]]).item() - y_best_true for i in range(len(ix_init)) ]
regret_per_epoch = regret.copy()
regret_per_epoch2 = regret.copy()

x_pending = set(np.arange(errmtx_common.shape[1]))
x_observed = set(ix_init.copy())
x_pending = x_pending.difference(x_observed)
x_pending = x_pending.difference(set(invalid_response))


for rank in range(initial_rank, final_rank+1):   

    #select the top rank columns using QR decoposition

    #if computation_type == "ED":
    #    v_opt = number_of_entries_solve(rank-n_init, Y_pca, scalarization="D")
    #    to_sample = np.argsort(-v_opt)[:(rank-n_init)]
    #else:
    try:
        to_sample = list(linalg.pivot_columns(errmtx_train[:,~invalid_pipelines], rank=rank-n_init))
        to_sample_real_index = np.arange(n_pipelines)[~invalid_pipelines][to_sample].tolist()+ix_init
        
        ix= linalg.pivot_columns(errmtx_train[:,list(x_pending)], rank=1)
        x_observed =  x_observed.union(set([list(x_pending)[ix.item()]]))
        x_pending = x_pending.difference(x_observed)

        new_row[:, to_sample] = errmtx_common[test_index, to_sample]
        new_row2[:, list(x_observed)] = errmtx_common[test_index, list(x_observed)]

        #complete the matrix
        errmtx_pred[test_index, :] = linalg.impute(errmtx_train, new_row, to_sample_real_index, rank=rank)
        errmtx_pred2[test_index, :] = linalg.impute(errmtx_train, new_row2, list(x_observed), rank=rank)
        
        errmtx_pred[test_index, invalid_pipelines] = 1.
        errmtx_pred2[test_index, invalid_pipelines] = 1.

        # predicted best
        y_best_pred = min(y_best_pred, min(errmtx_common[test_index, to_sample_real_index]), errmtx_common[test_index, np.argmin(errmtx_pred[test_index, :])])

        # this is the "actual" best pred, because in a real scenario this is not performed sequentially, therefore we should not keep track of the y_best_pred
        y_best_pred_per_epoch = min(min(errmtx_common[test_index, to_sample_real_index]),errmtx_common[test_index, np.argmin(errmtx_pred[test_index, :])])
        y_best_pred_per_epoch2 = min(min(errmtx_common[test_index, list(x_observed)]),errmtx_common[test_index, np.argmin(errmtx_pred2[test_index, :])])
    except Exception as e:

        print(e)
        

    # collect regret
    regret.append(y_best_pred - y_best_true)
    regret_per_epoch.append(y_best_pred_per_epoch - y_best_true)
    regret_per_epoch2.append(min(y_best_pred_per_epoch2 - y_best_true, min(regret_per_epoch2)))

    print("rank: ", rank, " ", regret[-1], " and ", regret_per_epoch[-1], " and ", regret_per_epoch2[-1])

with open(path_results+"/"+task+".json", "w") as f:

    json.dump({"regret1": regret,
                "regret2": regret_per_epoch,
                "regret3": regret_per_epoch2}, f)

print("End")