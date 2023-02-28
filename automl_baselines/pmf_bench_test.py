import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")


import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.impute
import time
import torch
import kernels
import gplvm
from utils import transform_forward, transform_backward
import bo
import json
import os 
import argparse
import time
from torch.utils.tensorboard import SummaryWriter

from utils import get_pipelines_settings_on_dataset, get_response, generate_settings, load_data
import openml

torch.set_default_tensor_type(torch.FloatTensor)

rootdir  = os.path.dirname(os.path.realpath(__file__))

path = os.path.join(rootdir,'../data') + "/"

def remove_wrong_ids(ix_test, pretrain_seed):

    if pretrain_seed == 2:
        ix_test.remove(307) #it has NAs
    elif pretrain_seed == 3:
        ix_test.remove(121) ##it has NAs
    else:
        pass

    return ix_test

def get_data(dataset="pmf-dataset", pretrain_seed_id = 0):
    """
    returns the train/test splits of the dataset as N x D matrices and the
    train/test dataset features used for warm-starting bo as D x F matrices.
    N is the number of pipelines, D is the number of datasets (in train/test),
    and F is the number of dataset features.
    """
    if dataset == "pmf-dataset":

        fn_data = path+'all_normalized_accuracy_with_pipelineID.csv'
        fn_train_ix = path+'ids_train.csv'
        fn_test_ix = path+'ids_test.csv'
        fn_data_feats = path+'data_feats_featurized.csv'
        df = pd.read_csv(fn_data)
        pipeline_ids = df['Unnamed: 0'].tolist()
        dataset_ids = df.columns.tolist()[1:]
        dataset_ids = [int(dataset_ids[i]) for i in range(len(dataset_ids))]
        Y = df.values[:,1:].astype(np.float64)

        ids_train = np.loadtxt(fn_train_ix).astype(int).tolist()
        ids_test = np.loadtxt(fn_test_ix).astype(int).tolist()

        ix_train = [dataset_ids.index(i) for i in ids_train]
        ix_test = [dataset_ids.index(i) for i in ids_test]

        Ytrain = Y[:, ix_train]
        Ytest = Y[:, ix_test]

        df = pd.read_csv(fn_data_feats)
        dataset_ids = df[df.columns[0]].tolist()

        ix_train = [dataset_ids.index(i) for i in ids_train]
        ix_test = [dataset_ids.index("10") for i in ids_test]

        Ftrain = df.values[ix_train, 1:]
        Ftest = df.values[ix_test, 1:]

    elif dataset == "oboe-dataset":
        
        with open(path+"oboe_data_v2.json") as f:
            data = json.load(f)
        folds_index = data["folds_index"]
        error_matrix = data["error"]
        init_ids = data["init_ids"]

        perf_matrix = 1-np.array(error_matrix).T.astype(np.float64)
        perf_matrix[perf_matrix>1.] = float("nan")
        #perf_matrix[np.isnan(perf_matrix)] = 0.0

        #get data
        assert pretrain_seed_id < len(folds_index)
        ix_val = folds_index[pretrain_seed_id]
        ix_test = folds_index[(pretrain_seed_id+1)%5]
        ix_init = init_ids[(pretrain_seed_id+1)%5]

        ix_train = []
        for i in range(len(folds_index)):
            if i not in [pretrain_seed_id, (pretrain_seed_id+1)%5]:
                for j in range(len(folds_index[i])):
                    ix_train.append(folds_index[i][j])

        ix_test = remove_wrong_ids(ix_test, pretrain_seed)

        Ytrain = perf_matrix[:, ix_train]
        Ytest = perf_matrix[:, ix_test]


        Ftrain = np.zeros((Ytrain.shape[0],1))
        Ftest = np.array([ix_init[str("10")] for i in ix_test])

    else:
        Ytrain, Ytest, Ftrain, Ftest = None, None, None, None

    return Ytrain, Ytest, Ftrain, Ftest

def train(m, optimizer, f_callback=None, f_stop=None):

    it = 0
    while True:

        try:
            t = time.time()

            optimizer.zero_grad()
            nll = m()
            nll.backward()
            optimizer.step()

            it += 1
            t = time.time() - t

            if f_callback is not None:
                f_callback(m, nll, it, t)

            # f_stop should not be a substantial portion of total iteration time
            if f_stop is not None and f_stop(m, nll, it, t):
                break

        except KeyboardInterrupt:
            break

    return m

def bo_search(data, m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, ytest, xi=0.0,
              do_print=False, ix_init = None,  valid_pipelines= None,  valid_pipelines_id = None, categorical = None,
              time_limit = 100):
    """
    initializes BO with L1 warm-start (using dataset features). returns a
    numpy array of length bo_n_iters holding the best performance attained
    so far per iteration (including initialization).

    bo_n_iters includes initialization iterations, i.e., after warm-start, BO
    will run for bo_n_iters - bo_n_init iterations.
    """

    preds = bo.BO(m.dim, m.kernel, bo.ei, xi,
                  variance=transform_forward(m.variance))
    
    ix_evaled = []
    #ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ix_candidates = valid_pipelines_id
    ybest_list = []
    ytest = torch.zeros((max(valid_pipelines_id)+1,1))

    if ix_init is None:
        ix_init = bo.init_l1(Ytrain, Ftrain, ftest).tolist()
    else:
        ix_init = ftest.tolist() #todo

    start = time.time()
    spent_time = lambda _ : time.time()-start
    time_list = []

    for l in range(bo_n_init):
        ix = ix_init[l]

        try:
            ix = valid_pipelines_id.index(ix)
            ytest[ix] = get_response(data, ix, valid_pipelines, categorical)
            successful = True
        except Exception as e:
            print(e)
            successful = False
            #ytest[ix] = np.nan
            
        

        if successful:
            preds.add(m.X[ix], ytest[ix])
            ix_evaled.append(ix)
            if ix in ix_candidates:
                ix_candidates.remove(ix)
        yb = preds.ybest
        if yb is None:
            yb = np.nan
        ybest_list.append(float(yb))
        
        time_list.append(spent_time(0))
        if time_list[-1] > time_limit: break

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' % (l, ytest[ix], ix, yb))

    if time_list[-1] > time_limit: return np.asarray(ybest_list), time_list

    for l in range(bo_n_init, bo_n_iters):

        ix = ix_candidates[preds.next(m.X[ix_candidates])]

        ytest[ix] = get_response(data, ix, valid_pipelines, categorical)

        preds.add(m.X[ix], ytest[ix])
        ix_evaled.append(ix)
        ix_candidates.remove(ix)
        ybest_list.append(float(preds.ybest))

        time_list.append(spent_time(0))
        if time_list[-1] > time_limit: break

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' \
                                    % (l, ytest[ix], ix, preds.ybest))

    return np.asarray(ybest_list), time_list


def descale_data(data, max_conf=None, max_datasets=None):

    Ytrain, Ytest, Ftrain, Ftest = data
    if max_conf is None: max_conf = Ytrain.shape[0]
    if max_datasets is None: max_datasets = Ytrain.shape[1]


    Ytrain = Ytrain[:max_conf, :max_datasets]
    valid_datasets = np.where(np.isnan(Ytrain).mean(axis=0)<1.0)[0]
    Ytrain = Ytrain[:, valid_datasets]


    return Ytrain, Ytest[:max_conf,:], Ftrain[valid_datasets][:max_datasets], Ftest

if __name__=='__main__':

    # train and evaluation settings


    parser = argparse.ArgumentParser()
    parser.add_argument('--max_conf', help='maximum conf. to consider', type=int, default=None) # small scale: 2000
    parser.add_argument('--max_datasets', help='maximum datasets to consider', type=int, default=None) #small scale: 100
    parser.add_argument('--n_epochs', help='maximum epoch to consider', type=int, default=300)
    parser.add_argument('--experiment_id', help='experiment id', type=str, default="PMF1.4")
    parser.add_argument('--dataset', help='dataset', type=str, default="oboe-dataset")
    parser.add_argument('--verbose', help='verbose', type=bool, default=False)
    parser.add_argument('--bo_n_iters', help='n iters for bo', type=int, default=10000)
    parser.add_argument('--latent_dim', help='latent dim', type=int, default=20)
    parser.add_argument('--xi', help='xi', type=float, default=0.01)
    parser.add_argument('--pretrain_seed', help='pretrain_seed', type=int, default=0)
    parser.add_argument('--test_id', help='test id', type=str, default="test0")
    parser.add_argument('--task_id', help='task', type=int, default=7592)
    parser.add_argument('--get_dummies', help='dummies', type=int, default=1)
    parser.add_argument('--time_limit', default=30, type=int)

    args = parser.parse_args()
    print(args)

    
    n_epochs = args.n_epochs
    max_conf = args.max_conf
    max_datasets = args.max_datasets
    #exp_name = args.experiment
    dataset = args.dataset
    verbose = args.verbose
    bo_n_iters = args.bo_n_iters
    latent_dim = args.latent_dim
    xi = args.xi
    pretrain_seed = args.pretrain_seed
    test_id = args.test_id
    task_id = args.task_id
    get_dummies = args.get_dummies
    time_limit = args.time_limit
    experiment_id = args.experiment_id
    

    if max_datasets is not None:
        Q = min(latent_dim, max_datasets)
    else:
        Q = latent_dim
    batch_size = 50 
    lr = 1e-7
    N_max = 1000
    bo_n_init = 5
    #bo_n_iters = 100
    save_checkpoint = True

    fn_checkpoint = rootdir+"/../models/pmf/"+experiment_id+"/"
    os.makedirs(fn_checkpoint,exist_ok=True)
    checkpoint_period = 500

    # train
    data = get_data(dataset, pretrain_seed)
    savedir     = os.path.join(rootdir,"..", "results","bench", experiment_id, test_id)
    #descaling data
    Ytrain, Ytest, Ftrain, Ftest = descale_data(data, max_conf, max_datasets)
    maxiter = int(Ytrain.shape[1]/batch_size*n_epochs)

    def f_stop(m, v, it, t):

        if it >= maxiter-1:
            print('maxiter (%d) reached' % maxiter)
            return True

        return False

    varn_list = []
    logpr_list = []
    t_list = []
    def f_callback(m, v, it, t):
        varn_list.append(transform_forward(m.variance).item())
        logpr_list.append(m().item()/m.D)
        if it == 1:
            t_list.append(t)
        else:
            t_list.append(t_list[-1] + t)

        if save_checkpoint and not (it % checkpoint_period):
            torch.save(m.state_dict(), fn_checkpoint + '_it%d.pt' % it)

        print('it=%d, f=%g, varn=%g, t: %g'
              % (it, logpr_list[-1], transform_forward(m.variance), t_list[-1]))
        
        #writer.add_scalar("loss/train", logpr_list[-1], it)

    device = "cuda:0" if torch.cuda.is_available() else 'cpu:0'
    device = "cpu:0"

    # create initial latent space with PCA, first imputing missing observations
    imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    X = sklearn.decomposition.PCA(Q).fit_transform(
                                            imp.fit(Ytrain).transform(Ytrain))

    # define model
    kernel = kernels.Add(kernels.RBF(Q, lengthscale=None), kernels.White(Q))
    #m = gplvm.GPLVM(Q, X, Ytrain, kernel, N_max=N_max, D_max=batch_size)
    m = gplvm.GPLVM(Q, X, Ytrain, kernel, N_max=N_max, D_max=batch_size, device= device)
    m.to_device(device)

    #m = gplvm.GPLVM(Q, X, Ytrain, kernel, N_max=N_max, D_max=batch_size).cuda()
    with open(rootdir+"/../data/classification.json") as f:
        configs = json.load(f)



    settings = generate_settings(configs)
    task = openml.tasks.get_task(task_id)
    n_repeats, n_folds, n_samples = task.get_split_dimensions()




    #ids_test = np.loadtxt(fn_test_ix).astype(int).tolist()

    # evaluate model and random baselines
    print('evaluating...')
    with torch.no_grad():

        m.load_state_dict(torch.load( fn_checkpoint + '_itFinal.pt'))   

        regrets_automl = np.zeros((bo_n_iters, n_folds))
        time_list = []

        for fold_id in np.arange(n_folds):

            print(fold_id)
            ybest = np.nanmax(Ytest[:,fold_id])

            if dataset=="oboe-dataset":
                ix_init = Ftest[10,:].tolist()
            else:
                ix_init = None

            X_train, X_test, y_train, y_test, num_points_tr, num_features_tr = load_data(task, fold=fold_id)
            

            if get_dummies:
                #X_train = pd.get_dummies(X_train, drop_first=True)
                #y_train = pd.get_dummies(y_train, drop_first=True)
                #X_test = pd.get_dummies(X_test, drop_first=True)
                #y_test= pd.get_dummies(y_test, drop_first=True)   
                X_train = pd.get_dummies(X_train, drop_first=True)
                y_train = np.array(y_train).reshape(-1)
                #y_train = np.array(pd.get_dummies(y_train, drop_first=True)).reshape(-1)
                X_test = pd.get_dummies(X_test, drop_first=True)
                y_test = np.array(y_test).reshape(-1)
             
            
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

            data = (X_train, y_train, X_test, y_test)
            response, spent_time = bo_search(data, m, bo_n_init, bo_n_iters,
                                                    Ytrain, Ftrain, Ftest[0,:],
                                                    None,
                                                    xi = xi,
                                                    do_print=verbose,
                                                    ix_init = ix_init,
                                                    valid_pipelines = valid_pipelines,
                                                    valid_pipelines_id = valid_pipelines_id,
                                                    categorical =  categorical,
                                                    time_limit=time_limit)


            #regrets_automl[:,fold_id] = response
            
            results = pd.DataFrame({"response": response, "time": spent_time})

            savedir = rootdir+"/../results/bench/"+experiment_id + "/" + test_id + "/"
            os.makedirs(savedir,exist_ok=True)

            results.to_csv(os.path.join(savedir,str(task_id)+"_"+str(fold_id)+".csv"))
        


        #with open(rootdir+"/results/test_ids_"+exp_name+".json", "w") as f:
        #    json.dump({"ids": list(ids_test)},f )
