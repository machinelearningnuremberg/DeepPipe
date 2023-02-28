
from oboe import AutoLearner, error  # This may take around 15 seconds at first run.
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import openml
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer


def load_data(task, fold=0):


    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0,)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    num_points_tr, num_features_tr = X_train.shape

    return X_train, X_test, y_train, y_test, num_points_tr, num_features_tr


def dummy_test():
    data = load_iris()
    x = np.array(data['data'])
    y = np.array(data['target'])
    print(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    m = AutoLearner(p_type=problem_type, runtime_limit=30, method=method, verbose=False, ensemble_max_size=1)
    m.fit(x_train, y_train)
    y_predicted = m.predict(x_test)

    print("prediction error (balanced error rate): {}".format(error(y_test, y_predicted, 'classification')))    
    print("selected models: {}".format(m.get_models()))

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', default=7592, type=int)
parser.add_argument('--time_limit', default=30, type=int)
parser.add_argument('--experiment_id', default="oboe-bench", type=str)
parser.add_argument('--test_id', default="test1", type=str)
parser.add_argument('--method', default="Oboe", type=str)# 'Oboe' or 'TensorOboe'

args = parser.parse_args()
print(args)

#dummy_test()
task_id = args.task_id
time_limit = args.time_limit
experiment_id = args.experiment_id
test_id = args.test_id
method = args.method

rootdir     = os.path.dirname(os.path.realpath(__file__))
problem_type = 'classification'


savedir     = os.path.join(rootdir,"..", "results","bench", experiment_id, test_id)

task = openml.tasks.get_task(task_id)
n_repeats, n_folds, n_samples = task.get_split_dimensions()
perf_list = []

for fold_id in range(n_folds):

    task = openml.tasks.get_task(task_id)
    X_train, X_test, y_train, y_test, num_points_tr, num_features_tr = load_data(task, fold=fold_id)
    m = AutoLearner(p_type=problem_type, runtime_limit=time_limit, method=method, verbose=False, ensemble_max_size=1)
    
    x_train = np.array(pd.get_dummies(X_train, drop_first=True))
    #y_train = np.array(pd.get_dummies(y_train, drop_first=True)).reshape(-1)
    y_train = np.array(y_train.cat.codes).reshape(-1)
    x_test = np.array(pd.get_dummies(X_test, drop_first=True))
    #y_test= np.array(pd.get_dummies(y_test, drop_first=True)).reshape(-1)
    y_test = np.array(y_test.cat.codes).reshape(-1)

    if np.isnan(x_train.values[:]).sum() > 0:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        x_train = imp_mean.fit_transform(x_train)
        x_test = imp_mean.transform(x_test)

    try: 
        m.fit(x_train, y_train)
        y_predicted = m.predict(x_test)
        e = error(y_test, y_predicted, 'classification')
        perf_list.append((fold_id, 1-e))
        print(e)
    except Exception as e:
        perf_list.append((fold_id, np.nan))
        print(e)

print(perf_list)

os.makedirs(savedir,exist_ok=True)
pd.DataFrame(perf_list).to_csv(os.path.join(savedir,str(task_id)+".csv"))

