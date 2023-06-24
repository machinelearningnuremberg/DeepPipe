from tpot import TPOTClassifier
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import openml
import pandas as pd
import argparse
import os

def dummy_test():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
        iris.target.astype(np.float64), train_size=0.75, test_size=0.25, random_state=42)

    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_iris_pipeline.py')


def load_data(task, fold=0):


    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0,)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    num_points_tr, num_features_tr = X_train.shape

    return X_train, X_test, y_train, y_test, num_points_tr, num_features_tr

def dummy_test_with_conf():


    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                        train_size=0.75, test_size=0.25)

    tpot_config = {
        'sklearn.naive_bayes.GaussianNB': {
        },

        'sklearn.ensemble.AdaBoostClassifier': {
            "n_estimators": [50, 100],
            "learning_rate": [1, 1.5, 2., 2.5, 3]
        },

        'sklearn.tree.DecisionTreeClassifier' : {'min_samples_split': [2,4,8,16,32,64,128,256,512,1024,0.01, 0.001, 0.0001, 0.00001]},

        'sklearn.tree.ExtraTreeClassifier': {'min_samples_split': [2,4,8,16,32,64,128,256,512,1024,0.01, 0.001, 0.0001, 0.00001], 'criterion': ['gini', 'entropy']},

        'sklearn.ensemble.GradientBoosting': {'learning_rate': [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5], 
                                            'max_depth': [3,6], 'max_features': [None, 'log2']},

        'sklearn.linear_model.Perceptron': {},

        'sklearn.neighbors.KNeighborsClassifier': {'n_neighbors': [1,3,5,7,9,11,13,15], 'p':[1,2]},

        'sklearn.linear_model.LogisticRegression': {'C': [0.25, 0.5, 0.75,1,1.5, 2,3,4], 
                                                    'solver': ['liblinear', 'saga'],
                                                    'penalty': ['l1', 'l2']},

        'sklearn.neural_network.MLPClassifier' : {'learning_rate_init': [0.0001, 0.001, 0.01],
                                                 'learning_rate': ['adaptive'], 'solver': ['sgd', 'adam'],
                                                 'alpha': [0.0001, 0.01]},

        'sklearn.ensemble.RandomForest': {'min_samples_split':[2,4,8,16,32,64,128,256,512,1024,0.01, 0.001, 0.0001, 0.00001],
                                          'criterion': ['gini', 'entropy']},

        'sklearn.Linear.SVC': {'C': [0.125, 0.25, 0.5, 0.75, 1,2,4,8,16]}
    }
    X_train, X_test, y_train, y_test, num_points_tr, num_features_tr = load_data(task, fold=fold_id)

    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2,
                        config_dict=tpot_config, max_time_mins=1)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_digits_pipeline.py')


#dummy_test_with_conf()


tpot_config = {
    'sklearn.impute.SimpleImputer': {'strategy': ['mean', 'median', 'most_frequent', 'constant']},
    #'sklearn.preprocessing.StandardScaler': {},
    'sklearn.decomposition.PCA' : {'n_components': [0.25, 0.5, 0.75]},
    'sklearn.feature_selection.SelectKBest' : { 'k': [0.25, 0.5, 0.75]},
    'sklearn.naive_bayes.GaussianNB': {
    },

    'sklearn.ensemble.AdaBoostClassifier': {
        "n_estimators": [50, 100],
        "learning_rate": [1, 1.5, 2., 2.5, 3]
    },

    'sklearn.tree.DecisionTreeClassifier' : {'min_samples_split': [2,4,8,16,32,64,128,256,512,1024,0.01, 0.001, 0.0001, 0.00001]},

    'sklearn.tree.ExtraTreeClassifier': {'min_samples_split': [2,4,8,16,32,64,128,256,512,1024,0.01, 0.001, 0.0001, 0.00001], 'criterion': ['gini', 'entropy']},

    'sklearn.ensemble.GradientBoostingClassifier': {'learning_rate': [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5], 
                                        'max_depth': [3,6], 'max_features': [None, 'log2']},

    'sklearn.linear_model.Perceptron': {},

    'sklearn.neighbors.KNeighborsClassifier': {'n_neighbors': [1,3,5,7,9,11,13,15], 'p':[1,2]},

    'sklearn.linear_model.LogisticRegression': {'C': [0.25, 0.5, 0.75,1,1.5, 2,3,4], 
                                                'solver': ['liblinear', 'saga'],
                                                'penalty': ['l1', 'l2']},

    'sklearn.neural_network.MLPClassifier' : {'learning_rate_init': [0.0001, 0.001, 0.01],
                                                'learning_rate': ['adaptive'], 'solver': ['sgd', 'adam'],
                                                'alpha': [0.0001, 0.01]},

    'sklearn.ensemble.RandomForestClassifier': {'min_samples_split':[2,4,8,16,32,64,128,256,512,1024,0.01, 0.001, 0.0001, 0.00001],
                                        'criterion': ['gini', 'entropy']},

    'sklearn.svm.LinearSVC': {'C': [0.125, 0.25, 0.5, 0.75, 1,2,4,8,16]}
}


parser = argparse.ArgumentParser()
parser.add_argument('--task_id', default=7592, type=int)
parser.add_argument('--time_limit', default=30, type=int)
parser.add_argument('--experiment_id', default="tpot-bench", type=str)
parser.add_argument('--test_id', default="test1", type=str)

args = parser.parse_args()
print(args)

#dummy_test()
task_id = args.task_id
time_limit = args.time_limit
experiment_id = args.experiment_id
test_id = args.test_id

rootdir     = os.path.dirname(os.path.realpath(__file__))
problem_type = 'classification'


savedir     = os.path.join(rootdir,"..", "results","bench", experiment_id, test_id)

task = openml.tasks.get_task(task_id)
n_repeats, n_folds, n_samples = task.get_split_dimensions()
perf_list = []


for fold_id in range(n_folds):

    task = openml.tasks.get_task(task_id)
    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    X_train, X_test, y_train, y_test, num_points_tr, num_features_tr = load_data(task, fold=fold_id)

    tpot_config['sklearn.decomposition.PCA' ]['n_components'] = [x*num_features_tr for x in [0.25, 0.5, 0.75]],
    tpot_config[ 'sklearn.feature_selection.SelectKBest'][ 'k'] = [ x*num_features_tr for x in [0.25, 0.5, 0.75]]

    X_train = np.array(pd.get_dummies(X_train, drop_first=True))
    y_train = np.array(y_train).reshape(-1)
    #y_train = np.array(pd.get_dummies(y_train, drop_first=True)).reshape(-1)
    X_test = np.array(pd.get_dummies(X_test, drop_first=True))
    y_test = np.array(y_test).reshape(-1)

    #y_test= np.array(pd.get_dummies(y_test, drop_first=True)).reshape(-1)

    try:
        tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2,
                            config_dict=tpot_config, max_time_mins=time_limit//60, cv=5
                        )
        tpot.fit(X_train, y_train)
        #score = tpot.score(X_test, y_test)
        y_pred_test = tpot.predict(X_test)
        score = np.mean(y_pred_test==y_test)
    except Exception as e:
        score = np.nan
        print(e)
        
    perf_list.append((fold_id, score))

print(perf_list)

os.makedirs(savedir,exist_ok=True)
pd.DataFrame(perf_list).to_csv(os.path.join(savedir,str(task_id)+".csv"))
