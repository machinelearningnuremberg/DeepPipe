from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import openml
import pandas as pd
import argparse
import os
from pathlib import Path
from autoprognosis.studies.classifiers import ClassifierStudy
from autoprognosis.plugins import Plugins
from autoprognosis.utils.tester import evaluate_estimator
import time
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_estimator
from sklearn.preprocessing import LabelEncoder

def load_data(task, fold=0):


    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0,)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    num_points_tr, num_features_tr = X_train.shape

    return X_train, X_test, y_train, y_test, num_points_tr, num_features_tr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default=31, type=int)
    parser.add_argument('--time_limit', default=600, type=int)
    parser.add_argument('--experiment_id', default="autoprognosis", type=str)
    parser.add_argument('--test_id', default="test1", type=str)
    parser.add_argument('--n_bo_iters', default=3, type=int)
    parser.add_argument('--fold_id', default=-1, type=int)

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
    os.makedirs(savedir,exist_ok=True)

    task = openml.tasks.get_task(task_id)
    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    perf_list = []

    if args.fold_id == -1:
        folds = range(n_folds)
    else:
        folds = [args.fold_id]

    for fold_id in folds:

        X_train, X_test, y_train, y_test, num_points_tr, num_features_tr = load_data(task, fold=fold_id)
        X_train = pd.get_dummies(X_train, drop_first=True)
        X_test = pd.get_dummies(X_test, drop_first=True)
        start = time.time()
        workspace = Path("workspace")
        workspace.mkdir(parents=True, exist_ok=True)

        df = X_train.copy()
        df["target"] = y_train
        #X_test["target"] = y_test
        study_name = "classification_example"
        classifiers = [ "random_forest", "extra_tree_classifier", "gradient_boosting", "logistic_regression", "neural_nets", "linear_svm", "knn", "decision_trees", "adaboost", "bernoulli_naive_bayes", "gaussian_naive_bayes", "perceptron"]
        imputers = ["most_frequent", "median", "mean"]
        feature_scaling = [ "normal_transform"]
        study = ClassifierStudy(
            study_name=study_name,
            dataset=df,  # pandas DataFrame
            target="target",  # the label column in the dataset
            num_iter=args.n_bo_iters,  # how many trials to do for each candidate. Default: 50
            num_study_iter=1,  # how many outer iterations to do. Default: 5
            timeout=time_limit,  # timeout for optimization for each classfier. Default: 600 seconds
            classifiers=classifiers,
            imputers=imputers,
            feature_scaling=feature_scaling,
            workspace=workspace,
        )

        study.run()
        output = workspace / study_name / "model.p"
        model = load_model_from_file(output)
        model.fit(X_train, y_train)
        pred_prob = model.predict_proba(X_test)
        y_test_enc = LabelEncoder().fit_transform(y_test)
        score = (np.array(pred_prob).argmax(axis=1)==np.array(y_test_enc)).mean()
        total_time = time.time() - start
        perf_list.append((fold_id, score, total_time))

        if args.fold_id == -1:
            pd.DataFrame(perf_list).to_csv(os.path.join(savedir,str(task_id)+".csv"))
        else:
            pd.DataFrame(perf_list).to_csv(os.path.join(savedir,str(task_id)+"_"+str(fold_id)+".csv"))
