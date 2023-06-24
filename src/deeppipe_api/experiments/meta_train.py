import torch
import numpy as np
import os
import json
import argparse
import warnings
from modules import MaskedMLP, DeepPipe
from utils import get_matrix_and_masks, add_encoder_for_algorithms
from sklearn.preprocessing import  OneHotEncoder
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):

    experiment_id = args.experiment_id
    iterations = args.iterations
    aggregated_layers = args.aggregated_layers
    encoder_layers = args.encoder_layers
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    hidden_dim_factor = args.hidden_dim_factor
    load_from_checkpoint = args.load_from_checkpoint
    kernel = args.kernel
    out_features = args.out_features
    lr = args.lr
    pretrain_size = args.pretrain_size
    include_algorithm_encoding = args.include_algorithm_encoding
    dataset = args.dataset
    omit_estimator = args.omit_estimator
    device = args.device
    verbose = args.verbose

    pretrain_seed_id = 0
    test_batch_size = 100

    rootdir     = os.path.dirname(os.path.realpath(__file__))

    experiment_conf_dir = os.path.join(rootdir, "configurations")

    if device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(experiment_conf_dir,exist_ok=True)
    with open(experiment_conf_dir+"/"+experiment_id+".json", "w") as f:
        json.dump(args.__dict__, f)

    if dataset == "oboe":
        with open(rootdir+"/data/tensor_oboe_meta_dataset.json") as f:
            data = json.load(f)
    elif dataset == "pmf":
        with open(rootdir+"/data/pmf_meta_dataset.json") as f:
            data = json.load(f)
    elif dataset == "zap":
        with open(rootdir+"/data/zap_meta_dataset.json") as f:
            data = json.load(f) 
    else:
        raise NameError

    error_matrix = torch.DoubleTensor(data["error"]).to(device)
    pipelines = torch.DoubleTensor(data["pipelines"]).to(device)
    task_index = data["tasks_index"]
    list_for_concatenator = data["list_for_concatenator"]
    algorithm_domain = data["algorithm_domain"]
    list_for_selector = np.array(data["list_for_selector"])

    if omit_estimator!=-1:            
        selected_pipelines_ids = np.where(list_for_selector[:,-1]!=omit_estimator)[0]
        pipelines = pipelines[selected_pipelines_ids]
        error_matrix = error_matrix[:, selected_pipelines_ids]
        list_for_selector = list_for_selector[selected_pipelines_ids]

    validation_tasks = torch.LongTensor(task_index[pretrain_seed_id])

    training_tasks = []
    for i in range(len(task_index)):
        if i not in [pretrain_seed_id, (pretrain_seed_id+1)%5]:
            for j in range(len(task_index[i])):
                training_tasks.append(task_index[i][j])

    n_training_tasks = len(training_tasks)
    new_n_training_tasks = int(n_training_tasks*pretrain_size)
    training_tasks = training_tasks[:new_n_training_tasks]

    #creating feature extractor
    if include_algorithm_encoding:
        assert encoder_layers>0, "cannot include algorithm encoding when 0 encoder layers"
        enc = OneHotEncoder(handle_unknown='ignore')

        if dataset == "pmf":
            init_point = 0
        else:
            init_point = 3 # the first two columns  of the list_for_selector are constant

        additional_hps = enc.fit_transform(list_for_selector).toarray()[:,init_point:]
        additional_hps = torch.DoubleTensor(additional_hps).to(device)
        pipelines = torch.cat((pipelines, additional_hps), axis=1)
        new_encoder_input_dim = additional_hps.shape[1]
        algorithm_domain, list_for_concatenator, list_for_selector = add_encoder_for_algorithms(algorithm_domain, list_for_selector, list_for_concatenator, new_encoder_input_dim)
    
    error_matrix[error_matrix<0] = float("nan")
    list_for_selector = torch.LongTensor(list_for_selector)
    concat_matrix, masks, hidden_dim, hidden_dim_per_stage, pipelines = get_matrix_and_masks(hidden_dim, pipelines, list_for_concatenator, algorithm_domain, list_for_selector, encoder_layers, hidden_dim_factor, device )
    backbone = MaskedMLP(masks, list_for_concatenator, hidden_dim=hidden_dim,  algorithm_domain=algorithm_domain,
                                                concat_matrix=concat_matrix,
                                                hidden_dim_per_stage= hidden_dim_per_stage, 
                                                n_non_masked_layers=aggregated_layers, 
                                                out_features = out_features ,
                                                omit_estimator = omit_estimator,
                                                device=device)


    checkpoint_path = os.path.join(rootdir,"checkpoints",experiment_id)

    model = DeepPipe(training_tasks = training_tasks,
                validation_tasks = validation_tasks,
                pipelines = pipelines, 
                error_matrix = error_matrix,
                feature_extractor = backbone,
                list_for_selector = list_for_selector,
                checkpoint_path = checkpoint_path,
                batch_size = batch_size,
                test_batch_size = test_batch_size,
                kernel = kernel,
                device = device)

    if load_from_checkpoint:
        model.load_checkpoint(os.path.join(checkpoint_path,"weights"))


    optimizer = torch.optim.Adam(model.parameters(), lr= lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=1e-7)

    for epoch in range(iterations):

        model.train(epoch, optimizer, verbose=verbose)
        scheduler.step()



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', help='experiment id', type=str, default="DeepPipe")
    parser.add_argument('--iterations', help='iterations for training', type=int, default=10000)
    parser.add_argument('--aggregated_layers', help='num. layers', type=int, default=4)
    parser.add_argument('--encoder_layers', help='num. layers for encoder', type=int, default=0)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1000)
    parser.add_argument('--hidden_dim', help='hidden dim', type=int, default=24)
    parser.add_argument('--hidden_dim_factor', help='hidden factor to increase net size', type=int, default=2)
    parser.add_argument('--load_from_checkpoint', help="whehter to load", type=int, default = 0)
    parser.add_argument('--kernel', help='kernel type', type=str, default="matern")
    parser.add_argument('--out_features', help='output features for backbone/ input to kernel', type=int, default=20)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
    parser.add_argument('--pretrain_size', help='percentage of metadata to use', type=float, default=1.0)
    parser.add_argument('--include_algorithm_encoding',help='whether to include OHE of the active algoritm' ,type=int, default=0)
    parser.add_argument('--dataset', help='dataset of evaluations to use (a.k.a metadataset)', type=str, default="oboe")
    parser.add_argument('--omit_estimator', help='id of estimator to omit during meta-training', default=-1, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--verbose', default=0, type=int)


    args = parser.parse_args()
    print(args)

    main(args)


    