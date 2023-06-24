import numpy as np
import argparse
import os
import json
import torch
from modules import MaskedMLP, DeepKernelGP
from utils import get_matrix_and_masks, add_encoder_for_algorithms, regret, get_scores
from sklearn.preprocessing import  OneHotEncoder


def main (args):


    n_iters = args.n_iters
    experiment_id = args.experiment_id
    aggregated_layers = args.aggregated_layers
    encoder_layers = args.encoder_layers
    task = args.task
    test_id = args.test_id
    hidden_dim = args.hidden_dim
    pretrain_seed_id = args.pretrain_seed_id
    lr = args.lr
    epochs = args.epochs
    kernel = args.kernel
    nu = args.nu
    meta_trained = args.meta_trained
    hidden_dim_factor = args.hidden_dim_factor
    freeze_backbone = args.freeze_backbone
    out_features = args.out_features
    include_algorithm_encoding = args.include_algorithm_encoding
    dataset = args.dataset
    omit_estimator = args.omit_estimator
    n_init_omit_estimator = args.n_init_omit_estimator
    omit_estimator_test = args.omit_estimator_test
    verbose = args.verbose


    rootdir     = os.path.dirname(os.path.realpath(__file__))
    savedir     = os.path.join(rootdir,"results",experiment_id,str(pretrain_seed_id), test_id)

    os.makedirs(savedir,exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    list_for_concatenator = data["list_for_concatenator"]
    list_for_selector = np.array(data["list_for_selector"])
    algorithm_domain = data["algorithm_domain"]
    error_matrix[error_matrix<0] = float("nan")
    task_index = data["tasks_index"]
    init_ids = data["init_ids"]


    if omit_estimator_test!=-1:
            
        selected_pipelines_ids = np.where(list_for_selector[:,-1]!=omit_estimator_test)[0]
        pipelines = pipelines[selected_pipelines_ids]
        error_matrix = error_matrix[:, selected_pipelines_ids]
        list_for_selector = list_for_selector[selected_pipelines_ids]

    assert pretrain_seed_id < len(task_index)

    if task == "all":
        test_tasks = [int(x) for x in task_index[(pretrain_seed_id+1)%5]]
   
    else:
        test_tasks = [int(task)]


    #creating feature extractor
    if include_algorithm_encoding:
        assert encoder_layers>0, "cannot include algorithm encoding when 0 encoder layers"
        enc = OneHotEncoder(handle_unknown='ignore')
        
        if dataset == "pmf":
            init_point = 0
        else:
            init_point = 3

        additional_hps = enc.fit_transform(list_for_selector).toarray()[:,init_point:]
        additional_hps = torch.DoubleTensor(additional_hps).to(device)
        pipelines = torch.cat((pipelines, additional_hps), axis=1)
        new_encoder_input_dim = additional_hps.shape[1]
        algorithm_domain, list_for_concatenator, list_for_selector = add_encoder_for_algorithms(algorithm_domain, list_for_selector, list_for_concatenator, new_encoder_input_dim)



    list_for_selector = torch.LongTensor(list_for_selector).to(device)
    concat_matrix, masks, hidden_dim, hidden_dim_per_stage, pipelines = get_matrix_and_masks(hidden_dim, pipelines, list_for_concatenator, algorithm_domain, list_for_selector, encoder_layers, hidden_dim_factor, device , omit_estimator=omit_estimator)


    for task in test_tasks:

        log_dir     = os.path.join(rootdir,"logs",experiment_id,str(pretrain_seed_id), test_id)
        os.makedirs(log_dir,exist_ok=True)
        logger = os.path.join(log_dir,f"{task}.txt")
        checkpoint_path = os.path.join(rootdir,"checkpoints",experiment_id)

        evaluated_pipelines = torch.where(~torch.isnan(error_matrix[task,:]))[0] 
        Lambda = pipelines[evaluated_pipelines].to(device)
        response = 1- error_matrix[task, evaluated_pipelines].to(device)

        original_x = init_ids[(1+pretrain_seed_id)%5][str(task)]

        x = []
        n_append_at_beginning = 0

        for x_i in original_x:

            try:
                x.append(torch.where(evaluated_pipelines==x_i)[0].item())
            except:
                n_append_at_beginning+=1

        y = response[x]

        if omit_estimator!=-1:
            omit_estimator_rs = np.random.RandomState(123)
            x_estimator = torch.where(list_for_selector[evaluated_pipelines,-1]==omit_estimator)[0].cpu().numpy()
            
            if len(x_estimator)>0:
                new_x = omit_estimator_rs.choice(x_estimator,n_init_omit_estimator)
                x += new_x.tolist()
                n_iters-=n_init_omit_estimator


        for _ in range(n_iters):
            if max(response) in y:
                break        
            done = False

        
            try:      
                feature_extractor = MaskedMLP(masks, list_for_concatenator, hidden_dim=hidden_dim, algorithm_domain=algorithm_domain,
                                                                    concat_matrix=concat_matrix, 
                                                                    hidden_dim_per_stage=hidden_dim_per_stage,
                                                                    n_non_masked_layers = aggregated_layers, 
                                                                    out_features = out_features,
                                                                    omit_estimator = omit_estimator,
                                                                    device=device)

                model     = DeepKernelGP(X = Lambda,
                                            Y = response.reshape(-1,),
                                            log_dir=logger,
                                            kernel=kernel, 
                                            nu = nu,
                                            support=x,
                                            feature_extractor = feature_extractor.double(), 
                                            list_for_selector = list_for_selector,
                                            device = device)
        
                optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': lr},
                                                {'params': model.feature_extractor.parameters(), 'lr': lr}])

                model.train(support = np.array(x),  load_model=meta_trained,
                                                            checkpoint=checkpoint_path,
                                                            epochs=epochs,
                                                            optimizer=optimizer,
                                                            verbose=verbose, 
                                                            freeze_backbone = freeze_backbone)


                scores = get_scores (model, x, Lambda, y)
                candidate = np.argmax(scores)
                x.append(candidate)
                y = response[x]
                
                done=True
            except Exception as e:
                print(e)

        if not done:
            break
        
        if done:
            y = np.array(y.cpu()).tolist()
            y = [y[0] for _ in range(n_append_at_beginning)] + y #to match the expected size
            x = [x[0] for _ in range(n_append_at_beginning)] + x

            results            = regret(np.array(y),response.cpu())
            results['indices'] = np.asarray(x).reshape(-1,)
            
            if omit_estimator!=-1:
                results['estimator_count'] = np.cumsum(list_for_selector[x,-1].cpu().numpy()==omit_estimator)
            results.to_csv(os.path.join(savedir,f"{task}.csv"))


if __name__ == "__main__":

        
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
    parser.add_argument('--task', help='Selected Task ID', type=str, default="all")
    parser.add_argument('--experiment_id', help='experiment_id', type=str, default="DeepPipe")
    parser.add_argument('--meta_trained', help="meta-trained", type=int, default=1)
    parser.add_argument('--aggregated_layers', help='iterations', type=int, default=4)
    parser.add_argument('--encoder_layers', help='iterations', type=int, default=0)
    parser.add_argument('--test_id', help='test id to identify a test experiment given an experiment id', type=str, default="test1")
    parser.add_argument('--hidden_dim', help='hidden_dim', type=int, default=24)
    parser.add_argument('--epochs', help='epochs for updating the DeepKernel', type=int, default=100)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--pretrain_seed_id', help='seed', type=int, default=0)
    parser.add_argument('--kernel', help='kernel', type=str, default="matern")
    parser.add_argument('--nu', help='nu parameter for kernel', type=float, default=2.5)
    parser.add_argument('--hidden_dim_factor', help='hidden factor to increase net size', type=int, default=8)
    parser.add_argument('--freeze_backbone', help='freeze backbone', type=int, default=0)
    parser.add_argument('--out_features', help='', type=int, default=20)
    parser.add_argument('--include_algorithm_encoding', type=int, default=0)
    parser.add_argument('--dataset', help='dataset of evaluations (a.k.a. meta-dataset). Options: oboe, pmf, zap', type=str, default="oboe")
    parser.add_argument('--omit_estimator', help='id of estimator to omit on meta-train', default=-1, type=int)
    parser.add_argument('--omit_estimator_test', help='id of estimator to omit on meta-test', default=-1, type=int)
    parser.add_argument('--n_init_omit_estimator', help='number of pipelines to see at the beginning from the omitted estimator', default=1, type=int)
    parser.add_argument('--verbose', default=0, type=int)


    args = parser.parse_args()
    print(args)

    main(args)