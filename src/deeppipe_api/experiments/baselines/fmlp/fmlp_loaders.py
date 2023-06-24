from random import sample
import numpy as np
import json
import os
from sklearn.preprocessing import OneHotEncoder as OHE
import pandas as pd


class TOboeMetaDatasetLoader:

    def __init__(self, path, test=False, fold=0):

        with open(path+"/oboe_data_v3.json") as f:
            data = json.load(f)
        folds_index = data["folds_index"]
        error_matrix = data["error"]
        init_ids = data["init_ids"]

        self.select_list = np.array(data["select_list"])
        self.pipelines = np.array(data["pipelines"]) 
        self.perf_matrix = 1-np.array(error_matrix).T.astype(np.float64)
        self.perf_matrix[self.perf_matrix>1.] = float("nan")
        #perf_matrix[np.isnan(perf_matrix)] = 0.0

        #get data
        assert fold < len(folds_index)
        self.ix_val = folds_index[fold]
        self.ix_test = folds_index[(fold+1)%5]
        self.ix_init = init_ids[(fold+1)%5]

        self.ix_train = []
        for i in range(len(folds_index)):
            if i not in [fold, (fold+1)%5]:
                for j in range(len(folds_index[i])):
                    self.ix_train.append(folds_index[i][j])

        self.datasets_OHE = OHE().fit(np.array(self.ix_train).reshape(-1,1))
        algorithms_ohe = OHE().fit_transform(self.select_list).toarray()[:,3:]
        self.pipelines = np.concatenate((self.pipelines, algorithms_ohe), axis=1)

    def get_initial_data(self, task):

        
        x_observed = []
        x_pending = []
        response = self.perf_matrix[:, int(task)]
        original_x_pending = np.where(~np.isnan(response))[0].tolist()
        response = response[original_x_pending]
        dummy_algorithms = np.zeros((len(original_x_pending), len(self.ix_train)))
        Lambdas = np.concatenate((self.pipelines[original_x_pending], dummy_algorithms), axis=1)

        x_pending = np.arange(len(response)).tolist()
        for x in  self.ix_init[task]:

            x_observed.append(original_x_pending.index(x)) #the initial id may have changed
            x_pending.remove(original_x_pending.index(x)) #x is assumed to be in x_pending

        return Lambdas, response, x_pending, x_observed

    def get_total_dim(self):

        return self.pipelines.shape[1]+ len(self.ix_train)

    def batch_sampler(self, batch_size = 100, val=False):


        pipelines_id = np.zeros((0,0))

        while pipelines_id.shape[0]==0:
            if val:
                task = np.random.choice(self.ix_val, 1)
            else:
                task = np.random.choice(self.ix_train, 1)

            pipelines_id = np.where(~np.isnan(self.perf_matrix[:, task]))[0]


        sampled_pipelines_id = np.random.choice(pipelines_id, batch_size)
        sampled_pipelines = self.pipelines[sampled_pipelines_id]

        
        if val:
            ohe_datasets = np.zeros((batch_size, len(self.ix_train)))
        else:
            ohe_datasets = np.repeat(self.datasets_OHE.transform([task]).toarray(), batch_size, axis=0 )
        
        
        self.total_min = self.pipelines.shape[0] + ohe_datasets.shape[0]
        
        return np.concatenate((sampled_pipelines, ohe_datasets), axis=1), self.perf_matrix[sampled_pipelines_id, task]




class PMFMetaDatasetLoader:

    def __init__(self, path, test = False, fold = 0):

        with open(path+"/meta-validation-dataset.json", "r") as f:
            self.valid_data = json.load(f) 
            #valid_data = {args.space:valid_data[args.space]}    

        with open(path+"/meta-train-dataset.json", "r") as f:
            self.train_data = json.load(f) 

        with open(path+"/configurations.json", "r") as f:
            self.configurations = json.load(f)

            self.configurations_dict = {}

            for search_space in self.configurations.keys(): 
                self.configurations_dict[search_space] = dict(zip(self.configurations[search_space]["ID"], self.configurations[search_space]["X"]))

        temp_data_file = os.path.join(path,  f"conf_to_space.json")
        with open(temp_data_file, "rb") as f:
            self.conf_to_space = json.load(f)   


        if test:
            temp_data_file = os.path.join(path,  f"meta_dataset_v2.json")
            with open(temp_data_file, "rb") as f:
                self.meta_dataset_v2 = json.load(f)   

            with open(path+"/meta-test-dataset.json", "r") as f:
                self.test_data = json.load(f) 

            temp_data_file = os.path.join(path, "init_ids2.json")
            with open(temp_data_file, "rb") as f:
                self.init_ids = json.load(f)

            self.test_datasets = list(pd.read_csv(path+"/ids_test.csv", header=None).iloc[:,0])

        temp_data_file = os.path.join(path,  f"order_conf.csv")
        self.conf_order = list(pd.read_csv(temp_data_file)["id"])

        self.train_datasets = list(pd.read_csv(path+"/ids_train.csv", header=None).iloc[:,0])
        self.val_datasets = list(pd.read_csv(path+"/ids_val.csv", header=None).iloc[:,0])

        self.search_spaces = list(self.train_data.keys())
        self.datasets = list(self.train_data[self.search_spaces[0]].keys())
        self.n_spaces = len(self.search_spaces)
        self.n_train_datasets = len(self.train_datasets)
        self.n_val_datasets = len(self.val_datasets)

        dims_list = []
        self.idx_in_lambda = []
        self.total_dims = 0
        for space in self.train_data.keys():
            conf_id = self.train_data[space][self.datasets[1]]["ID"][0]
            conf = self.configurations_dict[space][conf_id]
            dims_list.append(len(conf)) 
            self.idx_in_lambda.append(self.total_dims)
            self.total_dims += len(conf)
    
        self.conf_order2 = [conf  for conf in self.conf_order if self.conf_to_space[conf]!= "libsvm_svc"]

        self.datasets_OHE = OHE().fit(np.array(self.train_datasets).reshape(-1,1))
        self.spaces_OHE = OHE().fit(np.array(self.search_spaces).reshape(-1,1))

    def get_total_dim (self ):

        return self.total_dims+self.n_train_datasets+self.n_spaces


    def get_Q(self, q):
        spaces_names = [self.conf_to_space[conf] for conf in q.tolist()]
        spaces_ids = [self.search_spaces.index(space) for space in spaces_names]
        ohe_spaces = self.spaces_OHE.transform(np.array(spaces_names).reshape(-1,1)).toarray()

        sample_size = q.shape[0]
        ohe_datasets = np.zeros((sample_size, self.n_train_datasets))
        lambdas = np.zeros((sample_size, self.total_dims))

        for i in range(q.shape[0]):
            conf_id = q[i].item()
            space = spaces_names[i]

            space_id = spaces_ids[i]
            conf = self.configurations_dict[space][conf_id]

            if space_id!=(self.n_spaces-1):
                lambdas[i, self.idx_in_lambda[space_id]: self.idx_in_lambda[space_id+1]] = np.array(conf).reshape(1,-1) 
            else: 
                lambdas[i, self.idx_in_lambda[space_id]:] = np.array(conf).reshape(1,-1)
            
        Q = np.concatenate((ohe_datasets, ohe_spaces, lambdas),axis=1)

        return Q
    
    def get_initial_data(self, task ):

        LambdaIDs, response = self.get_lambdaID_response(task)

        x_pending = list(np.arange(LambdaIDs.shape[0]))

        x0 = self.init_ids[task]
        x_observed = []
        for x_ in x0:
            i = LambdaIDs.tolist().index(self.conf_order[x_])
            x_observed.append(i)
            x_pending.remove(i)

        Lambdas = self.get_Q (LambdaIDs)

        return Lambdas, response, x_pending, x_observed
    
    def get_lambdaID_response(self, task):

        LambdaIDs=  np.array(self.meta_dataset_v2[task]["ID"])
        response = np.array(self.meta_dataset_v2[task]["y"]).reshape(-1,1)

        return LambdaIDs, response


    ##batch generator
    def batch_sampler(self, batch_size, val=False):

        if val:
            data = self.valid_data
        else:
            data = self.train_data

        sample_ids = np.random.choice(self.conf_order2, batch_size).tolist()
        sampled_spaces = [self.conf_to_space[conf] for conf in sample_ids]
        sampled_spaces_ids = [self.search_spaces.index(space) for space in sampled_spaces]
        ohe_spaces = self.spaces_OHE.transform(np.array(sampled_spaces).reshape(-1,1)).toarray()

        #sampled_datasets =  np.random.choice(train_datasets, sample_size).tolist()
        #ohe_datasets = datasets_OHE.transform(np.array(sampled_datasets).reshape(-1,1)).toarray()

        ohe_datasets = np.zeros((batch_size, self.n_train_datasets))
        lambdas = np.zeros((batch_size, self.total_dims))
        response = np.zeros((batch_size,1))

        for i in range(len(sample_ids)):
            conf_id = sample_ids[i]
            space = sampled_spaces[i]
            temp_datasets = list(data[space].keys())
            temp_datasets = [d for d in temp_datasets if conf_id in data[space][d]["ID"]]
            dataset = np.random.choice(temp_datasets,1)

            if not val:

                ohe_datasets[i] = self.datasets_OHE.transform(dataset.astype(int).reshape(-1,1)).toarray()

            space_id = sampled_spaces_ids[i]
            conf = self.configurations_dict[space][conf_id]

            if space_id!=(self.n_spaces-1):
                lambdas[i, self.idx_in_lambda[space_id]: self.idx_in_lambda[space_id+1]] = np.array(conf).reshape(1,-1) 
            else: 
                lambdas[i, self.idx_in_lambda[space_id]:] = np.array(conf).reshape(1,-1)
            
            dataset = dataset.item()
            j = data[space][dataset]["ID"].index(conf_id)
            response[i] = data[space][dataset]["y"][j]

        return np.concatenate((ohe_datasets, ohe_spaces, lambdas),axis=1), response
