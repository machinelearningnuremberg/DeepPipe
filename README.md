# DeepPipe: Deep Pipeline Embeddings for AutoML

*DeepPipe* efficiently optimizes Machine Learning Pipelines using meta-learning. For detailed information, refer to our [paper](https://arxiv.org/abs/2305.14009) *Deep Pipeline Embeddings for AutoML* accepted at KDD 2023. Additionally, you can visit our [blog-post](https://releaunifreiburg.github.io/deepppipe/) to have a friendly insight on how our method works.

<p align="center">
  <img src="figures/DeepPipe_architecture.png" alt="DeepPipe Architecture" width="400px">
</p>


## Installation

We present an API for optimizing pipelines in scikit-learn based on the TensorOboe search space. You can use it to search for accurate pipelines or for benchmarking your Machine Learning model on tabular data. 

```bash
conda create -n deeppipe_env python==3.9
conda activate deeppipe_env
pip install deeppipe_api==0.1.4
```

## Getting started

We present an example using an OpenML dataset. However, it works with any tabular data typed as pandas dataframe. 


```python
from deeppipe_api.deeppipe import load_data, openml, DeepPipe

task_id = 37
task = openml.tasks.get_task(task_id)
X_train, X_test, y_train, y_test = load_data(task, fold=0)
deep_pipe = DeepPipe(n_iters = 50,  #bo iterations
                    time_limit = 3600 #in seconds
                    )
deep_pipe.fit(X_train, y_train)
y_pred = deep_pipe.predict(X_test)

#Test
score = deep_pipe.score(X_test, y_test)
print("Test acc.:", score)

#print best pipeline
print(deep_pipe.model)
```
**Note**: When comparing with other AutoML optimizers have in mind that the search space might differ.

### Ensemble of Pipelines

It is possible to ensemble the best pipelines, by using a greedy approach. 


```python
from deeppipe_api.deeppipe import load_data, openml, DeepPipe

task = openml.tasks.get_task(task_id=37)
X_train, X_test, y_train, y_test = load_data(task, fold=0)
deep_pipe = DeepPipe(n_iters = 50,  #bo iterations
                    time_limit = 3600, #in seconds
                    create_ensemble = False,
                    ensemble_size = 10,
                    )
deep_pipe.fit(X_train, y_train)
y_pred = deep_pipe.predict(X_test)
score = deep_pipe.score(X_test, y_test)
print("Test acc.:", score) 
```

### Collab Notebook

You can try running DeepPipe in [this colab notebook](https://colab.research.google.com/drive/1uMJiHFn2hXwvm4KoJaOykz9lcXUKmY5A?usp=sharing).

## Advanced Usage

For meta-training *DeepPipe* or testing other search spaces, you can refer to the folder `src/deeppipe_api/experiments/`.


## Our Paper

If you use this repository/package, please cite our paper:

```
@inproceedings{pineda2023_deeppipe,
author = {Pineda Arango, Sebastian and Grabocka, Josif},
title = {Deep Pipeline Embeddings for AutoML},
year = {2023},
isbn = {9798400701030},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3580305.3599303},
doi = {10.1145/3580305.3599303}
booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {1907–1919},
numpages = {13},
location = {Long Beach, CA, USA},
series = {KDD '23}
}

```





