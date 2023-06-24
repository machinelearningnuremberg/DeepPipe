# DeepPipe

This is the repository of the [paper](https://arxiv.org/abs/2305.14009) *Deep Pipeline Embeddings for AutoML* accepted at KDD 2023.

Download the repo and install requirements to run the experiment scripts:

- openml 0.12.2
- torch 1.8.1
- gpytorch 1.4.2
- sklearn 0.24.2
- pandas 1.2.4
- scipy 1.7.1
- autoprognosis

## Get the Data

Get the data [HERE](https://www.dropbox.com/sh/mdd5p6g23cazeaw/AAChiIYcOaTicF388VEb9rYla?dl=0) and put it in a folder called **data**.

## Run Meta-Train

On PMF Meta-dataset:

`
python meta_train.py --dataset pmf --experiment_id pmf01
`

On Tensor-Oboe Meta-Dataset:

`
python meta_train.py --dataset oboe --experiment_id oboe01
`

On Zap Meta-Dataset:

`
python meta_train.py --dataset zap --experiment_id zap01
`

## Run Meta-Test

For meta-testing, you can specify the test_id for different meta-tests on the same meta-learned model. For instance meta-testing with 100 finetuning iterations (faster results), or 10000 iterations (more accurate results):

`
python meta_test.py --dataset oboe --experiment_id oboe01 --test_id 100iters --n_iters 100
`

`
python meta_test.py --dataset oboe --experiment_id oboe01 --test_id 10000iters --n_iters 10000
`



## Run Meta-Test on OpenML Benchmark

`
python test_openml.py --runtime 600 --experiment_id oboe01
`

