# DeepPipe: Deep Pipeline Embeddings for AutoML

This is the repository of the paper *Deep Pipeline Embeddings for AutoML* accepted at KDD 2023.

<p align="center">
  <img src="figures/DeepPipe_architecture.png" alt="Alt Text" width="400px">
</p>


## Prepare Environment
Create environment on Python with Conda:

`
conda -n deep_pipe python=3.9
`

Install dependencies for running DeepPipe:

`
pip install -r requirements_deep_pipe.txt
`

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

## Run on Previous Checkpoints

If you do not want to meta-train, you can use our checkpoints (already meta-trained DeepPipes):

`
python meta_test.py --dataset oboe --experiment_id DeepPipeTOboe
`

## Run Meta-Test on OpenML Benchmark

`
python test_openml.py --runtime 600 --experiment_id oboe01
`

