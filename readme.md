# DeepPipe

## Install Dependencies

Download the repo and install requirements:

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

## Run Meta-Test

On Tensor-Oboe Meta-Dataset:

`
python meta_test.py --dataset oboe --experiment_id oboe01
`

## Run Meta-Test on OpenML Benchmark

`
python test_openml.py --runtime 600 --experiment_id oboe01
`

