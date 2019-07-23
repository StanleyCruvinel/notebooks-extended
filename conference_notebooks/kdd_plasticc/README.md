# 1. Introduction
These notebokks demo the single GPU model of 8th place [Rapids.ai](https://rapids.ai) solution of [PLAsTiCC Astronomical Classification](https://www.kaggle.com/c/PLAsTiCC-2018). The full blog can be found [here](https://medium.com/rapids-ai/make-sense-of-the-universe-with-rapids-ai-d105b0e5ec95).

# 2. Build and run with bare-metal conda install
## Requirement
1. cuda==9.2 or 10.0
2. anaconda

## Install depencencies. Please see [Rapids.ai get started](https://rapids.ai/start.html) for detailed instructions
```bash
# assume you have cuda 10.0 and use python 3.6
$ conda create -n rapids python=3.6
$ source activate rapids
$ conda install -c nvidia -c rapidsai -c numba -c conda-forge -c pytorch -c defaults \
    cudf=0.8 cuml=0.8 cugraph=0.8 python=3.6 cudatoolkit=10.0
$ pip install xgboost seaborn termcolor scikit-learn tensorflow-gpu
$ conda install jupyter notebook
```  

## Download data
Download data from [link](https://www.kaggle.com/c/PLAsTiCC-2018/data) and modify the `PATH` variable in notebooks accordingly.

## Run the notebooks in following order
1. `rapids_lsst_gpu_only_demo.ipynb`
2. `preprocess_for_rnn_step_by_step.ipynb`
3. `preprocess_for_rnn.ipynb`
4. `rnn_bottleneck_extraction.ipynb`
5. `rapids_lsst_gpu_only_demo_with_bottleneck.ipynb`

## To compare with CPU solution please run `rapids_lsst_full_demo.ipynb`
