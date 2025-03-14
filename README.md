# This repository provides the solution for <br/>*The IBIS Challenge*
## Installation
To create and activate a conda environment:
```console
conda env create -f environment.yml
conda activate lcnn
```

## Examples
To run training procedure for a particular protein you could use next commands </br>(Please note,that you must change the variables in brackets):
```console
python train_pipe_g2a.py --neg_type alien_mono --tf_name {TF_name} --n_workers 2 --device_id 0  --exp_name {checkp_log_dir}
python train_pipe_a2g.py --neg_type alien_mono --tf_name {TF_name} --n_workers 2 --device_id 0  --exp_name {checkp_log_dir}
```
To assemble your submit:
```console
python final_assembler.py --device_id 0 --subm_name {exp_name} --exp_type {HTS|PBM|SMS}
python final_assembler_chs_bi.py --device_id 0 --subm_name {exp_name}
python final_assembler_ghts_bi.py --device_id 0 --subm_name {exp_name}
```
Note, you should specify the path of your directory with IBIS templates and test data through "--template_dir {Path}"
Files with scores will be saved in "{Path}/{exp_name}"

## Important remark 1, The assembly of datasets!
You should have the bibis package for the assembly of new datasets,</br>Please, follow this link: https://github.com/autosome-ru/ibis-challenge</br> 

## Important remark 2, Logger!
I use MLFlow logger during the training procedure, so you should change it <br/>or use the following before and during execution of train scripts:
```console
tmux new -t mlflow
mlflow server --host localhost --port 5005
```
Then you could look at your results in browser at the following link : "http://localhost:5005/"
