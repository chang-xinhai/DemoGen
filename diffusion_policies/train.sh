demo=${1}
algo=${2}
task=${3}
seed=${4}
wandb_mode=${5:-online}

data_root=../data
exp_name=${demo}-${algo}-seed${seed}
run_dir=${data_root}/ckpts/${exp_name}

zarr_path=${data_root}/datasets/demogen/${demo}.zarr

export HYDRA_FULL_ERROR=1
python -W ignore train.py --config-name=${algo}.yaml \
                          task=${task} \
                          hydra.run.dir=${run_dir} \
                          training.debug=False \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            training.num_epochs=1 \
                            task.dataset.zarr_path=${zarr_path}
