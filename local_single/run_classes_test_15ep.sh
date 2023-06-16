#!/bin/bash
#SBATCH -A ICT23_CMSP_C
#SBATCH -p m100_usr_prod
## SBATCH --qos=qos_prio
#SBATCH --time 24:00:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=60G
## SBATCH --ntasks-per-node=128 # out of 128
#SBATCH --gres=gpu:4          # 1 gpus per node out of 4
#SBATCH --job-name=cl_test
## SBATCH --mail-type=FAIL,END
## SBATCH --mail-user=sdigioia@sissa.it
#SBATCH -o /m100_scratch/userexternal/sdigioia/climate-DL/local_single/cl_test/run.out
#SBATCH -e /m100_scratch/userexternal/sdigioia/climate-DL/local_single/cl_test/run.err

source /m100/home/userexternal/sdigioia/.bashrc

module load --auto profile/deeplrn
module load --auto openblas/0.3.9--gnu--8.4.0 cuda/11.0

conda activate /m100_scratch/userexternal/sdigioia/geom2

cd /m100_scratch/userexternal/sdigioia/github/climate-DL/local_single/

## training
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config_4.yaml main.py --input_path="/m100_scratch/userexternal/sdigioia/climate-DL/data_fvg_preprocessed/" --output_path="/m100_scratch/userexternal/sdigioia/climate-DL/local_single/cl_test/" --input_file="input_standard.pkl"  --idx_file="idx_train_cl.pkl" --log_file="log.txt" --target_file="target_train_cl.pkl" --mask_target_file="mask_train_cl.pkl" --graph_file="G_train.pkl" --mask_target_file="mask_train_cl.pkl" --subgraphs_file="subgraphs.pkl" --out_checkpoint_file="checkpoint.pth" --out_loss_file="loss.csv" --pct_trainset=1 --epochs=15 --batch_size=64 --step_size=25 --lr=0.0001 --weight_decay=0.0 --model_name="Classifier_old" --loss_fn="sigmoid_focal_loss" --model_type="cl" --use_accelerate --load_checkpoint --no-test_model --no-ctd_training --fine_tuning --performance="accuracy"  --checkpoint_file="/m100_scratch/userexternal/sdigioia/climate-DL/data_fvg_preprocessed/checkpoint_ae_e3.pth" --wandb_project_name="Classifier-test" --mode="train" --lon_dim=7 --lat_dim=7

## continue the training
#accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config_4.yaml main.py --input_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_fvg_preprocessed/" --output_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/cl_test/ctd/" --input_file="input_standard.pkl"  --idx_file="idx_train_cl.pkl" --log_file="log.txt" --target_file="target_train_cl.pkl" --mask_target_file="mask_train_cl.pkl" --graph_file="G_train.pkl" --mask_target_file="mask_train_cl.pkl" --subgraphs_file="subgraphs.pkl" --out_checkpoint_file="checkpoint.pth" --out_loss_file="loss.csv" --pct_trainset=1 --epochs=25 --batch_size=64 --step_size=25 --lr=0.0001 --weight_decay=0.0 --model_name="Classifier_old" --loss_fn="sigmoid_focal_loss" --model_type="cl" --use_accelerate --no-load_checkpoint --no-test_model --ctd_training --fine_tuning --performance="accuracy"  --checkpoint_ctd="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/cl_test_wd0/checkpoint_24.pth" --wandb_project_name="Classifier-test-ctd" --mode="train" --lon_dim=7 --lat_dim=7

