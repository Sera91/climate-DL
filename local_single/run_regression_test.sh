#!/bin/bash
#SBATCH -A ict23_esp_C
#SBATCH -p m100_usr_prod
## SBATCH --qos=qos_prio
#SBATCH --time 24:00:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=60G
## SBATCH --ntasks-per-node=128   # 8 tasks out of 128
#SBATCH --gres=gpu:4          # 1 gpus per node out of 4
#SBATCH --job-name=reg_test
## SBATCH --mail-type=FAIL,END
## SBATCH --mail-user=vblasone@ictp.it
#SBATCH -o /m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/reg_test/run.out
#SBATCH -e /m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/reg_test/run.err

source /m100/home/userexternal/vblasone/.bashrc

conda activate /m100/home/userexternal/vblasone/geometric

cd /m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/

## training
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config_4.yaml main.py --input_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_fvg_preprocessed/" --output_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/reg_test/" --input_file="input_standard.pkl"  --idx_file="idx_train_reg.pkl" --log_file="log.txt" --target_file="target_train_reg.pkl" --mask_target_file="mask_train_reg.pkl" --graph_file="G_train.pkl" --mask_target_file="mask_train_reg.pkl" --subgraphs_file="subgraphs.pkl" --out_checkpoint_file="checkpoint.pth" --out_loss_file="loss.csv" --pct_trainset=1 --epochs=50 --batch_size=64 --step_size=25 --lr=0.0001 --weight_decay=0.0 --model_name="Regressor_old" --loss_fn="mse_loss" --model_type="reg" --use_accelerate --load_checkpoint --no-test_model --no-ctd_training --fine_tuning --checkpoint_file="/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_fvg_preprocessed/checkpoint_ae_e3.pth" --wandb_project_name="Regressor-test" --mode="train" --lon_dim=7 --lat_dim=7 --weights_file="weights_reg_train.pkl"

## continue training
#accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config_4.yaml main.py --input_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_fvg_preprocessed/" --output_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/reg_test/ctd/" --input_file="input_standard.pkl"  --idx_file="idx_train_reg.pkl" --log_file="log.txt" --target_file="target_train_reg.pkl" --mask_target_file="mask_train_reg.pkl" --graph_file="G_train.pkl" --mask_target_file="mask_train_reg.pkl" --subgraphs_file="subgraphs.pkl" --out_checkpoint_file="checkpoint.pth" --out_loss_file="loss.csv" --pct_trainset=1 --epochs=50 --batch_size=64 --step_size=25 --lr=0.0001 --weight_decay=0.0 --model_name="Regressor_old" --loss_fn="mse_loss" --model_type="reg" --use_accelerate --no-load_checkpoint --no-test_model --ctd_training --fine_tuning --checkpoint_ctd="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/reg_test/checkpoint_49.pth" --wandb_project_name="Regressor-test-ctd" --mode="train" --lon_dim=7 --lat_dim=7 --weights_file="weights_reg_train.pkl"

