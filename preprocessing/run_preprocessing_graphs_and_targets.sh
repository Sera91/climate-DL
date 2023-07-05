#!/bin/bash
#SBATCH -A ict23_esp_C
#SBATCH -p m100_usr_prod
#SBATCH --time 01:00:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=0
#SBATCH --ntasks-per-node=128   # 8 tasks out of 128
#SBATCH --job-name=prep-fvg
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=
#SBATCH -o /m100_work/ICT23_ESP_C/SHARED/preprocessed/fvg/prep-graph.out
#SBATCH -e /m100_work/ICT23_ESP_C/SHARED/preprocessed/fvg/prep-graph.err

source /m100/home/userexternal/vblasone/.bashrc

conda activate /m100/home/userexternal/vblasone/geometric

cd /m100_work/ICT23_ESP_C/SHARED/climate-DL/preprocessing/

## fvg
python3 preprocessing_graphs_and_targets.py --input_path='/m100_work/ICT23_ESP_C/SHARED/' --output_path='/m100_work/ICT23_ESP_C/SHARED/prerocessed/fvg/' --log_file='log.txt' --lon_min=12.75 --lon_max=14.00 --lat_min=45.50 --lat_max=46.75 --interval=0.25 --time_dim=140256 --suffix="" --make_plots

## north
#python3 preprocessing_graphs_and_targets.py --input_path='/m100_work/ICT23_ESP_C/SHARED/' --output_path='/m100_work/ICT23_ESP_C/SHARED/preprocessed/fvg/' --log_file='log.txt' --lon_min=6.75 --lon_max=14.00 --lat_min=43.75 --lat_max=47.00 --interval=0.25 --time_dim=140256 --suffix="" --make_plots

