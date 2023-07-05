#!/bin/bash
#SBATCH -A ict23_esp_C
#SBATCH -p m100_usr_prod
#SBATCH --time 20:00:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=64   # 8 tasks out of 128
# --gres=gpu:1          # 1 gpus per node out of 4
#SBATCH --job-name=slice_merge_era5
# SBATCH --mail-type=FAIL,END
# SBATCH --mail-user=vblasone@ictp.it
#SBATCH -o /m100_work/ICT23_ESP_C/vblasone/climate-DL/preprocessing/slice_merge_era5.out
#SBATCH -e /m100_work/ICT23_ESP_C/vblasone/climate-DL/preprocessing/slice_merge_era5.err

source /m100/home/userexternal/vblasone/.bashrc

conda deactivate

module load profile/advanced netcdf/4.7.3--spectrum_mpi--10.3.1--binary eccodes/2.23.0 szip/2.1.1--gnu--8.4.0 cdo

cd /m100_work/ICT23_ESP_C/vblasone/climate-DL/preprocessing/

bash slice_and_merge_years_era5.bash

