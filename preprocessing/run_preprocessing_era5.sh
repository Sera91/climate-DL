#!/bin/bash
#SBATCH -A ict23_esp_C
#SBATCH -p m100_usr_prod
#SBATCH --time 2:00:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=64   # 8 tasks out of 128
#SBATCH --job-name=prep_era5
# SBATCH --mail-type=FAIL,END
# SBATCH --mail-user=
#SBATCH -o /m100_work/ICT23_ESP_C/SHARED/preprocessed/fvg/prep_era5.out
#SBATCH -e /m100_work/ICT23_ESP_C/SHARED/preprocessed/fvg/prep_era5.err

source /m100/home/userexternal/vblasone/.bashrc

conda activate /m100/home/userexternal/vblasone/geometric

module load profile/advanced netcdf/4.7.3--spectrum_mpi--10.3.1--binary eccodes/2.23.0 szip/2.1.1--gnu--8.4.0 cdo

cd /m100_work/ICT23_ESP_C/SHARED/climate-DL/preprocessing/

python3 preprocessing_era5.py --input_path='/m100_work/ICT23_ESP_C/SHARED/preprocessed/fvg/' --output_path='/m100_work/ICT23_ESP_C/SHARED/preprocessed/fvg/' --input_files_prefix="sliced_" --stats_path='/m100_work/ICT23_ESP_C/SHARED/preprocessed/italy/' --means_file='means.pkl' --stds_file='stds.pkl' --load_stats

