#!/bin/bash

##---------------------------------#
#				   #
#    INPUT DATA TO PERSONALIZE 	   #
#				   #
##---------------------------------#


input_path='/m100_work/ICT23_ESP_C/SHARED/ERA5/'
output_path='/m100_work/ICT23_ESP_C/SHARED/preprocessed/fvg/'
prefix='sliced_'

## fvg
lon_min=12.75
lon_max=14.00
lat_min=45.50
lat_max=46.75

## north
#lon_min=6.75
#lon_max=14.00
#lat_min=43.75
#lat_max=47.00

## interval
interval=0.25


##---------------------------------#
#				   #
#               CODE         	   #
#				   #
##---------------------------------#


module load profile/advanced netcdf/4.7.3--spectrum_mpi--10.3.1--binary eccodes/2.23.0 cdo

cd ${input_path}

lon_min_era5=$(echo $lon_min-3*$interval | bc)
lon_max_era5=$(echo $lon_max+3*$interval | bc)
lat_min_era5=$(echo $lat_min-3*$interval | bc)
lat_max_era5=$(echo $lat_max+3*$interval | bc)

echo $lon_min_era5
echo $lon_max_era5
echo $lat_min_era5
echo $lat_max_era5

## slice each file to the desired lon and lat window
for v in 'q' 't' 'u' 'v' 'z' ; do
    files=$(ls ${v}_*.nc)
    for file in $files ; do
        cdo sellonlatbox,$lon_min_era5,$lon_max_era5,$lat_min_era5,$lat_max_era5 "${input_path}${file}" "${output_path}${prefix}${file}"
    done
done

echo "Done"

cd ${output_path}

## merge all the sliced files into a single file for all the time span considered
for v in 'q' ; do #'t' 'u' 'v' 'z' ; do
    cdo -O -f nc4 -z zip -L -b F32 mergetime "${prefix}${v}_2001.nc" "${prefix}${v}_2002.nc" "${prefix}${v}_2003.nc" "${prefix}${v}_2004.nc" "${prefix}${v}_2005.nc" "${prefix}${v}_2006.nc" "${prefix}${v}_2007.nc" "${prefix}${v}_2008.nc" "${prefix}${v}_2009.nc" "${prefix}${v}_2010.nc" "${prefix}${v}_2011.nc" "${prefix}${v}_2012.nc" "${prefix}${v}_2013.nc" "${prefix}${v}_2014.nc" "${prefix}${v}_2015.nc" "${prefix}${v}_2016.nc" "${prefix}${v}.nc"
done

