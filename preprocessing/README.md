
The preprocessing consists in three different phases as listed below. Phase 1.b requires results of Phase 1.a,
while Phase 3 is completely independent on Phase 1.


- Phase 1.a: slice and merge ERA5. In this phase the different NetCDF files are first sliced to recover
	the desired latitude and longitude area, according to the window specified by the user. Single
	files should be named by the following convenction:
		- ${variable_name}_${year}.nc
	Then, files corresponding to different years but same variable are merged together, in order to obtain
	just one file for each variable, named by the following convenction:
		- ${prefix}${variable_name}.nc

	Phase 1.a is implemented in the bash script `slice_and_merge_years_era5.bash` and can be run by using
	the file `run_slice_and_merge_era5.sh`. Parameters of the preprocessing should be set directly in the
	bash file. Paramenters of the run should be instead set in the .sh file.


- Phase 1.b: preprocessing ERA5. In this phase the NetCDF files created in Phase 1 are used to derive the Python
	array corresponding to the input standardised dataset. Means and standard deviation can be either
	computed on the fly from the data at hand, or precomputed values can be loaded and used directly.
	The latter is the preferred choice when a smaller part of a larger geographical area is investigated
	(e.g. only a region of the italian territory).

	Phase 1.b is implemented in the Python script `preprocessing_era5.sh` and can be run using the file
	`run_preprocessing_era5.sh`. Both parameters of the preprocessing and parameters of the run should be
	set in the .sh file.


- Phase 2: preprocessing GRIPHO and TOPO in order to derive the graphs, the targets and all the other files
	necessary for training and testing wich do not depend on the input. A more detailed description of
	the files is contined in the README in the folder `data_fvg_preprocessed`
	
	Phase 2 is implemented in the Python script `preprocessing_graphs_and_targets.py` and can be run using
	the file `run_preprocessing_graphs_and_targets.sh`. Both parameters of the preprocessing and parameters
	of the run should be set in the .sh file.


Parameters lon_min, lon_max, lat_min, lat_max delineate the area of interest for the prediction and should be set
manually both in Phase 1.a and in Phase 2.

