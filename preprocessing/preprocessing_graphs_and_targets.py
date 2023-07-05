import numpy as np
import xarray as xr
import pickle
import time
import argparse
import sys
import torch
import os
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from utils_preprocessing import create_zones, plot_italy

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--output_path', type=str, default='/m100_work/ICT23_ESP_C/vblasone/climate-DL/preprocessing/')
parser.add_argument('--log_file', type=str, default='log_ae.txt')
parser.add_argument('--input_path', type=str, default='/m100_work/ICT23_ESP_C/SHARED/')
parser.add_argument('--target_file', type=str, default='GRIPHO/gripho-v1_1h_TSmin30pct_2001-2016_cut.nc')
parser.add_argument('--topo_file', type=str, default='TOPO/GMTED_DEM_30s_remapdis_GRIPHO.nc')

#-- lat lon grid values
parser.add_argument('--lon_min', type=float, default=6.50)
parser.add_argument('--lon_max', type=float, default=14.25)
parser.add_argument('--lat_min', type=float, default=43.50)
parser.add_argument('--lat_max', type=float, default=47.50)
parser.add_argument('--interval', type=float, default=0.25)
parser.add_argument('--time_dim', type=float, default=140256)
parser.add_argument('--offset_9_cells', type=float, default=0.25)
parser.add_argument('--make_plots', action='store_true', default=False)

#-- other
parser.add_argument('--suffix', type=str, default='')

def cut_window(lon_min, lon_max, lat_min, lat_max, lon, lat, z, pr, time_dim):
    '''
    Derives a new version of the longitude, latitude and precipitation
    tensors, by only retaining the values inside the specified lon-lat rectangle
    Arguments:
        lon_min, lon_max, lat_min, lat_max: integers
        lon, lat, z, pr: tensors
        time_dim: an integet
    Returns:
        The new tensors with the selected values

    '''
    bool_lon = np.logical_and(lon >= lon_min, lon <= lon_max)
    bool_lat = np.logical_and(lat >= lat_min, lat <= lat_max)
    bool_both = np.logical_and(bool_lon, bool_lat)
    lon_sel = lon[bool_both]
    lat_sel = lat[bool_both]
    z_sel = z[bool_both]
    pr_sel = np.array(pr[:,bool_both])
    return lon_sel, lat_sel, z_sel, pr_sel

def select_nodes(lon_centre, lat_centre, lon, lat, pr, cell_idx, cell_idx_array, mask_1_cell_subgraphs, mask_9_cells_subgraphs, offset=0.25, offset_9=0.25):
    '''
    Creates the single cell data structure, by only retaining the values
    correspondent to the nodes that fall inside the considered cell, which
    is identified by its centre lon and lat values and a specified offset
    Arguments:
        lon_centre, lat_centre = integers
        lon, lat, pr = tensors
        cell_idx:
        cell_idx_array:
        offset, offset_9: integers
        mask_1_cell_subgraphs, mask_9_cells_subgraphs: lists
    Returns:

    '''
    bool_lon = np.logical_and(lon >= lon_centre, lon <= lon_centre+offset)
    bool_lat = np.logical_and(lat >= lat_centre, lat <= lat_centre+offset)
    bool_both = np.logical_and(bool_lon, bool_lat)
    bool_lon_9 = np.logical_and(lon >= lon_centre - offset_9, lon <= lon_centre + offset + offset_9)
    bool_lat_9 = np.logical_and(lat >= lat_centre - offset_9, lat <= lat_centre + offset + offset_9)
    bool_both_9 = np.logical_and(bool_lon_9, bool_lat_9)
    bool_both_9 = np.logical_or(bool_both, bool_both_9)
    mask_1_cell_subgraphs[cell_idx, :] = bool_both
    mask_9_cells_subgraphs[cell_idx, :] = bool_both_9
    cell_idx_array[bool_both] = cell_idx
    flag_valid_example = False
    for i in np.argwhere(bool_both):
        if np.all(np.isnan(pr[:,i])):
            cell_idx_array[i] *= -1
        else:
            flag_valid_example = True
    return cell_idx_array, flag_valid_example, mask_1_cell_subgraphs, mask_9_cells_subgraphs

def write_log(s, args, mode='a'):
    with open(args.output_path + args.log_file, mode) as f:
        f.write(s)

def subdivide_train_test_time_indexes(idx_time_years, first_test_year=2016, end_year=2016):
    idx_time_train = []
    idx_time_test = []
    idx_first_test_year = first_test_year-2000-1
    for idx_time_y in idx_time_years[:idx_first_test_year-1]:
        idx_time_train += idx_time_y
    idx_time_train += idx_time_years[idx_first_test_year-1][:-31*24]
    idx_time_test = idx_time_years[idx_first_test_year-1][-31*24:]
    for y in range(first_test_year, end_year + 1): 
        idx_time_test += idx_time_years[y-2000-1]
    idx_time_train.sort(); idx_time_test.sort()
    for i in range(24):
        idx_time_train.remove(i)
    return idx_time_train, idx_time_test


if __name__ == '__main__':

    args = parser.parse_args()
    
    #-----------------------------------------------------
    #----------------- PRELIMINARY STUFF -----------------
    #-----------------------------------------------------

    ## lon/lat diff max to identify when two nodes are connected by an edge
    ## this value is hand-calibrated for an interval of 0.25 degreesand may
    ## be different if the interval changes
    LON_DIFF_MAX = 0.25 / 8 * 2
    LAT_DIFF_MAX = 0.25 / 10 * 2

    ## derive arrays corresponding to the lon/lat low resolution grid points
    lon_low_res_array = np.arange(args.lon_min-args.interval, args.lon_max+args.interval, args.interval)
    lat_low_res_array = np.arange(args.lat_min-args.interval, args.lat_max+args.interval, args.interval)
    lon_low_res_dim = lon_low_res_array.shape[0]
    lat_low_res_dim = lat_low_res_array.shape[0]
    space_low_res_dim = lon_low_res_dim * lat_low_res_dim
    
    lon_high_res_array = np.arange(args.lon_min, args.lon_max, args.interval)
    lat_high_res_array = np.arange(args.lat_min, args.lat_max, args.interval)
    lon_high_res_dim = lon_high_res_array.shape[0] - 1
    lat_high_res_dim = lat_high_res_array.shape[0] - 1

    lon_input_points_array = np.arange(args.lon_min-args.interval*3, args.lon_max+args.interval*4, args.interval)
    lat_input_points_array = np.arange(args.lat_min-args.interval*3, args.lat_max+args.interval*4, args.interval)
    lon_input_points_dim = lon_input_points_array.shape[0]
    lat_input_points_dim = lat_input_points_array.shape[0]
    space_input_points_dim = lon_input_points_dim * lat_input_points_dim

    write_log("\nStart!", args, 'w')
    write_log(f"\nThe considered low-res lon-lat windows is [{lon_low_res_array.min()}, {lon_low_res_array.max()+args.interval}] x [{lat_low_res_array.min()}, {lat_low_res_array.max()+args.interval}]. " + 
              f"\nThe number of points is (lon x lat) {lon_low_res_dim} x {lat_low_res_dim}.", args)
    write_log(f"\nThe considered high-res lon-lat windows is [{lon_high_res_array.min()}, {lon_high_res_array.max()+args.interval}] x [{lat_high_res_array.min()}, {lat_high_res_array.max()+args.interval}]. " + 
              f"\nThe number of cells is (lon x lat) {lon_high_res_dim} x {lat_high_res_dim}", args)

    args.lon_min = args.lon_min - args.interval
    args.lon_max = args.lon_max + args.interval
    args.lat_min = args.lat_min - args.interval
    args.lat_max = args.lat_max + args.interval

    #-----------------------------------------------------
    #------------------- TIME INDEXES --------------------
    #-----------------------------------------------------

    with open("idx_time_2001-2016.pkl", 'rb') as f:
        idx_time_years = pickle.load(f)

    idx_time_train, idx_time_test = subdivide_train_test_time_indexes(idx_time_years)
    time_train_dim = len(range(min(idx_time_train), max(idx_time_train)+1))

    with open(args.output_path + "idx_time_test.pkl", 'wb') as f:
        pickle.dump(idx_time_test, f)

    with open(args.output_path + "idx_time_train.pkl", 'wb') as f:
        pickle.dump(idx_time_train, f)

    write_log(f"\nTrain idxs from {min(idx_time_train)} to {max(idx_time_train)}. Test idxs from {min(idx_time_test)} to {max(idx_time_test)}.", args)
    
    #-----------------------------------------------------
    #----------- CUT LON, LAT, PR, Z TO WINDOW -----------
    #-----------------------------------------------------

    gripho = xr.open_dataset(args.input_path + args.target_file)
    topo = xr.open_dataset(args.input_path + args.topo_file)

    lon = gripho.lon.to_numpy()
    lat = gripho.lat.to_numpy()
    pr = gripho.pr.to_numpy()
    z = topo.z.to_numpy()

    write_log("\nCutting the window...", args)

    # cut gripho and topo to the desired window
    lon_sel, lat_sel, z_sel, pr_sel = cut_window(args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon, lat, z, pr, args.time_dim)
    n_nodes = pr_sel.shape[1]

    write_log(f"\nDone! Window is [{lon_sel.min()}, {lon_sel.max()}] x [{lat_sel.min()}, {lat_sel.max()}] with {n_nodes} nodes.", args)

    #-----------------------------------------------------
    #--------------- DERIVE CELLS MAPPINGS ---------------
    #-----------------------------------------------------

    cell_idx_array = np.zeros(n_nodes) # will contain the mapping of each node to the corresponding low_res cell idx
    mask_1_cell_subgraphs = np.zeros((space_low_res_dim, n_nodes)).astype(bool)     # maps each low_res_cell idx to the corresponding 9 cell mask
    mask_9_cells_subgraphs = np.zeros((space_low_res_dim,n_nodes)).astype(bool)     # maps each low_res_cell idx to the corresponding 9 cells mask
    graph_cells_space = []

    valid_examples_space = [ii * lon_low_res_dim + jj for ii in range(1,lat_low_res_dim-1) for jj in range(1,lon_low_res_dim-1)]
    
    ## start the preprocessing
    write_log(f"\nStarting the preprocessing.", args)
    start = time.time()

    for i, lat_low_res in enumerate(lat_low_res_array):
        for j, lon_low_res in enumerate(lon_low_res_array):
            cell_idx = i * lon_low_res_dim + j
            cell_idx_array, flag_valid_example, mask_1_cell_subgraphs, mask_9_cells_subgraphs = select_nodes(lon_low_res, lat_low_res, lon_sel, lat_sel, pr_sel, cell_idx,
                    cell_idx_array, mask_1_cell_subgraphs, mask_9_cells_subgraphs, offset=args.interval, offset_9=args.offset_9_cells)         
            if cell_idx in valid_examples_space:
                if flag_valid_example:
                    idx_list = np.array([ii * lon_low_res_dim + jj for ii in range(i-1,i+2) for jj in range(j-1,j+2)])
                    _ = [graph_cells_space.append(abs(d)) for d in idx_list]
                else:
                    valid_examples_space.remove(cell_idx)
    
    graph_cells_space = list(set(graph_cells_space))
    graph_cells_space.sort()
    valid_examples_space.sort()

    end = time.time()
    write_log(f'\nLoop took {end - start} s', args)

    ## keep only the graph cells space idxs
    mask_graph_cells_space = np.in1d(abs(cell_idx_array), graph_cells_space)
    mask_1_cell_subgraphs = mask_1_cell_subgraphs[:,mask_graph_cells_space]
    mask_1_cell_subgraphs = torch.tensor(mask_1_cell_subgraphs)
    mask_9_cells_subgraphs = mask_9_cells_subgraphs[:,mask_graph_cells_space]
    mask_9_cells_subgraphs = torch.tensor(mask_9_cells_subgraphs)
    
    idx_test = [t * space_low_res_dim + s for s in range(space_low_res_dim) for t in idx_time_test if s in valid_examples_space]
    idx_test = np.array(idx_test)
   
    idx_train_ae = [t * space_low_res_dim + s for s in range(space_low_res_dim) for t in idx_time_train if s in valid_examples_space]
    idx_train_ae = np.array(idx_train_ae)

    lon_sel = lon_sel[mask_graph_cells_space]
    lat_sel = lat_sel[mask_graph_cells_space]
    z_sel = z_sel[mask_graph_cells_space]
    pr_sel = pr_sel[:, mask_graph_cells_space] # (time, num_nodes)
    cell_idx_array = cell_idx_array[mask_graph_cells_space]

    n_nodes = cell_idx_array.shape[0]
    
    ## write some files
    with open(args.output_path + 'mask_1_cell_subgraphs' + args.suffix + '.pkl', 'wb') as f:
        pickle.dump(mask_1_cell_subgraphs, f)
    
    with open(args.output_path + 'mask_9_cells_subgraphs' + args.suffix + '.pkl', 'wb') as f:
        pickle.dump(mask_9_cells_subgraphs, f)
   
    with open(args.output_path + 'idx_test.pkl', 'wb') as f:
        pickle.dump(idx_test, f)
    
    with open(args.output_path + 'idx_train_ae.pkl', 'wb') as f:
        pickle.dump(idx_train_ae, f)

    with open(args.output_path + 'valid_examples_space.pkl', 'wb') as f:   # low res cells indexes valid as examples for the training
        pickle.dump(valid_examples_space, f)

    with open(args.output_path + 'graph_cells_space.pkl', 'wb') as f:      # all low res cells that are used (examples + surroundings)
        pickle.dump(graph_cells_space, f)
        
    with open(args.output_path + 'cell_idx_array.pkl', 'wb') as f:         # array that assigns to each high res node the corresponding low res cell index
        pickle.dump(cell_idx_array, f)

    #-------------------------------------------------
    #----- CLASSIFICATION AND REGRESSION TARGETS -----
    #-------------------------------------------------

    threshold = 0.1 # mm
    pr_sel = pr_sel.swapaxes(0,1) # (num_nodes, time)
    pr_sel_train = pr_sel[:,:max(idx_time_train)+1]
    pr_sel_train_cl = np.array([np.where(pr >= threshold, 1, 0) for pr in pr_sel_train], dtype=np.float32)
    pr_sel_train_cl[np.isnan(pr_sel_train)] = np.nan
    pr_sel_train_reg = np.array([np.where(pr >= threshold, np.log1p(pr), np.nan) for pr in pr_sel_train], dtype=np.float32)
    pr_sel_test = pr_sel[:,min(idx_time_test):max(idx_time_test)+1]

    weights = [1,2,5,10,20,50]
    weights_thresholds = [0,1,5,10,20,50]

    reg_weights = np.ones(pr_sel_train.shape, dtype=np.float32) * weights[0]
    
    for i, w in enumerate(weights):
        thresh = weights_thresholds[i]
        reg_weights[pr_sel_train >= thresh] = w

    reg_weights[np.isnan(pr_sel_train)] = np.nan

    reg_weights = torch.tensor(reg_weights)

    with open(args.output_path + 'weights_reg_train.pkl', 'wb') as f:
        pickle.dump(reg_weights, f)

    #-------------------------------------------------
    #----------- STANDARDISE LON LAT AND Z -----------
    #-------------------------------------------------
    
    use_precomputed_means_stds = True

    if use_precomputed_means_stds:
        write_log(f"\nUsing statistics over italy for lat, lon and z.", args)
        with open("lat_lon_z_best.pkl", 'rb') as f:
            lat_lon_z_best = pickle.load(f)
        z_sel_s = (z_sel - np.mean(lat_lon_z_best[:,2])) / np.std(lat_lon_z_best[:,2])
        lon_sel_s = (lon_sel - np.mean(lat_lon_z_best[:,1])) / np.std(lat_lon_z_best[:,1])
        lat_sel_s = (lat_sel - np.mean(lat_lon_z_best[:,0])) / np.std(lat_lon_z_best[:,0])
    else:
        write_log(f"\nUsing local statistics for lat, lon and z.", args)
        z_sel_s = (z_sel - z_sel.mean()) / z_sel.std()
        lon_sel_s = (lon_sel - lon_sel.mean()) / lon_sel.std()
        lat_sel_s = (lat_sel - lat_sel.mean()) / lat_sel.std()
    
    lon_lat_z_s = np.empty((z_sel_s.shape[0], 3))

    lon_lat_z_s[:,0] = lon_sel_s
    lon_lat_z_s[:,1] = lat_sel_s
    lon_lat_z_s[:,2] = z_sel_s

    pos = np.column_stack((lon_sel,lat_sel))

    #-----------------------------------------------------
    #----------------------- EDGES -----------------------
    #-----------------------------------------------------

    edge_index = np.empty((2,0), dtype=int)
    edge_attr = np.empty((2,0), dtype=float)

    for ii, xi in enumerate(pos):
        bool_lon = abs(pos[:,0] - xi[0]) < LON_DIFF_MAX
        bool_lat = abs(pos[:,1] - xi[1]) < LAT_DIFF_MAX
        bool_both = np.logical_and(bool_lon, bool_lat)
        jj_list = np.flatnonzero(bool_both)
        xj_list = pos[bool_both, :]
        for jj, xj in zip(jj_list, xj_list):
            if not np.array_equal(xi, xj):
                edge_index = np.concatenate((edge_index, np.array([[ii], [jj]])), axis=-1, dtype=int)
                edge_attr = np.concatenate((edge_attr, np.array([[xj[0] - xi[0]], [xj[1] - xi[1]]])), axis=-1, dtype=float)
        #write_log(f"\nStart node: {xi} - done. Node has {n_neighbours} neighbours.", args)

    edge_attr = edge_attr.swapaxes(0,1)
    edge_attr[:,0] = edge_attr[:,0] / edge_attr[:,0].max() 
    edge_attr[:,1] = edge_attr[:,1] / edge_attr[:,1].max()
    
    ## intervals to categorical values
    lon_m1 = edge_attr[:,0]<-0.5
    lon_0 = np.logical_and(edge_attr[:,0]>-0.5, edge_attr[:,0]<0.5)
    lon_1 = edge_attr[:,0]>0.5
    lat_m1 = edge_attr[:,1]<-0.5
    lat_0 = np.logical_and(edge_attr[:,1]>-0.5, edge_attr[:,1]<0.5)
    lat_1 = edge_attr[:,1]>0.5
    
    bool_N = np.logical_and(lon_0, lat_m1)     # (0, -1)
    bool_NE = np.logical_and(lon_m1, lat_m1)   # (-1,-1)
    bool_E = np.logical_and(lon_m1, lat_0)      # (-1, 0)
    bool_SE = np.logical_and(lon_m1, lat_1)    # (-1, 1)
    bool_S = np.logical_and(lon_0, lat_1)      # (0, 1)
    bool_SO = np.logical_and(lon_1, lat_1)     # (1, 1)
    bool_O = np.logical_and(lon_1, lat_0)      # (1, 0)
    bool_NO = np.logical_and(lon_1, lat_m1)    # (1, m)

    edge_attr_cat = np.empty(edge_attr.shape[0], dtype=int)
    
    edge_attr_cat[bool_N] = 0
    edge_attr_cat[bool_NE] = 1
    edge_attr_cat[bool_E] = 2
    edge_attr_cat[bool_SE] = 3
    edge_attr_cat[bool_S] = 4
    edge_attr_cat[bool_SO] = 5
    edge_attr_cat[bool_O] = 6
    edge_attr_cat[bool_NO] = 7
    
    #-----------------------------------------------------
    #---------------------- GRAPHS -----------------------
    #-----------------------------------------------------

    ## create the graph objects
    G_test = Data(num_nodes=z_sel_s.shape[0], pos=torch.tensor(pos), y=torch.tensor(pr_sel_test), pr_cl=torch.zeros(pr_sel_test.shape),
            pr_reg=torch.zeros(pr_sel_test.shape), low_res=torch.tensor(abs(cell_idx_array)).int(), edge_index=torch.tensor(edge_index),
            edge_attr=torch.tensor(edge_attr_cat), x=torch.tensor(lon_lat_z_s))
    G_train = Data(num_nodes=z_sel_s.shape[0], x=torch.tensor(lon_lat_z_s), edge_index=torch.tensor(edge_index), edge_attr=torch.tensor(edge_attr_cat),
            low_res=torch.tensor(abs(cell_idx_array)).int())

    ## write some files
    with open(args.output_path + 'G_test' + args.suffix + '.pkl', 'wb') as f:
        pickle.dump(G_test, f)

    with open(args.output_path + 'G_train' + args.suffix + '.pkl', 'wb') as f:
        pickle.dump(G_train, f)
    
    with open(args.output_path + 'target_train_cl.pkl', 'wb') as f:
        pickle.dump(torch.tensor(pr_sel_train_cl), f)    
     
    with open(args.output_path + 'target_train_reg.pkl', 'wb') as f:
        pickle.dump(torch.tensor(pr_sel_train_reg), f)    
     
    write_log(f"\nIn total, preprocessing took {time.time() - start} seconds", args)    

    #-----------------------------------------------------
    #---------------------- INDEXES ----------------------
    #-----------------------------------------------------

    ## create the indexes list for the dataloader
    write_log("\nLet's now create the list of indexes for the training.", args)

    start = time.time()

    idx_train_cl = []
    idx_train_reg = []
    
    mask_train_cl = ~np.isnan(pr_sel_train_cl)
    mask_train_reg = np.logical_and(~np.isnan(pr_sel_train_reg), pr_sel_train_reg >= threshold) 

    c = 0
    for s in range(space_low_res_dim):
        mask_1 = np.in1d(cell_idx_array, s) # shape = (n_nodes)
        if s in valid_examples_space:
            c += 1
            i = s // space_low_res_dim
            j = s % space_low_res_dim
            idx_list = np.array([ii * lon_low_res_dim + jj for ii in range(i-1,i+2) for jj in range(j-1,j+2)])
            for t in idx_time_train:
                if not (~mask_train_cl[mask_1,t]).all(): # if at leat one value is not nan
                    k = t * space_low_res_dim + s
                    idx_train_cl.append(k)
                    if not (~mask_train_reg[mask_1,t]).all(): # if at least one value is not nan and >= threshold
                        idx_train_reg.append(k)
            if c % 10 == 0:
                write_log(f"\nSpace idx {s} done.", args)     
    
    idx_train_cl = np.array(idx_train_cl)
    idx_train_reg = np.array(idx_train_reg)

    write_log(f"\nCreating the idx array took {time.time() - start} seconds", args)    

    ## write some files
    with open(args.output_path + 'idx_train_cl.pkl', 'wb') as f:
        pickle.dump(idx_train_cl, f)

    with open(args.output_path + 'idx_train_reg.pkl', 'wb') as f:
        pickle.dump(idx_train_reg, f)
    
    with open(args.output_path + 'mask_train_cl.pkl', 'wb') as f:
        pickle.dump(torch.tensor(mask_train_cl), f)

    with open(args.output_path + 'mask_train_reg.pkl', 'wb') as f:
        pickle.dump(torch.tensor(mask_train_reg), f)
   
    #-----------------------------------------------------
    #--------------------- SUBGRAPHS ---------------------
    #-----------------------------------------------------

    subgraphs = [[] for i in range(max(valid_examples_space)+1)]

    for space_idx in valid_examples_space:
        lat_idx = space_idx // lon_low_res_dim
        lon_idx = space_idx % lon_low_res_dim
        mask_subgraph = mask_1_cell_subgraphs[space_idx] # shape = (n_nodes,)
        subgraph = G_train.subgraph(subset=mask_subgraph)
        cell_idx_list = torch.tensor([ii * lon_low_res_dim + jj for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        subgraph["mask_1_cell"] = mask_1_cell_subgraphs[space_idx].cpu()  # (n_nodes)
        subgraph["mask_9_cells"] = mask_9_cells_subgraphs[space_idx].cpu()         # (n_nodes)
        subgraphs[space_idx] = subgraph
        
        if space_idx % 10 == 0:
            write_log(f"\nDone until {space_idx}.", args)
            
    with open(args.output_path + 'subgraphs.pkl', 'wb') as f:
        pickle.dump(subgraphs, f)

    #-----------------------------------------------------
    #----------------------- PLOTS -----------------------
    #-----------------------------------------------------

    if args.make_plots:

        zones = create_zones()

        x_size = lon_input_points_dim
        y_size = lat_input_points_dim
        font_size = int(18 / 7 * x_size)

        fig, ax = plt.subplots(figsize=(x_size, y_size))

        _ = ax.scatter(pos[:,0], pos[:,1], c='grey',alpha=0.2,marker="s", s=40, cmap='turbo')
        c = torch.tensor(abs(cell_idx_array)).int()

        for s in subgraphs:
            if s != []:
                alpha = np.ones(s.mask_1_cell.sum())*0.2
                _ = ax.scatter(pos[:,0][s.mask_1_cell],pos[:,1][s.mask_1_cell],
                        c=c[s.mask_1_cell], marker="s", s=40, vmin = 0,
                        vmax = space_low_res_dim, cmap='turbo', alpha=alpha)

        for l in lon_input_points_array:
            _ = ax.plot([l, l], [lat_input_points_array.min(), lat_input_points_array.max()], 'b', alpha=0.2)

        for l in lat_input_points_array:
            _ = ax.plot([lon_input_points_array.min(), lon_input_points_array.max()], [l,l], 'b', alpha=0.2)

        lon_plot_array = np.arange(lon_low_res_array.min(), lon_low_res_array.max()+args.interval*2, args.interval)
        lat_plot_array = np.arange(lat_low_res_array.min(), lat_low_res_array.max()+args.interval*2, args.interval)

        for l in lon_plot_array:
            _ = ax.plot([l, l], [lat_plot_array.min(), lat_plot_array.max()], 'k', alpha=0.4)

        for l in lat_plot_array:
            _ = ax.plot([lon_plot_array.min(), lon_plot_array.max()], [l,l], 'k', alpha=0.4)

        for i, lat_i in enumerate(lat_low_res_array):
            for j, lon_j in enumerate(lon_low_res_array):
                k = i * lon_low_res_dim + j
                _ = ax.text(lon_j+0.1, lat_i+0.1, k, fontsize=22)
                                                                                                        
        plot_italy(zones, color='black', ax=ax, alpha_fill=0, linewidth=2)
        plt.xlim([lon_input_points_array.min() - 0.25, lon_input_points_array.max() + 0.25])
        plt.ylim([lat_input_points_array.min() - 0.25, lat_input_points_array.max() + 0.25])
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig(args.output_path + f'graph.pdf', dpi=400, bbox_inches='tight', pad_inches=0.0)

    write_log("\nDone!", args)


