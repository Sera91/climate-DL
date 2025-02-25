import pickle
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

import time

from torch_geometric.data import Data

class Dataset_pr(Dataset):

    def __init__(self, args, pad=2, lat_dim=16, lon_dim=31):
        super().__init__()
        self.pad = pad
        self.lat_low_res_dim = lat_dim # number of points in the GRIPHO rectangle (0.25 grid)
        self.lon_low_res_dim = lon_dim
        self.space_low_res_dim = self.lat_low_res_dim * self.lon_low_res_dim
        self.args = args
        self.length = None

    def _load_data_into_memory(self):
        raise NotImplementedError
    
    def __len__(self):
        return self.length

class Dataset_pr_ae(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key = self._load_data_into_memory()
    
    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f) 
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        self.length = len(idx_to_key)
        return input, idx_to_key

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        input = torch.zeros((25, 5, 5, 6, 6))
        input[:] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        return input

class Dataset_e(Dataset_pr_ae):
    
    def __getitem__(self, idx):
        k = self.idx_to_key[idx]
        time_idx = k[1]
        space_idx = k[0]
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        input = torch.zeros((25, 5, 5, 6, 6))
        input[:] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        return input, k 

class Dataset_pr_gnn(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key, self.target, self.graph, self.subgraphs, self.cell_idxs = self._load_data_into_memory()
        #self.t_input=0
        #self.t_gnn=0

    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        with open(self.args.input_path + self.args.target_file, 'rb') as f:
            target = pickle.load(f)        
        with open(self.args.input_path + self.args.graph_file, 'rb') as f:
            graph = pickle.load(f)
        #with open(self.args.input_path + self.args.mask_target_file, 'rb') as f:
        #    mask_target = pickle.load(f)
        with open(self.args.input_path + self.args.subgraphs_file, 'rb') as f:
            subgraphs = pickle.load(f)
        with open(self.args.input_path + self.args.cell_idxs_file, 'rb') as f:
            cell_idxs = pickle.load(f)
        self.length = len(idx_to_key)
        self.low_res_abs = abs(graph.low_res)
        return input, idx_to_key, target, graph, subgraphs, cell_idxs

    def __getitem__(self, idx):
        #t0 = time.time()
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #print(self.space_low_res_dim, self.lon_low_res_dim, self.lat_low_res_dim, idx, k, time_idx, space_idx, lat_idx, lon_idx)
        
        #-- derive input
        input = torch.zeros((25, 5, 5, 6, 6)) # (time, var, lev, lat, lon)
        input[:, :] = torch.tensor(self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4])
        #t1 = time.time()
        #self.t_input += (t1 - t0)
        #-- derive gnn data
#        subgraph = self.subgraphs[space_idx].clone()#.cuda()
        s = self.subgraphs[space_idx]
        subgraph = Data(edge_index = torch.tensor(s['edge_index']), x = torch.tensor(s['x']), num_nodes = s['x'].shape[0])
#        subgraph = Data(edge_index = s['edge_index'], edge_attr = s['edge_attr'], num_nodes = s['num_nodes'], z = s['z'], low_res = s['low_res'], mask_1_cell = s['mask_1_cell'])
        #print(subgraph.mask_1_cell.device, self.mask_target[:,time_idx].device)
        #train_mask = subgraph.mask_1_cell * self.mask_target[:,time_idx]#.cuda() # shape = (n_nodes,)
        #subgraph["train_mask"] = train_mask[subgraph.mask_1_cell]
#        train_mask = self.mask_target[:,time_idx][subgraph.mask_1_cell]
#        subgraph["train_mask"] = train_mask    
#        y = self.target[subgraph.mask_1_cell, time_idx][train_mask] # shape = (n_nodes_subgraph,)
        y = torch.tensor(self.target[self.cell_idxs == space_idx, time_idx]) # shape = (n_nodes_subgraph,)
        subgraph["y"] = y#.cuda()
        #self.t_gnn += (time.time() - t1)
        return input, subgraph
    
class Dataset_pr_test(Dataset_pr):

    def __init__(self, time_min, time_max, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_min = time_min
        self.time_max = time_max
        self.input, self.idx_to_key, self.subgraphs, self.test_graph = self._load_data_into_memory()

    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)   
        with open(self.args.input_path + self.args.subgraphs, 'rb') as f:
            subgraphs = pickle.load(f)
        with open(self.args.input_path + self.args.graph_file_test, 'rb') as f:
            test_graph = pickle.load(f)
        self.length = len(idx_to_key)
        return input, idx_to_key, subgraphs, test_graph

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        input = torch.zeros((25, 5, 5, 6, 6)) # (time, var, lev, lat, lon)
        input[:, :] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        ##-- derive gnn data
        #print(idx, k, time_idx, space_idx, lat_idx, lon_idx)
        subgraph = self.subgraphs[space_idx].clone()
        #cell_idx_list = torch.tensor([ii * self.lon_low_res_dim + jj for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        #subgraph["idx_list"] = cell_idx_list
        subgraph["time_idx"] = time_idx - self.time_min
        #subgraph["test_mask"] = subgraph.mask_1_cell[subgraph.mask_1_cell]
        y = self.test_graph.y[subgraph.mask_1_cell, time_idx - self.time_min]
        subgraph["y"] = y
        return input, subgraph

class Dataset_pr_ft_gnn(Dataset_pr_gnn):

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        encoding = torch.zeros((9, 128))
        cell_idx_list = torch.tensor([ii * self.lon_low_res_dim + jj for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        for i, s in enumerate(cell_idx_list):
            encoding[i, :] = self.input[s, time_idx, :]
        
        #-- derive gnn data
        mask_subgraph = self.mask_9_cells[space_idx] # shape = (n_nodes,)
        subgraph = self.graph.subgraph(subset=mask_subgraph)
        mask_y_nodes = self.mask_1_cell[space_idx] * self.mask_target[:,time_idx] # shape = (n_nodes,)
        subgraph["train_mask"] = mask_y_nodes[mask_subgraph]
        y = self.target[mask_subgraph, time_idx] # shape = (n_nodes_subgraph,)
        subgraph["y"] = y
        subgraph["idx_list"] = cell_idx_list
        return encoding, subgraph


def custom_collate_fn_ae(batch):
    input = torch.stack(batch)
    input = default_convert(input)
    return input

def custom_collate_fn_e(batch):
    input = torch.stack([item[0] for item in batch])
    idxs = [item[1] for item in batch] 
    input = default_convert(input)
    idxs = default_convert(idxs)
    idxs = torch.stack(idxs)
    return input, idxs

def custom_collate_fn_gnn(batch):
    input = torch.stack([item[0] for item in batch]) # shape = (batch_size, 9, 25, 5, 5, 6, 6)
    data = [item[1] for item in batch]
    input = default_convert(input)
    return input, data
    
