import pickle
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

import time

from torch_geometric.data import Data

class Dataset_pr(Dataset):

    def __init__(self, args, lat_dim, lon_dim, pad=2):
        super().__init__()
        self.pad = pad
        self.lon_low_res_dim = lon_dim
        self.lat_low_res_dim = lat_dim
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
        self.input, self.idx_to_key, self.target, self.graph, self.subgraphs, self.mask_target, self.weights = self._load_data_into_memory()

    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        with open(self.args.input_path + self.args.target_file, 'rb') as f:
            target = pickle.load(f)        
        with open(self.args.input_path + self.args.graph_file, 'rb') as f:
            graph = pickle.load(f)
        with open(self.args.input_path + self.args.mask_target_file, 'rb') as f:
            mask_target = pickle.load(f)
        with open(self.args.input_path + self.args.subgraphs_file, 'rb') as f:
            subgraphs = pickle.load(f)
        if self.args.weights_file is not None:
            with open(self.args.input_path + self.args.weights_file, 'rb') as f:
                weights = pickle.load(f)
        else:
            weights = None
        self.length = len(idx_to_key)
        self.low_res_abs = abs(graph.low_res)
        return input, idx_to_key, target, graph, subgraphs, mask_target, weights

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        input = torch.zeros((25, 5, 5, 6, 6))                               # (time, var, lev, lat, lon)
        input[:, :] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        #-- derive graphs and target
        subgraph = self.subgraphs[space_idx].clone()
        train_mask = self.mask_target[:,time_idx][subgraph.mask_1_cell]     # train_mask.shape = (self.graph.n_nodes, )
        subgraph["train_mask"] = train_mask
        y = self.target[subgraph.mask_1_cell, time_idx][train_mask]         # y.shape = (subgraph.n_nodes,)
        subgraph["y"] = y 
        if self.weights is not None:
            w = self.weights[subgraph.mask_1_cell, time_idx][train_mask]         # y.shape = (subgraph.n_nodes,)
            subgraph["w"] = w
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
        input = torch.zeros((25, 5, 5, 6, 6))                               # (time, var, lev, lat, lon)
        input[:, :] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        #-- derive graphs and target
        subgraph = self.subgraphs[space_idx].clone()
        subgraph["time_idx"] = time_idx - self.time_min
        y = self.test_graph.y[subgraph.mask_1_cell, time_idx - self.time_min]
        subgraph["y"] = y
        return input, subgraph


def custom_collate_fn_ae(batch):
    input = torch.stack(batch)
    input = default_convert(input)
    return input

def custom_collate_fn_gnn(batch):
    input = torch.stack([item[0] for item in batch])                        # shape = (batch_size, 25, 5, 5, 6, 6)
    data = [item[1] for item in batch]
    input = default_convert(input)
    return input, data
    
