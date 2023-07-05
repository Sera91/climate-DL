import numpy as np
import torch
from torch import nn
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Batch
import sys
import time
import copy

class Autoencoder(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2):
        super().__init__() 
        self.cnn_output_dim = cnn_output_dim
        self.gru_hidden_dim = gru_hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(cnn_output_dim, gru_hidden_dim, n_layers, batch_first=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.Unflatten(-1,(256, 2, 2, 2)),
            nn.Upsample(size=(3,4,4)),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.ReLU(),
            nn.Upsample(size=(5,6,6)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 5, kernel_size=3, padding=(1,1,1), stride=1),
            )

    def forward(self, X):
        s = X.shape
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])            # (batch_dim*25, 5, 5, 6, 6)
        X = self.encoder(X)                                         # (batch_dim*25, cnn_output_dim)
        X = X.reshape(s[0], s[1], self.cnn_output_dim)              # (batch_dim, 25, cnn_output_dim)
        out, _ = self.gru(X) # out, h                               # (batch_dim, 25, gru_hidden_dim
        out = out.reshape(s[0]*s[1], self.gru_hidden_dim)           # (batch_dim*25, gru_hidden_dim)
        out = self.decoder(out)                                     # (batch_dim*25, gru_hidden_dim)
        out = out.reshape(s[0], s[1], s[2], s[3], s[4], s[5])       # (batch_dim, 25, 5, 5, 6, 6)
        return out


#class Classifier(nn.Module):
#    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, num_node_features=1):
#        super().__init__()
#        self.cnn_output_dim = cnn_output_dim
#        self.gru_hidden_dim = gru_hidden_dim
#        self.gnn_node_dim = gru_hidden_dim * 25 + num_node_features
#
#        self.encoder = nn.Sequential(
#            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
#            nn.BatchNorm3d(64),
#            nn.ReLU(),
#            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
#            nn.BatchNorm3d(64),
#            nn.ReLU(),
#            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
#            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
#            nn.BatchNorm3d(256),
#            nn.ReLU(),
#            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
#            nn.Flatten(),
#            nn.Linear(2048, 512),
#            nn.BatchNorm1d(512),
#            nn.ReLU(),
#            nn.Linear(512, cnn_output_dim),
#            nn.BatchNorm1d(cnn_output_dim),
#            nn.ReLU()
#            )
#
#        # define the decoder modules
#        self.gru = nn.Sequential(
#            nn.GRU(cnn_output_dim, gru_hidden_dim, n_layers, batch_first=True),
#        )
#
#        #gnn
#        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
#            (geometric_nn.BatchNorm(self.gnn_node_dim), 'x -> x'),
#            (GATv2Conv(self.gnn_node_dim, self.gnn_node_dim, aggr='mean', dropout=0.6, edge_dim=1), 'x, edge_index, edge_attr -> x'), 
#            (geometric_nn.BatchNorm(self.gnn_node_dim), 'x -> x'),
#            nn.ReLU(),                                                     
#            (GATv2Conv(self.gnn_node_dim, self.gnn_node_dim, aggr='mean', dropout=0.6, edge_dim=1), 'x, edge_index, edge_attr -> x'),
#            (geometric_nn.BatchNorm(self.gnn_node_dim), 'x -> x'),
#            nn.ReLU(),
#            #(GATv2Conv(self.gnn_node_dim, 2, aggr='mean'), 'x, edge_index -> x'), # weighted cross entropy
#            #nn.Softmax(dim=-1)                                      # weighted cross entropy
#            (GATv2Conv(self.gnn_node_dim, 1, aggr='mean', edge_dim=1), 'x, edge_index, edge_attr -> x'), # focal loss
#            nn.Sigmoid()                                            # focal loss
#            ])
#        
#    def forward(self, X_batch, data_list):
#        s = X_batch.shape
#        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
#        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
#        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
#        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
#        encoding = encoding.reshape(s[0], s[1]*self.gru_hidden_dim)         # (batch_dim, 25*gru_hidden_dim)
#
#        for i, data in enumerate(data_list):
#            data['x'] = torch.zeros((data.num_nodes, 1 + encoding.shape[1])).cuda()
#            data['x'][:,1] = data.z.squeeze()
#            data['x'][:,1:] = encoding[i,:]
#
#        data_batch = Batch.from_data_list(data_list)                                                    
#        
#        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
#        train_mask = data_batch.train_mask
#
#        return y_pred[train_mask].squeeze(), data_batch.y[train_mask]       # focal loss       
#        #return y_pred[train_mask].squeeze(), data_batch.y[train_mask].squeeze().to(torch.long)  # weighted cross entropy loss
#
#class Regressor(nn.Module):
#    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, num_node_features=3):
#        super().__init__()
#        self.cnn_output_dim = cnn_output_dim
#        self.gru_hidden_dim = gru_hidden_dim
#        self.gnn_node_dim = gru_hidden_dim * 25 + num_node_features
#
#        self.encoder = nn.Sequential(
#            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
#            nn.BatchNorm3d(64),
#            nn.ReLU(),
#            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
#            nn.BatchNorm3d(64),
#            nn.ReLU(),
#            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
#            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
#            nn.BatchNorm3d(256),
#            nn.ReLU(),
#            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
#            nn.Flatten(),
#            nn.Linear(2048, 512),
#            nn.BatchNorm1d(512),
#            nn.ReLU(),
#            nn.Linear(512, cnn_output_dim),
#            nn.BatchNorm1d(cnn_output_dim),
#            nn.ReLU()
#            )
#
#        self.gru = nn.Sequential(
#            nn.GRU(cnn_output_dim, gru_hidden_dim, n_layers, batch_first=True),
#        )
#
#        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
#            (geometric_nn.BatchNorm(self.gnn_node_dim), 'x -> x'),
#            (GATv2Conv(self.gnn_node_dim, 128, heads=2, aggr='mean', dropout=0.6),  'x, edge_index, edge_attr -> x'), 
#            (geometric_nn.BatchNorm(256), 'x -> x'),
#            nn.ReLU(),                                                     
#            (GATv2Conv(256, 128, aggr='mean', dropout=0.6), 'x, edge_index, edge_attr-> x'),
#            (geometric_nn.BatchNorm(128), 'x -> x'),
#            nn.ReLU(),
#            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index, edge_attr -> x'),
#            ])
#        
#    def forward(self, X_batch, data_list):
#        s = X_batch.shape
#        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
#        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
#        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
#        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
#        encoding = encoding.reshape(s[0], s[1]*self.gru_hidden_dim)         # (batch_dim, 25*gru_hidden_dim)
#        for i, data in enumerate(data_list):
#            data['x'] = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).cuda()
#            data['x'][:,:3] = data.z.squeeze()
#            data['x'][:,3:] = encoding[i,:]       
#
#        data_batch = Batch.from_data_list(data_list, exclude_keys=["z", "low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped"]) 
#        
#        y_pred = self.gnn(data_batch.x, data_batch.edge_index)    
#        #y_pred = self.linear(y_pred)
#        train_mask = data_batch.train_mask
#        return y_pred[train_mask].squeeze(), data_batch.y[train_mask]       # focal loss
#    #return y_pred, data_batch.y.squeeze().to(torch.long), data_batch.batch  # weighted cross entropy loss


class Classifier_old(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, num_node_features=3, input_dim=256, hidden_dim=256):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            )
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(3+512), 'x -> x'),
            (GATv2Conv(3+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            nn.Sigmoid()
            ])

    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_batch, exclude_keys=["low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index) 
        train_mask = data_batch.train_mask
        return y_pred.squeeze()[train_mask], data_batch.y.squeeze()     


class Regressor_old(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, num_node_features=3, input_dim=256, hidden_dim=256):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            )
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(3+512), 'x -> x'),
            (GATv2Conv(3+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            ])

    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_batch, exclude_keys=["low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        train_mask = data_batch.train_mask
        return y_pred.squeeze()[train_mask], data_batch.y.squeeze(), data_batch.w.squeeze()

#----------------------------------------------
#-------- no lat lon in node features ---------
#----------------------------------------------

class Classifier_z_only(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, input_dim=256, hidden_dim=256, node_dim=1):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.node_dim = node_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            )
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+512), 'x -> x'),
            (GATv2Conv(node_dim+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            nn.Sigmoid()
            ])
    
    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_batch, exclude_keys=["low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index) 
        train_mask = data_batch.train_mask
        return y_pred.squeeze()[train_mask], data_batch.y.squeeze()     


class Regressor_z_only(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, input_dim=256, hidden_dim=256, node_dim=1):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.node_dim = node_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            )
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+512), 'x -> x'),
            (GATv2Conv(node_dim+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            ])

    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_batch, exclude_keys=["low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        train_mask = data_batch.train_mask
        return y_pred.squeeze()[train_mask], data_batch.y.squeeze(), data_batch.w.squeeze()


#----------------------------------------------
#-------------- edge attributes ---------------
#----------------------------------------------

class Classifier_edges(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, input_dim=256, hidden_dim=256, node_dim=1, edge_attr_dim=1):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.node_dim = node_dim
        self.edge_attr_dim = edge_attr_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            )
        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(node_dim+512), 'x -> x'),
            (GATv2Conv(node_dim+512, 128, heads=2, aggr='mean', dropout=0.5, edge_dim=edge_attr_dim),  'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            nn.Sigmoid()
            ])
    
    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_batch, exclude_keys=["low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float()) 
        train_mask = data_batch.train_mask
        return y_pred.squeeze()[train_mask], data_batch.y.squeeze()     


class Regressor_edges(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, num_node_features=3, input_dim=256, hidden_dim=256, node_dim=1, edge_attr_dim=1):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.node_dim = node_dim
        self.edge_attr_dim = edge_attr_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            )
        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(node_dim+512), 'x -> x'),
            (GATv2Conv(node_dim+512, 128, heads=2, aggr='mean', dropout=0.5, edge_dim=edge_attr_dim),  'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            ])

    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_batch, exclude_keys=["low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        train_mask = data_batch.train_mask
        return y_pred.squeeze()[train_mask], data_batch.y.squeeze(), data_batch.w.squeeze()


#----------------------------------------------
#---------------- test models -----------------
#----------------------------------------------

#class Classifier_test(Classifier):
#
#    def __init__(self):
#        super().__init__()
#    
#    def forward(self, X_batch, data_list, G_test):
#        s = X_batch.shape
#        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])   # (batch_dim*9*25, 5, 5, 6, 6)
#        X_batch = self.encoder(X_batch)                                     # (batch_dim*9*25, cnn_output_dim)
#        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.cnn_output_dim)     # (batch_dim*9, 25, cnn_output_dim)
#        encoding, _ = self.gru(X_batch)                                     # (batch_dim*9, 25, gru_hidden_dim)
#        encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)   # (batch_dim, 9, 25*gru_hidden_dim)
#
#        for i, data in enumerate(data_list):
#            data['x'] = torch.cat((data.z, encoding[i,data.idx_list_mapped,:]),dim=-1)
#        
#        data_batch = Batch.from_data_list(data_list, exclude_keys=["z", "low_res", "idx_list", "idx_list_mapped"]) 
#        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
#        
#        data_list = data_batch.to_data_list()        
#        
#        for i, data in enumerate(data_list):
#            y_pred_i = data.x[data.test_mask].squeeze()
#            G_test['pr_cl'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i > 0.5, 1.0, 0.0).cpu()
#
#        return
#    #return y_pred, data_batch.y.squeeze().to(torch.long), data_batch.batch  # weighted cross entropy loss
#
#class Regressor_test(Regressor):
#    
#    def __init__(self):
#        super().__init__()
#    
#    def forward(self, X, data_list, G_test):
#        s = X.shape
#        X = X.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])                   # (time_dim*9*25, 5, 5, 6, 6)
#        X = self.encoder(X)                                                     # (time_dim*9*25, cnn_output_dim)
#        X = X.reshape(s[0]*s[1], s[2], self.cnn_output_dim)                     # (time_dim*9, 25, cnn_output_dim)
#        encoding, _ = self.gru(X)                                               # (time_dim*9, 25, gru_hidden_dim)
#        encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)       # (time, 9, 25*gru_hidden_dim)
#
#        for i, data in enumerate(data_list):
#            data['x'] = torch.cat((data.z, encoding[i,data.idx_list_mapped,:]),dim=-1)
#        
#        data_batch = Batch.from_data_list(data_list, exclude_keys=["z", "low_res", "idx_list", "idx_list_mapped"]) 
#        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
#        data_batch.x = torch.expm1(data_batch.x)
#
#        data_list = data_batch.to_data_list()        
#        
#        for i, data in enumerate(data_list):
#            y_pred_i = data.x[data.test_mask].squeeze().cpu()
#            G_test['pr_reg'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i >= 0.1, y_pred_i, torch.tensor(0.0, dtype=y_pred_i.dtype))
#        return     

## old

class Classifier_old_test(Classifier_old):

    def __init__(self):
        super().__init__()
    
    def forward(self, X, data_list, G_test, device):
        s = X.shape
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])                        # (batch_dim*9*25, 5, 5, 6, 6)
        X = self.encoder(X)                                                     # (batch_dim*9*25, cnn_output_dim)
        X = X.reshape(s[0], s[1], self.cnn_output_dim)                          # (batch_dim*9, 25, cnn_output_dim)
        encoding, _ = self.gru(X)                                               # (batch_dim*9, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)             # (batch_dim, 9, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_list):
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_list, exclude_keys=["low_res", "mask_subgraph", "idx_list", "idx_list_mapped"]).to(device) 
        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index)
        
        data_list = data_batch.to_data_list()       

        for data in data_list:
            y_pred_i = data.x.squeeze()
            G_test['pr_cl'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i > 0.5, 1.0, 0.0).cpu()        
        return

class Regressor_old_test(Regressor_old):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, X, data_list, G_test, device):
        s = X.shape
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])                        # (time_dim*25, 5, 5, 6, 6)
        X = self.encoder(X)                                                     # (time_dim*25, cnn_output_dim)
        X = X.reshape(s[0], s[1], self.cnn_output_dim)                          # (time_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X)                                               # (time_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)             # (time_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding) 

        for i, data in enumerate(data_list):
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        
        data_batch = Batch.from_data_list(data_list, exclude_keys=["low_res", "mask_subgraph", "idx_list", "idx_list_mapped"]).to(device) 
        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index)
        data_batch.x = torch.expm1(data_batch.x)

        data_list = data_batch.to_data_list()        
        
        for data in data_list:
            y_pred_i = data.x.squeeze().cpu()
            G_test['pr_reg'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i >= 0.1, y_pred_i, torch.tensor(0.0, dtype=y_pred_i.dtype)) 
        return

## z only

class Classifier_z_only_test(Classifier_z_only):

    def __init__(self):
        super().__init__()

    def forward(self, X_batch, data_list, G_test, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_list):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_list, exclude_keys=["low_res", "mask_subgraph", "idx_list", "idx_list_mapped"]).to(device) 
        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index)
        
        data_list = data_batch.to_data_list()       

        for data in data_list:
            y_pred_i = data.x.squeeze()
            G_test['pr_cl'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i > 0.5, 1.0, 0.0).cpu()        
        return

class Regressor_z_only_test(Regressor_z_only):

    def __init__(self):
        super().__init__()
    
    def forward(self, X_batch, data_list, G_test, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_list):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_list, exclude_keys=["low_res", "mask_subgraph", "idx_list", "idx_list_mapped"]).to(device) 
        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index)
        data_batch.x = torch.expm1(data_batch.x)

        data_list = data_batch.to_data_list()        
        
        for data in data_list:
            y_pred_i = data.x.squeeze().cpu()
            G_test['pr_reg'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i >= 0.1, y_pred_i, torch.tensor(0.0, dtype=y_pred_i.dtype)) 
        return



## edges

class Classifier_edges_test(Classifier_edges):

    def __init__(self):
        super().__init__()

    def forward(self, X_batch, data_list, G_test, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_list):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_list, exclude_keys=["low_res", "mask_subgraph", "idx_list", "idx_list_mapped"]).to(device) 
        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        
        data_list = data_batch.to_data_list()       

        for data in data_list:
            y_pred_i = data.x.squeeze()
            G_test['pr_cl'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i > 0.5, 1.0, 0.0).cpu()        
        return

class Regressor_edges_test(Regressor_edges):

    def __init__(self):
        super().__init__()
    
    def forward(self, X_batch, data_list, G_test, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_list):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_list, exclude_keys=["low_res", "mask_subgraph", "idx_list", "idx_list_mapped"]).to(device) 
        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        data_batch.x = torch.expm1(data_batch.x)

        data_list = data_batch.to_data_list()        
        
        for data in data_list:
            y_pred_i = data.x.squeeze().cpu()
            G_test['pr_reg'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i >= 0.1, y_pred_i, torch.tensor(0.0, dtype=y_pred_i.dtype)) 
        return


if __name__ =='__main__':

    model = Regressor()
    batch_dim = 64
    input_batch = torch.rand((batch_dim, 25, 5, 5, 6, 6))

    start = time.time()
    X = model(input_batch)
    print(f"{time.time()-start} s\n")
    print(X.shape) 
