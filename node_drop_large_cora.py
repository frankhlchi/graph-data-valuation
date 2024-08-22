"""
This script implements a dropping node experiment using the PC-Winter algorithm results.

It includes the following main components:
1. SGCNet: A SGC model used for downstream task evaluation
2. Data processing functions for graph data
3. PC-Winter value aggregation and node ranking
4. Node dropping experiment to evaluate the effectiveness of the valuation
"""

import pickle
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import accuracy_score
import random
from torch_geometric.transforms import RootedEgoNets
from torch_geometric.utils import k_hop_subgraph
import torch_geometric
import time
from itertools import chain, combinations
import numpy as np
import pickle
import pandas as pd
import argparse
import torch.nn.functional as F
import collections
import copy
from torch_geometric.nn import SGConv
import os 
import torch_geometric.transforms as T
import os
import re
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import itertools

class SGCNet(nn.Module):
    """
    Simple Graph Convolutional Network model
    """
    def __init__(self, num_features, num_classes, seed=0):
        super(SGCNet, self).__init__()
        torch.manual_seed(seed)  # Set the seed for CPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs
        self.conv = SGConv(num_features, num_classes, K=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_data):
        x, edge_index = input_data.x, input_data.edge_index
        x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)

    def fit(self, dataset, num_epochs=200):
        """Train the model"""
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = model(input_data)
            loss = F.nll_loss(out[input_data.train_mask], input_data.y[input_data.train_mask])
            loss.backward()
            optimizer.step()

    def predict(self, dataset):
        """Predict on test set and return accuracy"""
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        model.eval()
        _, pred = model(input_data).max(dim=1)
        correct = float (pred[input_data.test_mask].eq(input_data.y[input_data.test_mask]).sum().item())
        acc = correct / input_data.test_mask.sum().item()
        print('Test Accuracy: {:.4f}'.format(acc))
        return acc

    def predict_valid(self, dataset):
        """Predict on validation set and return accuracy"""
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        model.eval()
        _, pred = model(input_data).max(dim=1)
        correct = float (pred[input_data.val_mask].eq(input_data.y[input_data.val_mask]).sum().item())
        acc = correct / input_data.val_mask.sum().item()
        print('Validation Accuracy: {:.6f}'.format(acc))
        return acc

    
def set_masks_from_indices(data, indices_dict, device):
    """Set train, validation, and test masks for the data"""
    num_nodes = data.num_nodes
    
    train_mask = torch.zeros(num_nodes, dtype=bool).to(device)
    train_mask[indices_dict["train"]] = 1

    val_mask = torch.zeros(num_nodes, dtype=bool).to(device)
    val_mask[indices_dict["val"]] = 1

    test_mask = torch.zeros(num_nodes, dtype=bool).to(device)
    test_mask[indices_dict["test"]] = 1

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask

    return data


def get_subgraph_data(data, mask):
    """Extract subgraph data based on the given mask"""
    nodes = mask.nonzero().view(-1)
    edge_mask_src = (data.edge_index[0].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask_dst = (data.edge_index[1].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask = edge_mask_src & edge_mask_dst
    sub_edge_index = data.edge_index[:, edge_mask]

    test_mask = data.test_mask
    val_mask = data.val_mask

    sub_data = Data(x=data.x, edge_index=sub_edge_index, y=data.y, test_mask=test_mask, val_mask=val_mask)
    return sub_data

# Experimental parameters
group_trunc_ratio_hop_1 = 0.5
group_trunc_ratio_hop_2 = 0.7
ratio = 3
directory = 'value/'
pattern = re.compile(r'^Cora_(\d+)_10_0_0\.5_0\.7_pc_value\.pkl$')

# Find matching files for PC-Winter results
matching_files = []
for filename in os.listdir(directory):
    if pattern.match(filename):
        matching_files.append(filename)
filenames = matching_files[:ratio]

# Extract and aggregate PC-Winter values
results = collections.defaultdict(list)
counter = 0 
for filename in filenames:
    with open('value/' + filename, 'rb') as f:
        data = pickle.load(f)
    for key, sub_dict in data.items():
        for sub_key, sub_sub_dict in sub_dict.items():
            for sub_sub_key, value in sub_sub_dict.items():
                results[(key, sub_key, sub_sub_key)].append(value)
    counter += 1

# Average the values
for key, values in results.items():
    results[key] = sum(values) / (len(values)*10)

# Convert to DataFrame
data = [{'key1': k1, 'key2': k2, 'key3': k3, 'value': v} for (k1, k2, k3), v in results.items()]
win_df = pd.DataFrame(data)

# Aggregate values for different hop levels
win_df_11 = pd.DataFrame(win_df [win_df['key2'].isin(win_df['key1']) == False] .groupby('key2').value.sum().sort_values()).reset_index()
win_df_11.columns= ['key', 'value']
hop_1_list = win_df [win_df['key2'].isin(win_df['key1']) == False]['key2'].unique()
win_df_12 = pd.DataFrame(win_df [(win_df['key3'] != win_df['key2'])&(win_df['key3'].isin(hop_1_list) )].groupby('key3').value.sum().sort_values()).reset_index()
win_df_12.columns= ['key', 'value']

win_df_1 =  pd.DataFrame(pd.concat([win_df_11, win_df_12]).groupby('key').value.sum().sort_values()).reset_index()
win_df_2 = pd.DataFrame(win_df [(win_df['key3'].isin(win_df['key2']) == False)&(win_df['key3'].isin(win_df['key1']) == False)] .groupby('key3').value.sum().sort_values()).reset_index()
win_df_2.columns= ['key', 'value']

# Combine and sort unlabeled nodes
unlabled_win_df = pd.concat([win_df_1,win_df_2])
unlabled_win_df = unlabled_win_df.sort_values('value',ascending= False)
unlabeled_win = torch.tensor(unlabled_win_df['key'].values)
unlabeled_win_value = unlabled_win_df['value'].values

# Load and preprocess the Cora dataset
dataset = Planetoid(root='dataset/', name='Cora', transform=T.NormalizeFeatures())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

train_mask = data.train_mask
val_mask = data.val_mask 
test_mask = data.test_mask 

# Print dataset sizes
print(f"Training size: {train_mask.sum().item()}")
print(f"Validation size: {val_mask.sum().item()}")
print(f"Test size: {test_mask.sum().item()}")

# Create inductive edge index (removing edges to val/test nodes)
inductive_edge_index = []
for src, tgt in data.edge_index.t().tolist():
    if not (val_mask[src] or test_mask[src] or val_mask[tgt] or test_mask[tgt]):
        inductive_edge_index.append([src, tgt])
inductive_edge_index = torch.tensor(inductive_edge_index).t().contiguous()

indu_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
for i, (src, tgt) in enumerate(data.edge_index.t().tolist()):
    if val_mask[src] or test_mask[src] or val_mask[tgt] or test_mask[tgt]:
        indu_mask[i] = False
indu_mask  = indu_mask.to(device)

# Prepare test and validation data
test_data = get_subgraph_data(data, data.test_mask)
val_data = get_subgraph_data(data, data.val_mask)

# Node dropping experiment
win_acc = []
val_acc_list = []
node_list = unlabeled_win.cpu().numpy()
drop_num = len(node_list)+1

# Initial model training and evaluation
data_copy = data.clone()
data_copy = data_copy.to(device)
data_copy.edge_index = data_copy.edge_index[:,  indu_mask]

model = SGCNet(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
model.fit(data_copy)
test_acc = model.predict(test_data)
val_acc = model.predict_valid(val_data )
win_acc  +=[test_acc]
val_acc_list += [val_acc]

# Iteratively drop nodes and evaluate
for j in range(1,drop_num):
    cur_player = node_list[j-1]
    print('cur_player: ',cur_player)
    cur_node_list = node_list[:j]

    edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=device)
    for node in cur_node_list:
        edge_mask[data.edge_index[0] == node] = False
        edge_mask[data.edge_index[1] == node] = False

    edge_mask = edge_mask & indu_mask
    data_copy = data.clone()
    data_copy = data_copy.to(device)
    data_copy.edge_index = data_copy.edge_index[:, edge_mask]

    model = SGCNet(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
    model.fit(data_copy)
    test_acc = model.predict(test_data)
    val_acc = model.predict_valid(val_data )
    win_acc  +=[test_acc]
    val_acc_list += [val_acc]
        
# Save results
path = 'res/'
os.makedirs(path, exist_ok=True)
with open(os.path.join(path, f'node_drop_large_winter_value_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_{counter}_cora_test.pkl'), 'wb') as file:
    pickle.dump(win_acc, file)

with open(os.path.join(path, f'node_drop_large_winter_value_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_{counter}_cora_vali.pkl'), 'wb') as file:
    pickle.dump(val_acc_list, file)
