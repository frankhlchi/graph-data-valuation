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
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        model.eval()
        _, pred = model(input_data).max(dim=1)
        correct = float (pred[input_data.test_mask].eq(input_data.y[input_data.test_mask]).sum().item())
        acc = correct / input_data.test_mask.sum().item()
        print('Test Accuracy: {:.4f}'.format(acc))
        return acc

    def predict_valid(self, dataset):
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        model.eval()
        _, pred = model(input_data).max(dim=1)
        correct = float (pred[input_data.val_mask].eq(input_data.y[input_data.val_mask]).sum().item())
        acc = correct / input_data.val_mask.sum().item()
        print('Validation Accuracy: {:.6f}'.format(acc))
        return acc

    
def set_masks_from_indices(data, indices_dict, device):
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
    # Nodes to be considered
    nodes = mask.nonzero().view(-1)
    # Extract the edges for these nodes
    edge_mask_src = (data.edge_index[0].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask_dst = (data.edge_index[1].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask = edge_mask_src & edge_mask_dst
    sub_edge_index = data.edge_index[:, edge_mask]

    test_mask = data.test_mask
    val_mask = data.val_mask

    # Return a new data object with the subgraph data
    sub_data = Data(x=data.x, edge_index=sub_edge_index, y=data.y, test_mask=test_mask, val_mask=val_mask)
    return sub_data

# Values for each position

group_trunc_ratio_hop_1 = 0.5
group_trunc_ratio_hop_2 = 0.7
ratio = 3
directory = 'value/'
pattern = re.compile(r'^Cora_(\d+)_10_0_0\.5_0\.7_pc_value\.pkl$')

matching_files = []

for filename in os.listdir(directory):
    if pattern.match(filename):
        matching_files.append(filename)

    
filenames = matching_files[:ratio]

# extract the saved pc value into a dataframe
results = collections.defaultdict(list)
counter =0 
for filename in filenames:
    with open('value/' + filename, 'rb') as f:
        data = pickle.load(f)
    for key, sub_dict in data.items():
        for sub_key, sub_sub_dict in sub_dict.items():
            for sub_sub_key, value in sub_sub_dict.items():
                results[(key, sub_key, sub_sub_key)].append(value)
    counter += 1
    

for key, values in results.items():
    results[key] = sum(values) / (len(values)*10)

data = [{'key1': k1, 'key2': k2, 'key3': k3, 'value': v} for (k1, k2, k3), v in results.items()]
win_df = pd.DataFrame(data)

#aggregate hop-1 neighbors value (cannot be a labled node), for a node with different ranks, we label it as the highest rank
#aggregate hop-1 neighbors value when their position is hop-1 neighbors (key2)
win_df_11 = pd.DataFrame(win_df [win_df['key2'].isin(win_df['key1']) == False] .groupby('key2').\
                    value.sum().sort_values()).reset_index()
win_df_11.columns= ['key', 'value']
hop_1_list = win_df [win_df['key2'].isin(win_df['key1']) == False]['key2'].unique()
#aggregate hop-1 neighbors value when their position is hop-2 neighbors (key3)
win_df_12 = pd.DataFrame(win_df [(win_df['key3'] != win_df['key2'])&(win_df['key3'].isin(hop_1_list) )].\
                     groupby('key3').value.sum().sort_values()).reset_index()
win_df_12.columns= ['key', 'value']

win_df_1 =  pd.DataFrame(pd.concat([win_df_11, win_df_12]).groupby('key').value.sum().sort_values()).reset_index()
#aggregate hop-2 neighbors value (hop-2 neighbors vale can only come from hop-2)
win_df_2 = pd.DataFrame(win_df [(win_df['key3'].isin(win_df['key2']) == False)&\
        (win_df['key3'].isin(win_df['key1']) == False)] .groupby('key3').\
                    value.sum().sort_values()).reset_index()
win_df_2.columns= ['key', 'value']

unlabled_win_df = pd.concat([win_df_1,win_df_2])
unlabled_win_df = unlabled_win_df.sort_values('value',ascending= False)
unlabeled_win = torch.tensor(unlabled_win_df['key'].values)
unlabeled_win_value = unlabled_win_df['value'].values


dataset = Planetoid(root='dataset/', name='Cora', transform=T.NormalizeFeatures())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

train_mask = data.train_mask
val_mask = data.val_mask 
test_mask = data.test_mask 

# Printing the sizes
print(f"Training size: {train_mask.sum().item()}")
print(f"Validation size: {val_mask.sum().item()}")
print(f"Test size: {test_mask.sum().item()}")

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


test_data = get_subgraph_data(data, data.test_mask)
val_data = get_subgraph_data(data, data.val_mask)

win_acc = []
val_acc_list = []
node_list = unlabeled_win.cpu().numpy()
drop_num = len(node_list)+1

data_copy = data.clone()
data_copy = data_copy.to(device)
data_copy.edge_index = data_copy.edge_index[:,  indu_mask]

model = SGCNet(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
model.fit(data_copy)
test_acc = model.predict(test_data)
val_acc = model.predict_valid(val_data )
win_acc  +=[test_acc]
val_acc_list += [val_acc]

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
        
path = 'res/'
# Ensure the directory exists
os.makedirs(path, exist_ok=True)
# Open the file in write binary mode
with open(os.path.join(path, f'node_drop_large_winter_value_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_{counter}_cora_test.pkl'), 'wb') as file:
    # Dump the win_acc object into the file
    pickle.dump(win_acc, file)

with open(os.path.join(path, f'node_drop_large_winter_value_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_{counter}_cora_vali.pkl'), 'wb') as file:
    # Dump the win_acc object into the file
    pickle.dump(val_acc_list, file)


