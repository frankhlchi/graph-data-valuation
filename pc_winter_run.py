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
import torch_geometric.transforms as T
from torch.nn.functional import cosine_similarity
from torch_geometric.datasets import Amazon

from typing import Optional
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm

class SGConvNoWeight(MessagePassing):
    r"""The modified SGConv operator without the trainable linear layer."""


    def __init__(self, K: int = 2,
                 cached: bool = False, add_self_loops: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)

        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)


        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, K={self.K})')

                
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layers(x)
        return F.log_softmax(out, dim=1)

    def predict(self, x):
        output = self.forward(x)
        return output

    def compute_accuracy(self, output, y):
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        return correct / len(y)

    def fit(self, X, y, val_X, val_y, num_iter=200, verbosity=False, patience=20):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01,weight_decay=5e-4)
        best_val_accuracy = 0.0
        counter = 0

        for epoch in range(num_iter):
            self.train()
            optimizer.zero_grad()
            output = self(X)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
            
def adjacency_to_edge_list(adj_matrix):
    # Convert the adjacency matrix to COO format
    source, target = adj_matrix.nonzero().unbind(dim=1)
    
    # Create the edge_index tensor
    edge_index = torch.stack([source, target], dim=0)
    
    return edge_index


def stack_torch_tensors(input_tensors):
    
    unrolled = [input_tensors[k].view(-1,1) for k in range(len(input_tensors))]

    return torch.cat(unrolled)


def generate_features_and_labels_ind(cur_hop_1_list, cur_hop_2_list, cur_labeled_list, target_node, node_map, ind_edge_index, data, device):

    A_temp = torch.zeros((data.x.size(0), data.x.size(0)), device=device)
    A_temp[ind_edge_index[0], ind_edge_index[1]] = 1
    
    mask = torch.zeros_like(A_temp)
    cur_hop_1_list_torch = torch.tensor(cur_hop_1_list)
    mask[target_node, cur_hop_1_list] = 1
    mask[cur_hop_1_list, target_node] = 1
    if len(cur_hop_1_list) > 1:
        for hop_1_node in cur_hop_1_list[:-1]:
            hop_2_list = list(node_map[target_node][hop_1_node].keys())
            mask[hop_1_node, hop_2_list] = 1
            mask[hop_2_list, hop_1_node] = 1
        mask[cur_hop_1_list[-1], cur_hop_2_list] = 1
        mask[cur_hop_2_list, cur_hop_1_list[-1]] = 1
    else:
        mask[cur_hop_1_list[-1], cur_hop_2_list] = 1
        mask[cur_hop_2_list, cur_hop_1_list[-1]] = 1
    
    conv = SGConvNoWeight(K=2)
    cur_edge_index = adjacency_to_edge_list(mask)
    X_ind_propogated_temp_ = conv(data.x, cur_edge_index)
        
    train_features = torch.cat((X_ind_propogated[cur_labeled_list], X_ind_propogated_temp_[target_node].unsqueeze(0)), dim=0)
    train_labels = torch.cat((data.y[cur_labeled_list], data.y[target_node].unsqueeze(0)), dim=0)

    return train_features, train_labels



def evaluate_retrain_model(model_class, num_features, num_classes, train_features, train_labels, val_features, val_labels, device, num_iter=50):
    # Create and train the model
    model = model_class(num_features, num_classes).to(device)
    model.fit(train_features, train_labels,val_features, val_labels, verbosity=False)

    # Make predictions on the validation set
    predictions = model(val_features)

    # Calculate the accuracy of the model
    val_acc = (predictions.argmax(dim=1) == val_labels).float().mean().item()

    return val_acc


def generate_maps(train_idx_list, num_hops, edge_index):
    labeled_to_player_map = {}
    sample_value_dict = {}
    sample_counter_dict = {}
    for labeled in train_idx_list:
        hop_1_nodes, _,_,_ = k_hop_subgraph(int(labeled), num_hops=1, edge_index=edge_index, relabel_nodes=False)
        hop_1_nodes_list = list(hop_1_nodes.cpu().numpy())
        hop_1_nodes_list.remove(labeled) 
        labeled_to_player_map[labeled] = {}
        sample_value_dict[labeled] = {}
        sample_counter_dict[labeled] = {}
        labeled_to_player_map[labeled][labeled] = {}
        sample_value_dict[labeled][labeled] = {}
        sample_counter_dict[labeled][labeled] = {}
        
        for hop_1_node in hop_1_nodes_list:
            sub_nodes_2, _,_,_ = k_hop_subgraph(int(hop_1_node), num_hops=1, edge_index=edge_index, relabel_nodes=False)
            sub_nodes_2_list = list(sub_nodes_2.cpu().numpy())
            sub_nodes_2_list.remove(hop_1_node) 
            labeled_to_player_map[labeled][hop_1_node] = {}
            sample_value_dict[labeled][hop_1_node] = {}
            sample_counter_dict[labeled][hop_1_node] = {}
                
            for hop_2_node in sub_nodes_2_list:
                labeled_to_player_map[labeled][hop_1_node][hop_2_node] = [hop_2_node]
                sample_value_dict[labeled][hop_1_node][hop_2_node] = 0
                sample_counter_dict[labeled][hop_1_node][hop_2_node] = 0
            labeled_to_player_map[labeled][hop_1_node][hop_1_node] = [hop_1_node]
            sample_value_dict[labeled][hop_1_node][hop_1_node] = 0
            sample_counter_dict[labeled][hop_1_node][hop_1_node] = 0
            
        labeled_to_player_map[labeled][labeled][labeled] = [labeled]
        sample_value_dict[labeled][labeled][labeled] = 0
        sample_counter_dict[labeled][labeled][labeled] = 0

    return labeled_to_player_map, sample_value_dict, sample_counter_dict


def count_leaves(nested_dict):
    count = 0
    for key in nested_dict:
        if isinstance(nested_dict[key], dict):
            count += count_leaves(nested_dict[key])
        else:
            count += len(nested_dict[key])
    return count


def get_subgraph_data(data_edge_index, mask):
    # Nodes to be considered
    edge_index = data_edge_index.clone()
    nodes = mask.nonzero().view(-1)

    # Extract the edges for these nodes
    edge_mask_src = (edge_index[0].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask_dst = (edge_index[1].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask = edge_mask_src & edge_mask_dst

    sub_edge_index = edge_index[:, edge_mask]
    print ('sub_edge_index length:', len(sub_edge_index[0]))
    return sub_edge_index
    
    
def propagate_features(edge_index, node_features):
    """Double propagation of node features using the given edge_index."""
    A = torch.zeros((node_features.size(0), node_features.size(0)), device=device)
    A[edge_index[0], edge_index[1]] = 1
    A_hat = A + torch.eye(A.size(0), device=device)
    D_hat_diag = A_hat.sum(dim=1).pow(-0.5)
    D_hat = torch.diag(D_hat_diag)
    L_norm = D_hat.mm(A_hat).mm(D_hat)
    return L_norm.mm(L_norm.mm(node_features))

    
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

def parse_args():
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument('--dataset', default='computers', help='Input dataset.')
    parser.add_argument('--num_hops', type=int, default=2, help='Number of hops.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for permutation.')
    parser.add_argument('--num_perm', type=int, default=10, help='Number of permutations.')
    parser.add_argument('--label_trunc_ratio', type=float, default=0, help='Hop 1 Group trunc ratio')
    parser.add_argument('--group_trunc_ratio_hop_1', type=float, default=0.5, help='Hop 1 Group trunc ratio')
    parser.add_argument('--group_trunc_ratio_hop_2', type=float, default=0.7, help='Hop 2 Group trunc ratio.')
    parser.add_argument( '--verbose', type = bool, default = True)
    return parser.parse_args()



args = parse_args()
print(args)
dataset_name = args.dataset
num_hops = args.num_hops
seed = args.seed
num_perm = args.num_perm
label_trunc_ratio = args.label_trunc_ratio
group_trunc_ratio_hop_1 = args.group_trunc_ratio_hop_1
group_trunc_ratio_hop_2 = args.group_trunc_ratio_hop_2
verbose = args.verbose

np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='dataset/' + dataset_name, name=dataset_name, transform=T.NormalizeFeatures())

data = dataset[0].to(device)
num_classes = dataset.num_classes
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

if verbose:
    train_size = train_mask.sum().item()
    val_size = val_mask.sum().item()
    test_size = test_mask.sum().item()
    print(f"Train Size: {train_size}")
    print(f"Validation Size: {val_size}")
    print(f"Test Size: {test_size}")


X_full_propogated = propagate_features(data.edge_index, data.x)
val_edge_index =  get_subgraph_data(data.edge_index, val_mask)
X_val_propogated = propagate_features(val_edge_index , data.x)
test_edge_index =  get_subgraph_data(data.edge_index, test_mask)
X_test_propogated = propagate_features(test_edge_index, data.x)

val_features = X_val_propogated[val_mask]
val_labels = data.y[val_mask]
test_features = X_test_propogated[test_mask]
test_labels = data.y[test_mask]

inductive_edge_index = []
for src, tgt in data.edge_index.t().tolist():
    if not (val_mask[src] or test_mask[src] or val_mask[tgt] or test_mask[tgt]):
        inductive_edge_index.append([src, tgt])
inductive_edge_index = torch.tensor(inductive_edge_index).t().contiguous()
X_ind_propogated = propagate_features(inductive_edge_index, data.x)

if verbose:
    original_edge_count = data.edge_index.size(1)  
    inductive_edge_count = inductive_edge_index.size(1)  
    print(f"Original Edge Count: {original_edge_count}")
    print(f"Inductive Edge Count: {inductive_edge_count}")


train_idx = torch.nonzero(train_mask).cpu().numpy().flatten()
labeled_node_list = list(train_idx)
labeled_to_player_map, sample_value_dict, sample_counter_dict = \
        generate_maps( list(train_idx), num_hops, inductive_edge_index)

perf_dict = {
    'dataset': [],
    'seed': [],
    'perm': [],
    'label': [],
    'first_hop': [],
    'second_hop': [],
    'accu': []
}


total_time = 0
for i in range(num_perm):
    iteration_start_time = time.time()
    np.random.shuffle(labeled_node_list)
    cur_labeled_node_list = [] 
    pre_performance = 0
    
    trunc_label_len = int(np.ceil(len(labeled_node_list)  * (1-label_trunc_ratio)))
    for labeled_node in labeled_node_list[:trunc_label_len]:
        sample_value_dict_copy = copy.deepcopy(sample_value_dict)

        cur_hop_1_list = []
        hop_1_list = list(labeled_to_player_map [labeled_node].keys())
        np.random.shuffle(hop_1_list)
        hop_1_list.remove(labeled_node)
        hop_1_list.insert(0, labeled_node)
        print ('hop_1_list before truncation', len(hop_1_list), 'group_trunc_ratio_hop_1:', group_trunc_ratio_hop_1)
        truncate_length = int(np.ceil((len(hop_1_list) - 1) * (1- group_trunc_ratio_hop_1))) + 1
        truncate_length = min(truncate_length, len(hop_1_list))
        hop_1_list = hop_1_list[:truncate_length]
        print ('hop_1_list after truncation', len(hop_1_list) )
        
        print ('labeled_node iteration:', i)
        #print ('labeled_node permutation:', labeled_node_list)
        print ('current target labeled_node:',  cur_labeled_node_list, '=>', labeled_node)
        print ('hop_1_list ', hop_1_list )
        
        for player_hop_1 in hop_1_list:
            cur_hop_2_list = []
            cur_hop_1_list += [player_hop_1]
            hop_2_list = list(labeled_to_player_map [labeled_node][player_hop_1].keys())
            np.random.shuffle(hop_2_list)
            hop_2_list.remove(player_hop_1)
            hop_2_list.insert(0, player_hop_1)
            print ('hop_2_list before truncation', len(hop_2_list), 'group_trunc_ratio_hop_2:', group_trunc_ratio_hop_2)
            truncate_length = int(np.ceil((len(hop_2_list) - 1) * (1-group_trunc_ratio_hop_2))) + 1
            truncate_length = min(truncate_length, len(hop_2_list))
            hop_2_list = hop_2_list[:truncate_length]
            print ('hop_2_list after truncation', len(hop_2_list) )
            
            for player_hop_2 in hop_2_list:
                cur_hop_2_list += [player_hop_2]
                ind_train_features, ind_train_labels = \
                    generate_features_and_labels_ind(cur_hop_1_list, cur_hop_2_list, cur_labeled_node_list,\
                                            labeled_node, labeled_to_player_map, inductive_edge_index, data, device)
                val_acc = evaluate_retrain_model(MLP, dataset.num_features, dataset.num_classes, \
                                        ind_train_features, ind_train_labels, val_features, val_labels, device)
                sample_value_dict[labeled_node][player_hop_1][player_hop_2] += ( val_acc - pre_performance)
                sample_counter_dict [labeled_node][player_hop_1][player_hop_2] += 1
                pre_performance = val_acc
                
                perf_dict['dataset'] += [dataset_name]
                perf_dict['seed'] += [seed]
                perf_dict['perm'] += [i]
                perf_dict['label'] += [labeled_node]
                perf_dict['first_hop'] += [player_hop_1]
                perf_dict['second_hop'] += [player_hop_2]
                perf_dict['accu'] += [val_acc]         
                
                print('pre_performance ',pre_performance  ,'val_acc', val_acc)
        
        cur_labeled_node_list += [labeled_node]
        ind_train_features = X_ind_propogated[ cur_labeled_node_list]
        ind_train_labels = data.y[cur_labeled_node_list]
        val_acc = evaluate_retrain_model(MLP, dataset.num_features, dataset.num_classes, \
                                    ind_train_features, ind_train_labels, val_features, val_labels, device)
        pre_performance = val_acc
        print('full group acc',val_acc)
    
    
with open(f"value/{dataset_name}_{seed}_{num_perm}_{label_trunc_ratio}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_pc_value.pkl", "wb") as f:
    pickle.dump(sample_value_dict, f) 
with open(f"value/{dataset_name}_{seed}_{num_perm}_{label_trunc_ratio}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_pc_value_count.pkl", "wb") as f:
    pickle.dump( sample_counter_dict, f)
with open(f"value/{dataset_name}_{seed}_{num_perm}_{label_trunc_ratio}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_perf.pkl", "wb") as f:
    pickle.dump(perf_dict, f)
    