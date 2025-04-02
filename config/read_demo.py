# read demo for config split file

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
    parser.add_argument('--group_trunc_ratio_hop_1', type=float, default=0.7, help='Hop 1 Group trunc ratio')
    parser.add_argument('--group_trunc_ratio_hop_2', type=float, default=0.9, help='Hop 2 Group trunc ratio.')
    parser.add_argument( '--verbose', type = bool, default = True)
    return parser.parse_args()


args = parse_args()
print(args)
dataset_name = args.dataset
num_hops = args.num_hops
seed = args.seed
num_perm = args.num_perm
group_trunc_ratio_hop_1 = args.group_trunc_ratio_hop_1
group_trunc_ratio_hop_2 = args.group_trunc_ratio_hop_2
verbose = args.verbose

np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Amazon(root='/home/chih3/tmp/' + 'Amazon', name=dataset_name, transform=T.NormalizeFeatures())
data = dataset[0].to(device)
with open(f'/home/chih3/data_valuation/config/Amazon-{dataset_name}.pkl', 'rb') as f:
    loaded_indices_dict = pickle.load(f) 
data = set_masks_from_indices(data, loaded_indices_dict, device)

#dataset = Coauthor(root='/home/chih3/tmp/Coauthor', name=dataset_name, transform=T.NormalizeFeatures())
#data = dataset[0].to(device)
#with open(f'/home/chih3/data_valuation/config/Coauthor-{dataset_name}.pkl', 'rb') as f:
#    loaded_indices_dict = pickle.load(f) 
#data = set_masks_from_indices(data, loaded_indices_dict, device)
