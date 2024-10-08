# Graph Data Valuation with PC-Winter Value

This repository contains code for performing data valuation on graph-structured data using the Precedence-Constrained Winter (PC-Winter) value method. The PC-Winter algorithm is designed to identify influential graph elements in graph datasets by considering the hierarchical structure of node contributions.

---

**The image below illustrates our PC-Winter value estimation for graph data valuation.**

1. **Input Graphs (Inductive Setting)**: This includes three mutually exclusive graph structures:
   - Training Graph: Contains the elements to be evaluated. In this graph:
     - Nodes labeled as $$l$$ ($$l_0$$, $$l_1$$) represent labeled nodes
     - Nodes labeled as $$w$$ ($$w_0$$ - $$w_3$$) represent 1-distance neighbors of labeled nodes
     - Nodes labeled as $$u$$ ($$u_0$$ - $$u_6$$) represent 2-distance neighbors of labeled nodes
   - Validation Graph: Used for calculating accuracy as a utility function
   - Testing Graph: Reserved for downstream tasks such as node dropping

2. **PC-Winter Value Estimation Steps**:
   1. Perform a Preorder Traversal with hierarchical truncations on the Computational Tree $$\mathcal{T}$$
   2. Retrieve a permissive Permutation $$\pi$$
   3. Calculate the marginal contribution of each player based on validation accuracy difference
   4. Estimate PC-Winter Value by averaging marginal contributions of each player over multiple permutations

![Graph Data Valuation Framework](https://github.com/frankhlchi/graph-data-valuation/blob/main/framework_pc.png)

---

## Key Files

- `pc_winter_run.py`: Main script to run PC-Winter and calculate PC values for graph elements. This script implements an online version of the framework shown in the image, including local propagation, preorder traversal of the contribution tree, and hierarchical truncation.

- `node_drop_large_cora.py`: Script to aggregate node values calculated by PC-Winter and evaluate performance by sequentially dropping high-value nodes. This script demonstrates the effectiveness of the calculated PC values.

- `plot.ipynb`: Jupyter notebook to visualize the drop in model performance as high-value nodes are removed. This notebook helps in analyzing the results of the node dropping experiments.

- `value/`: Directory to store the sampled PC values output by `pc_winter_run.py`.

- `res/`: Directory to store the model performance results from the node dropping experiments conducted by `node_drop_large_cora.py`.

## Running PC-Winter

To calculate PC values for graph elements, use the following command:

```
python pc_winter_run.py --dataset <dataset> --seed <seed> --num_perm <num_permutations> --group_trunc_ratio_hop_1 <ratio1> --group_trunc_ratio_hop_2 <ratio2>
```

Parameters:
- `<dataset>`: Name of the dataset (e.g., 'Cora', 'Citeseer')
- `<seed>`: Random seed for reproducibility
- `<num_permutations>`: Number of permutations for PC-Winter algorithm
- `<ratio1>`: Truncation ratio for 1-hop neighbors
- `<ratio2>`: Truncation ratio for 2-hop neighbors

This will sample PC values and store them in the `value/` directory. The output files will be named according to the input parameters.

## Node Dropping Experiment

To evaluate the effectiveness of the PC values, run:

```
python node_drop_large_cora.py
```

This script performs the following steps:
1. Loads and aggregates the PC values calculated by `pc_winter_run.py`
2. Ranks nodes based on their aggregated PC values
3. Sequentially drops high-value nodes from the graph
4. Retrains a Simple Graph Convolution (SGC) model after each node removal
5. Saves the model performance results in the `res/` directory

The output files in `res/` will contain the test and validation accuracies at each step of the node dropping process.

## Visualizing Results

Use the `plot.ipynb` notebook to visualize the results of the node dropping experiment:

1. Open the notebook in Jupyter:
   ```
   jupyter notebook plot.ipynb
   ```
2. Run the cells to load the performance results from the `res/` directory
3. The notebook will generate plots showing the drop in model accuracy as high-value nodes are removed

A steeper drop in the accuracy curve indicates that the PC values more effectively identify influential nodes in the graph. This visualization helps in assessing the quality of the data valuation performed by the PC-Winter method.
