# Graph Data Valuation

This repository contains code for performing data valuation on graph-structured data using the Precedence-Constrained Winter (PC-Winter) value method. The key files are:

- `pc_winter_run.py`: Main script to run PC-Winter and calculate PC values for graph elements.
- `node_drop_large_cora.py`: Script to aggregate node values calculated by PC-Winter and evaluate performance by sequentially dropping high-value nodes. 
- `plot.ipynb`: Notebook to visualize the drop in model performance as high-value nodes are removed.
- `value/`: Directory to store the sampled PC values.
- `res/`: Directory to store the model performance results from the node dropping experiments.

## Running PC-Winter
To calculate PC values for graph elements, run:
python pc_winter_run.py --dataset <dataset> --seed <seed>
This will sample PC values and store them in the value/ directory.

## Node Dropping Experiment
To evaluate the effectiveness of the PC values, run:
python node_drop_large_cora.py
This will aggregate the node values, drop high-value nodes, retrain the model, and save the performance results in res/.

## Visualizing Results
Use the plot.ipynb notebook to load the performance results from res/ and plot the drop in model accuracy as high-value nodes are removed. A steeper drop indicates the PC values more effectively identify influential nodes.
