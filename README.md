# Link-Based Classification Using Graph Neural Networks
[![Python](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9-red)](https://pytorch.org/)

## Introduction

This repository is dedicated to the implementation and analysis of Graph Neural Networks (GNNs), particularly focusing on Graph Convolutional Networks (GCN) for the task of link-based classification. The project employs PyTorch and PyTorch Geometric to build and evaluate the model on the Cora dataset, which consists of scientific publications in a citation network.

### Background and Objective

Graph Neural Networks have emerged as a powerful paradigm for learning on graph-structured data, capturing intrinsic data patterns that are otherwise difficult to model. They have found extensive applications in various domains such as social networks, molecular discovery, and recommendation systems. The primary objective of this project is to leverage the strengths of GNNs for the semi-supervised classification of nodes in the Cora dataset. The dataset is composed of scientific papers linked by citations, making it an ideal candidate for demonstrating the efficacy of GNNs in capturing complex relationships between data points.

### Methodology and Performance Analysis

We implement a Graph Convolutional Network to classify these papers into predefined categories. Unlike conventional neural networks, GNNs can capture more sophisticated correlations in the dataset due to their ability to incorporate neighborhood information. The project includes a detailed analysis of model performance across different data splits to assess its robustness and generalizability.

## Theoretical Background

### Graph Neural Networks (GNNs)

Graph Neural Networks extend neural networks to graph-structured data. They operate on nodes and edges of a graph, enabling the model to capture the inherent relationships between connected nodes.

### Graph Convolutional Networks (GCN)

A Graph Convolutional Network (GCN) is a type of GNN particularly suited for node classification tasks. It leverages the graph structure to aggregate information from neighboring nodes to update the features of a given node.

#### Mathematical Formulation

A single layer of GCN can be mathematically represented as:

\[
H^{(l+1)} = \sigma\left( D^{-\frac{1}{2}} A D^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)
\]

Where \( H^{(l)} \) is the node feature matrix at layer \( l \), \( A \) is the adjacency matrix, \( D \) is the degree matrix, \( W^{(l)} \) is the weight matrix for layer \( l \), and \( \sigma \) is the activation function.

#### Loss Function and Optimization

We use the Cross-Entropy loss for the node classification task and optimize the model parameters using the Adam optimizer.

## Installation and Dependencies

To set up this project on your local machine, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/Link-Based-Classification-GNN.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Link-Based-Classification-GNN
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Dependencies

- Python 3.8+
- PyTorch 1.9
- PyTorch Geometric
- Matplotlib
- NumPy

## Results

The project successfully implements a Graph Convolutional Network (GCN) for semi-supervised node classification on the Cora dataset. Key metrics are as follows:

- Training Loss: Converges after X epochs.
- Validation Loss: Minimal overfitting observed.
- Test Accuracy: Achieved a peak test accuracy of X%.

For a detailed view of the evaluation metrics and plots, refer to the Jupyter notebook (`gnn.ipynb`).