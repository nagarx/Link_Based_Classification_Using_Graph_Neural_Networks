# Link_Based_Classification_Using_Graph_Neural_Networks

## Project Description:
Graph neural networks (GNNs) are a class of deep learning models that capture intrinsic data patterns to facilitate model training. They are finding extensive applications in social and ego networks, molecular discovery, and other domains where the data has an underlying graph structure.

In this project, we will develop a graph convolutional network (GCN) to classify the scientific publications in the Cora dataset. As the Cora dataset consists of interlinked data, using GNNs will allow us to capture more data correlations as compared to conventional neural networks for improved model performance. We will import the Cora dataset, implement the graph convolutional network, and use it to classify the scientific publications in the Cora dataset. Moreover, we will analyze the model performance for different split ratios of the dataset.

## Technologies:
- PyG
- PyTorch
- Python
- Matplotlib


# Link-Based Classification Using Graph Neural Networks
[![Python](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9-red)](https://pytorch.org/)

## Introduction

This repository contains an implementation of link-based classification using Graph Neural Networks (GNNs), specifically focusing on Graph Convolutional Networks (GCN). The project is built using PyTorch and PyTorch Geometric and is aimed at classifying nodes in a citation network (Cora dataset).

## Objective

The primary objective of this project is to leverage the power of Graph Neural Networks for semi-supervised node classification tasks. We focus on the Cora dataset, a citation network where each node represents a research paper and edges indicate citations between papers. The model aims to classify each paper into one of several predefined categories based on its features and its connections to other papers in the network.

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