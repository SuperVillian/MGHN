import time
import os
import sys
import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from copy import deepcopy, copy

# Import custom modules from the dhg library
from dhg import Graph, Hypergraph
from dhg.data import *
from dhg.models import (
    HGNN, HGNNP, HyperGCN, HNHN, UniGCN, UniGAT,
    UniSAGE, UniGIN, DHCF, GCN, GIN, GraphSAGE
)
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from MGHG import generate_hypergraph_from_graph
from model import MGHN


def train(model, features, hypergraph, labels, train_indices, optimizer, epoch):
    model.train()
    start_time = time.time()
    optimizer.zero_grad()
    outputs = model(features, hypergraph)
    
    # Only take outputs for the training indices
    outputs, labels = outputs[train_indices], labels[train_indices]
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    train_accuracy = evaluator.validate(labels, outputs)
    print(f"Epoch: {epoch}, Time: {time.time() - start_time:.5f}s, Loss: {loss.item():.5f}, Train Accuracy: {train_accuracy:.5f}")
    torch.cuda.empty_cache()  # Clear CUDA memory
    return loss.item()


@torch.no_grad()
def infer(model, features, hypergraph, labels, indices, test=False):
    model.eval()
    outputs = model(features, hypergraph)
    outputs, labels = outputs[indices], labels[indices]
    
    return evaluator.test(labels, outputs) if test else evaluator.validate(labels, outputs)


def generate_masks(num_vertices, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Generate training, validation, and test masks.

    Parameters:
    - num_vertices (int): Number of vertices in the new hypergraph.
    - train_ratio (float): Ratio for the training set. Default is 0.6.
    - val_ratio (float): Ratio for the validation set. Default is 0.2.
    - test_ratio (float): Ratio for the test set. Default is 0.2.

    Returns:
    - tuple: Contains train_mask, val_mask, test_mask.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of ratios must be 1"
    assert num_vertices > 0, "Number of vertices must be greater than 0"

    indices = np.random.permutation(num_vertices)
    train_size = int(train_ratio * num_vertices)
    val_size = int(val_ratio * num_vertices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_mask = torch.zeros(num_vertices, dtype=torch.bool)
    val_mask = torch.zeros(num_vertices, dtype=torch.bool)
    test_mask = torch.zeros(num_vertices, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def read_data(data_name):
    """Load dataset based on the provided data name."""
    dataset_map = {
        "Cora": Cora,
        "Pubmed": Pubmed,
        "Citeseer": Citeseer,
        "Facebook": Facebook,
        "BlogCatalog": BlogCatalog,
        "Flickr": Flickr,
        "Github": Github,
    }
    
    if data_name in dataset_map:
        return dataset_map[data_name]()
    
    print("Error: Invalid data_name")
    sys.exit()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="Cora", help='name of dataset (default: Cora)')
    parser.add_argument('--device', type=int, default=0, help='which GPU to use if any (default: 0)')
    parser.add_argument('--coarsening_method', type=str, default="khop", help='coarsening_method')  
    parser.add_argument('--seed', type=int, default=0, help='random seed')                       
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    
    data = read_data(args.data_name)
    print(f"Loading dataset: {args.data_name} - {type(data).__name__}")

    # Create graph from data
    G = Graph(data["num_vertices"], data["edge_list"])
    
    try:
        features, labels = data['features'], data['labels']
        print("Dataset contains features")
        if features is None or len(features) == 0:
            raise ValueError("Features are empty or None.")
    except Exception:
        features, labels = torch.eye(data["num_vertices"]), data["labels"]
        print("Dataset does not contain features")

    # Coarsening method selection
    if args.coarsening_method == "khop":
        HG = Hypergraph.from_graph(G)
        print('Original graph vertices:', G.num_v)
        print('Original graph edges:', G.num_e)
        HG.add_hyperedges_from_graph_kHop(G, k=1)
        print('Hypergraph vertices:', HG.num_v)
        print('Hypergraph edges:', HG.num_e)
    elif args.coarsening_method == "MGHN":
        HG = generate_hypergraph_from_graph(G)
        print('Original graph vertices:', G.num_v)
        print('Original graph edges:', G.num_e)
        print('Multi-Granularity hypergraph vertices:', HG.num_v)
        print('Multi-Granularity hypergraph edges:', HG.num_e)

    # Generate masks with fixed ratios
    train_mask, val_mask, test_mask = generate_masks(G.num_v)
    print(f"Training set size: {train_mask.sum().item()}")
    print(f"Validation set size: {val_mask.sum().item()}")
    print(f"Test set size: {test_mask.sum().item()}")
    
    # Initialize model and optimizer
    model = MGHN(4, features.shape[1], 16, data["num_classes"], num_heads=8)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Move features and labels to device
    features, labels = features.to(device), labels.to(device)
    print(f"Feature matrix shape: {features.shape}") 
    print(f"Hypergraph num_v: {HG.num_v}, Feature matrix shape[0]: {features.shape[0]}")  # Ensure they match

    HG = HG.to(features.device)
    model = model.to(device)

    best_state = None
    best_epoch, best_val = 0, float('-inf')  # Initialize best validation value
    patience = 50  # Patience parameter for early stopping
    trigger_times = 0  # Count epochs with no improvement

    for epoch in range(400):
        # Training
        train(model, features, HG, labels, train_mask, optimizer, epoch)
        
        # Validation
        if epoch % 1 == 0:
            val_res = infer(model, features, HG, labels, val_mask)
            if val_res > best_val:
                print(f"Updated best validation result: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = copy.deepcopy(model.state_dict())
                trigger_times = 0  # Reset counter
            else:
                trigger_times += 1  # Increment counter
                print(f"No improvement: {trigger_times} epochs")
        
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

    print("\nTraining finished!")
    print(f"Best validation result: {best_val:.5f}")

    # Testing
    print("Testing...")
    if best_state is not None:
        model.load_state_dict(best_state)
        test_result = infer(model, features, HG, labels, test_mask, test=True)
        print(f"Final result at epoch {best_epoch}:")
        print(test_result)
    else:
        print("No valid model found. Testing the last epoch model.")
        model.load_state_dict(model.state_dict())
        test_result = infer(model, features, HG, labels, test_mask, test=True)
        print(test_result)
