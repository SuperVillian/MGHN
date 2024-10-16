# Multi-Granularity Hypergraph Network (MGHN) for Node Classification

This repository contains the implementation of the Multi-Granularity Hypergraph Network (MGHN) model for node classification tasks on graph-structured data. The MGHN model leverages adaptive granular ball splitting and multi-column network structures to effectively capture the intricate characteristics of hyperedges across multiple levels.

## Experimental Setup

- **Hardware:**
  - Intel Xeon Gold 5218 CPU at 2.30 GHz
  - Four Tesla V100 GPUs

- **Software:**
  - Python 3.10.9
  - PyTorch 1.13
  - CUDA 11.6

- **Model Configuration:**
  - Number of Columns: 4
  - Hidden Dimension: 16
  - Number of Attention Heads: 8 (in the multi-head attention mechanism for each block)
  - Learning Rate: 0.01
  - Regularization Parameter: \(5 \times 10^{-4}\)

## Datasets

Node classification is evaluated using seven standard graph datasets that encompass various domains. These datasets are summarized in Table 2, where:
- \(N\) represents the number of nodes.
- \(M\) represents the number of edges.

| Dataset     | N     | M      |
| ----------- | ----- | ------ |
| Cora        | 2708  | 10858  |
| Pubmed      | 19717 | 88676  |
| Citeseer    | 3327  | 9464   |
| Facebook    | 22470 | 85501  |
| BlogCatalog | 5196  | 343486 |
| Flickr      | 7575  | 479476 |
| Github      | 37700 | 144501 |

## Getting Started

To run the MGHN model for node classification, follow these steps:

### Prerequisites

- Python 3.10.9
- PyTorch 1.13
- CUDA 11.6

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://anonymous.4open.science/r/MGHN.git
pip install -r requirements.txt
```

### Running the Model

Use the following command to run the MGHN model for node classification:

```
python -u ./train.py --coarsening_method <method> --data_name <dataset> --device <device_id> --seed <random_seed>
```