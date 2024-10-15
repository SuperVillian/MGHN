import torch
import torch.nn as nn
import networkx as nx
import math
from collections import deque
from dhg import Graph, Hypergraph

MAX_SPLIT_DEPTH = 2  # Define the maximum depth for splitting

def generate_hypergraph_from_graph(input_graph: Graph) -> Hypergraph:
    """
    Generate a hypergraph from the given graph input_graph.
    """
    # Create a networkx graph
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(input_graph.num_v))
    
    # Add weighted edges to the graph
    weighted_edges = [(u, v, {'weight': w}) for (u, v), w in zip(input_graph.e[0], input_graph.e[1])]
    nx_graph.add_edges_from(weighted_edges)

    if nx.is_connected(nx_graph):
        hyperedges = create_generalized_ball_graph(nx_graph)
    else:
        # Handle disconnected components
        connected_components = list(nx.connected_components(nx_graph))
        connected_subgraphs = [nx_graph.subgraph(component) for component in connected_components]
        print('Graph is not connected.')
        hyperedges = []
        
        for connected_subgraph in connected_subgraphs:
            hyperedge = create_generalized_ball_graph(connected_subgraph)
            hyperedges.extend(hyperedge)

    # Include original edges in the hyperedges
    hyperedges.extend(input_graph.e[0])
    
    return Hypergraph(input_graph.num_v, hyperedges)

def create_generalized_ball_graph(nx_graph: nx.Graph) -> list:
    """
    Obtain the generalized ball graph from the input graph.
    """
    initial_ball_count = math.isqrt(len(nx_graph))

    if initial_ball_count == 1:
        initial_balls = [nx_graph]
    else:
        initial_balls = initialize_generalized_ball_graph(nx_graph, initial_ball_count)

    generalized_balls = []
    for initial_ball in initial_balls:
        split_balls = []
        recursively_split_ball(initial_ball, split_balls, current_depth=0)  # Start splitting from depth 0
        generalized_balls.extend(split_balls)

    # Create hyperedges from the split graph list
    return [tuple(ball.nodes()) for ball in generalized_balls] if len(generalized_balls) > 1 else []

def initialize_generalized_ball_graph(nx_graph: nx.Graph, initial_ball_count: int) -> list:
    """
    Initialize the generalized ball graph based on node degrees.
    """
    # Get and sort nodes by degree
    degree_dict = dict(nx_graph.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)

    # Select center nodes
    center_nodes = sorted_nodes[:initial_ball_count]
    center_nodes_dict = assign_nodes_to_multiple_centers(nx_graph, center_nodes)

    # Build initial subgraph list
    return [nx.subgraph(nx_graph, cluster) for cluster in center_nodes_dict.values()]

def assign_nodes_to_multiple_centers(G: nx.Graph, centers: list) -> dict:
    """
    Assign nodes to multiple centers using a multi-source BFS approach.
    """
    center_nodes_dict = {center: set() for center in centers}
    center_queues = {center: deque([center]) for center in centers}
    visited_nodes = {center: center for center in centers}

    while any(center_queues.values()):
        for center in centers:
            if center_queues[center]:
                current_node = center_queues[center].popleft()
                center_nodes_dict[center].add(current_node)

                for neighbor in G.neighbors(current_node):
                    if neighbor not in visited_nodes:
                        visited_nodes[neighbor] = center
                        center_queues[center].append(neighbor)

    return center_nodes_dict

def recursively_split_ball(nx_graph: nx.Graph, split_balls: list, current_depth: int):
    """
    Recursively split the graph into smaller subgraphs.
    """
    if len(nx_graph) == 1:
        return

    degree_dict = dict(nx_graph.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
    center_nodes = sorted_nodes[:2]
    center_nodes_dict = assign_nodes_to_multiple_centers(nx_graph, center_nodes)
    clusters = [cluster for cluster in center_nodes_dict.values()]
    
    cluster_a, cluster_b = clusters[0], clusters[1]
    subgraph_a, subgraph_b = nx.subgraph(nx_graph, cluster_a), nx.subgraph(nx_graph, cluster_b)

    # Prevent splitting if there are no edges in either subgraph
    if len(subgraph_a.edges()) == 0 or len(subgraph_b.edges()) == 0:
        if current_depth >= MAX_SPLIT_DEPTH - 1:
            split_balls.append(nx_graph)
    else:
        # Calculate average degree
        avg_degree = nx_graph.number_of_edges() / len(nx_graph)
        avg_degree_a = subgraph_a.number_of_edges() / len(subgraph_a)
        avg_degree_b = subgraph_b.number_of_edges() / len(subgraph_b)

        # Check for splitting condition and record hyperedges at the penultimate depth
        if avg_degree < avg_degree_a + avg_degree_b and current_depth < MAX_SPLIT_DEPTH - 1:
            recursively_split_ball(subgraph_a, split_balls, current_depth + 1)
            recursively_split_ball(subgraph_b, split_balls, current_depth + 1)
        else:
            if current_depth >= MAX_SPLIT_DEPTH - 1:
                split_balls.append(nx_graph)
