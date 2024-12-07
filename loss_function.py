from itertools import product
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from config import Config
import torch.sparse as sparse
import networkx as nx

device = Config().device()

# Calculate B matrix
def get_B(G):
    Q = 0
    G = G.copy()
    nx.set_edge_attributes(G, {e: 1 for e in G.edges}, 'weight')
    A = nx.adjacency_matrix(G).astype(float)

    if type(G) == nx.Graph:
        # for undirected graphs, in and out treated as the same thing
        out_degree = in_degree = dict(nx.degree(G))
        M = 2. * (G.number_of_edges())
        print("Calculating modularity for undirected graph")
    elif type(G) == nx.DiGraph:
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        M = 1. * G.number_of_edges()
        print("Calculating modularity for directed graph")
    else:
        print('Invalid graph type')
        raise TypeError
    Q = np.zeros(A.shape)
    nodes = list(G)
    for i, j in product(range(len(nodes)), range(len(nodes))):
        Q[i, j] = A[i, j] - (in_degree[nodes[i]] * out_degree[nodes[j]] / M)
    return Q /4




def loss_modularity_trace(logits, B, m):
    
    
    logittrans = logits.transpose(0, 1)
    Module_middle = torch.matmul(logittrans, B.double())
    Modularity = torch.matmul(Module_middle, logits)
    
    modularity_loss = torch.div(torch.trace(Modularity),m)

    return  modularity_loss





def optimized_trace(C, A, d):
    """
    Compute Tr(C^T B C) efficiently using the decomposition Tr(C^T A C - C^T d^T d C)
    Args:
    C (torch.Tensor): Cluster assignment matrix of shape (n, k)
    A (torch.sparse.FloatTensor): Sparse adjacency matrix of shape (n, n)
    d (torch.Tensor): Degree vector of shape (n,)
    Returns:
    torch.Tensor: The trace of C^T B C
    """
    
    
        
    n, k = C.shape
    # Compute C^T A C using sparse operations
    CT_sparse = C.t().to_sparse()
    CT_A = sparse.mm(CT_sparse, A)
    CT_A_C = sparse.mm(CT_A, C)
    # Compute C^T d^T d C
    d_C = torch.mm(d.unsqueeze(0), C)  # Shape: (1, k)
    CT_dT_d_C = torch.mm(d_C.t(), d_C)
    # Compute the final result
    result = CT_A_C - CT_dT_d_C
    # Return the trace
    return torch.trace(result)

def fast_loss_modularity_trace(G, C, m):
    """
    Compute the modularity loss using the optimized trace computation
    Args:
    G (networkx.Graph): The input graph (unweighted)
    C (torch.Tensor): Cluster assignment matrix of shape (n, k)
    m (int): Number of edges in the graph
    Returns:
    torch.Tensor: The modularity loss
    """
    
    
    n = G.number_of_nodes()
    node_map = {node: i for i, node in enumerate(G.nodes())}
    # Convert edges to integer indices
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
    # Create the edge index tensor
    edge_index = torch.tensor(edges, dtype=torch.float).t()
    # Create the edge weight tensor (assuming all edges have weight 1)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    
    
    
    
    # Create sparse adjacency matrix
    #edge_index = torch.tensor(list(G.edges())).t()
    #edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    A = torch.sparse_coo_tensor(edge_index, edge_weight, (n, n),device=device)
    # Create degree vector
    d = torch.tensor([deg for _, deg in G.degree()], dtype=torch.float,device=device)
    # Normalize adjacency matrix and degree vector
    #A = A / 4
    d = d / m
    # Compute the trace

    
    trace = optimized_trace(C, A, d)
    # Compute modularity loss
    modularity_loss = trace/(4*m)
    return modularity_loss
