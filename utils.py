import os
from functools import lru_cache

from rdflib.graph import Graph
import torch
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.models import TransE,TorusE
import networkx as nx
from collections import Counter
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from config import Config


def Import_ESBM(path):
    lst = os.listdir(path)
    len_index = len(lst)
    g = Graph()
    for i in range(1, len_index + 1):
        g_temp = Graph()
        entity_file_name = path + str(i) + "_desc.nt"
        g = g + g_temp.parse(entity_file_name, format='nt')
    return g


def IMPORT_ESBM_plus(path):
    g = Graph()
    g.parse(path, format="nt")
    triples = []
    for s, p, o in g:
        triples.append([s.n3(), p.n3(), o.n3()])
    return (g)


def Import_ESBM_lmdb(path):
    lst = os.listdir(path)
    g = Graph()
    index = [i for i in range(101, 141)]
    index += [i for i in range(166, 176)]
    for i in index:
        g_temp = Graph()
        entity_file_name = path + str(i) + "_desc.nt"
        g = g + g_temp.parse(entity_file_name, format='nt')
    return g


def Import_ESBM_dbpedia(path):
    lst = os.listdir(path)
    g = Graph()
    index = [i for i in range(1, 101)]
    index += [i for i in range(141, 166)]
    for i in index:
        g_temp = Graph()
        entity_file_name = path + str(i) + "_desc.nt"
        g = g + g_temp.parse(entity_file_name, format='nt')
    return g


def TransETraining(triples, transe_save):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    triples_factory = TriplesFactory.from_labeled_triples(triples)
    #model = TransE(triples_factory=triples_factory, embedding_dim=50).to(device)
    model=TransE(triples_factory=triples_factory, embedding_dim=50).to(device)
    optimizer = Adam(params=model.get_grad_params())
    training_loop = SLCWATrainingLoop(model=model, triples_factory=triples_factory, optimizer=optimizer)
    training_loop.train(num_epochs=200, triples_factory=triples_factory, batch_size=32)
    torch.save(model, transe_save)

    entity_to_id = triples_factory.entity_to_id
    relation_to_id = triples_factory.relation_to_id

    return entity_to_id, relation_to_id


####### EVALUATION : ###########


def directed_ground_truth(path, euri):
    label = Graph()
    triples_labels = []
    label = label.parse(path, format='nt')

    for s, p, o in label:
        triples_labels.append([s.n3(), p.n3(), o.n3()])
    return triples_labels


def ground_truth(path, euri):
    label = Graph()
    triples_labels = []
    label = label.parse(path, format='nt')

    """for s, p, o in label:
        if o == euri:
            triples_labels.append([o.n3(), p.n3(), s.n3()])
        else:
            triples_labels.append([s.n3(), p.n3(), o.n3()])"""

    for s, p, o in label:
        triples_labels.append([s.n3(), p.n3(), o.n3()])

    return triples_labels


def import_directed_top_summary(path, triplenum, sumnum, topsum, targetentity):
    path_gt = path + "/ESBM-groundtruth/groundtruth/" + str(triplenum) + \
              '/' + str(triplenum) + "_gold_top" + str(topsum) + "_" + str(sumnum) + ".nt"
    triples_labels = directed_ground_truth(path_gt, targetentity)
    tpeval = [tuple(teval) for teval in triples_labels]
    tval = set(tpeval)
    return tval


def import_top_summary(path, triplenum, sumnum, topsum, targetentity):
    path_gt = path + "/ESBM-groundtruth/groundtruth/" + str(triplenum) + \
              '/' + str(triplenum) + "_gold_top" + str(topsum) + "_" + str(sumnum) + ".nt"
    triples_labels = ground_truth(path_gt, targetentity)
    tpeval = [tuple(teval) for teval in triples_labels]
    tval = set(tpeval)
    return tval


def fmeasure_score(tval, pval):
    try:
        precision = len(pval.intersection(tval)) / len(pval)
        recall = len(pval.intersection(tval)) / len(tval)
        fmeasure = (2 * precision * recall) / (precision + recall)
    except:
        fmeasure = 0
    return fmeasure


def average_top_fmeasure(triplenum, pval, topsum):
    fmeasurelist = []
    for i in range(6):
        tval = import_top_summary(triplenum, i, topsum)
        fmeasurelist.append(fmeasure_score(tval, pval))
    return fmeasurelist


import rdflib


def import_top_machine_summary(frequency_dict, topsum):
    tpsum = []
    for tsumm in frequency_dict:
        tpsum.append(tuple(ts.n3() if type(ts) == rdflib.term.URIRef else ts for ts in tsumm))
    pval = set(tpsum[0:topsum])
    return pval


### IMPORT DATASET:
def import_dataset():
    config = Config()
    dataset_type = config.format()
    dataset_version = config.benchmark()
    dataset_name = config.dataset()
    data_path = config.data_path()
    output_path = config.output_path()
    if dataset_version == 'esbm_plus':
        file_extension = '.tsv' if dataset_type == 'extract' else '.nt'
        path_data = f'{data_path}/ESBM_PLUS_descriptions/{dataset_name}/complete_{dataset_type}_{dataset_name}{file_extension}'
    else:
        path_data = f'{data_path}/ESBM_descriptions/'

    path_trained = f'{output_path}/Trained_Models'
    path_embedding = f'{output_path}/Trained_Models/obtained_embedding'
    os.makedirs(path_embedding, exist_ok=True)
    path_save_entity_id = f'{path_embedding}/entity_id_{dataset_type}_{dataset_name}_{dataset_version}.pkl'
    path_save_relation_id = f'{path_embedding}/relation_id_{dataset_type}_{dataset_name}_{dataset_version}.pkl'
    save_transe_model = f'{path_trained}/{dataset_name}_{dataset_type}_{dataset_version}.pkl'

    return path_data, path_save_entity_id, save_transe_model, path_save_relation_id


def generate_entity_dataset():
    config = Config()
    dataset_type = config.format()
    dataset_name = config.dataset()
    path = config.data_path()
    entitylist = pd.read_csv(path + '/ESBM-groundtruth/elist.txt', sep='\t', index_col=0)
    if dataset_type == 'extract':
        entitylist['last_part_euri'] = entitylist['euri'].str.split('/').str[-1]
        last_part_euri_list = entitylist[entitylist.dataset == 'dbpedia']['last_part_euri'].tolist()
    if dataset_name == 'lmdb':
        entity_dataset = entitylist[entitylist.dataset == 'lmdb']
    elif dataset_name == 'dbpedia':
        entity_dataset = entitylist[entitylist.dataset == 'dbpedia']
    if dataset_type == "extract":
        return entity_dataset, last_part_euri_list
    else:
        return entity_dataset, None


def generate_triples():
    path_dataset, path_entity_id, transe_save, path_relation_id = import_dataset()
    config = Config()
    dataset_type = config.format()
    dataset_version = config.benchmark()
    dataset_name = config.dataset()

    # TODO: Some triples are empty (subject or object) ?
    triples = []
    triples_test = []

    if dataset_version == 'esbm':
        if dataset_name == 'dbpedia':
            g = Import_ESBM_dbpedia(path_dataset)
            for s, p, o in g:
                triples.append([s.n3(), p.n3(), o.n3()])
            triples = np.array(triples)
            triples_factory = TriplesFactory.from_labeled_triples(triples)
        elif dataset_name == 'lmdb':
            g = Import_ESBM_lmdb(path_dataset)
            for s, p, o in g:
                triples.append([s.n3(), p.n3(), o.n3()])
            triples = np.array(triples)
            triples_factory = TriplesFactory.from_labeled_triples(triples)

    elif dataset_version == 'esbm_plus':
        if dataset_type == 'rdf':
            g = Graph()
            g = g.parse(path_dataset)
            for s, p, o in g:
                triples_test.append([s.n3(), p.n3(), o.n3()])
            triples = np.array(triples_test)
            triples_factory = TriplesFactory.from_labeled_triples(triples)
        else:
            triples_factory = TriplesFactory.from_path(path_dataset)
    triples = triples_factory.triples

    if not os.path.exists(path_entity_id):
        entity_id, relation_id = TransETraining(triples, transe_save)
        with open(path_entity_id, "wb") as file:
            pickle.dump(entity_id, file)
        with open(path_relation_id, "wb") as file:
            pickle.dump(relation_id, file)

    return triples


def load_transe(transe_save, path_entity_id, path_relation_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(transe_save).to(device)
    entity_embeddings = model.entity_representations[0](torch.arange(model.num_entities))
    relation_embeddings = model.relation_representations[0](torch.arange(model.num_relations))

    # The embeddings are PyTorch tensors. You can convert them to NumPy arrays with the following code:
    entity_embeddings_numpy = entity_embeddings.cpu().detach().numpy()
    relation_embeddings_numpy = relation_embeddings.cpu().detach().numpy()

    with open(path_entity_id, 'rb') as f:
        # Load the data from the file
        entity_id = pickle.load(f)
    with open(path_relation_id, 'rb') as f:
        # Load the data from the file
        relation_id = pickle.load(f)
    return entity_id, relation_id, relation_embeddings_numpy


def generate_adj_features(triples, entity_id, transe_save, last_part_euri_list):
    config = Config()
    dataset_type = config.format()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(transe_save).to(device)
    G = nx.DiGraph()
    for head, relation, tail in triples:
        G.add_edge(head, tail, relation=relation)
    adj = nx.adjacency_matrix(G)

    nodes = list(G.nodes())

    # Create a dictionary that maps matrix indices to nodes.
    entity_node_id = {node: i for i, node in enumerate(nodes)}

    label_entity = {}
    features = []

    for node in G.nodes():

        if dataset_type == 'extract':
            if node in last_part_euri_list:
                en_id = entity_id[node]
                # label_entity[node]=en_id
        else:
            en_id = entity_id[node]


        en_id_tensor = torch.tensor(en_id)
        entity_embeddings = model.entity_representations[0](en_id_tensor)
        features.append(entity_embeddings.cpu().detach().numpy())

    ## Covert to Tensor
    adj = adj.toarray()

    return G, adj, features


def generate_node_relation_weight_tensor(relation_id, G, edge_index):
    
    # Existing code to calculate relation frequency and node mapping
    relation_names = [G[u][v]['relation'] for u, v in G.edges()]
    relation_frequency = Counter(relation_names)
    relation_frequency = {relation: frequency for relation, frequency in relation_frequency.items()}
    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    # New code to calculate sum of reverse relation frequencies for each node
    node_relation_sum = {node: 0 for node in G.nodes()}

    for node in G.nodes():
        for neighbor in G[node]:
            relation = G[node][neighbor]['relation']
            en_rel = relation_id[relation]
            node_relation_sum[node] += relation_frequency.get(relation, 1)

    # Convert node_relation_sum to the desired format (list or tensor)
    node_relation_values = [node_relation_sum[node] for node in G.nodes()]
    return node_relation_values



def generate_node_in_relation_weight_tensor(relation_id, G, edge_index):
    
    relation_names = [G[u][v]['relation'] for u, v in G.edges()]
    relation_frequency = Counter(relation_names)
    relation_frequency = {relation: frequency for relation, frequency in relation_frequency.items()}
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    node_relation_sum_in = {node: 0 for node in G.nodes()}

    for node in G.nodes():
        # Process incoming edges
        for predecessor in G.predecessors(node):
            if 'relation' in G[predecessor][node]:  # Checking if the relation attribute exists
                relation = G[predecessor][node]['relation']
                en_rel = relation_id.get(relation, 0)  # Using get to avoid KeyError
                node_relation_sum_in[node] += relation_frequency.get(relation, 1)

    # Convert node_relation_sum to the desired format (list or tensor)
    node_relation_values_in = [node_relation_sum_in[node] for node in G.nodes()]

    # Converting to tensor if needed
    node_relation_tensor_in = torch.tensor(node_relation_values_in, dtype=torch.float32)

    return node_relation_values_in


def generate_edge_weight_tensor(G, edge_index):
    node_degrees = dict(G.degree())
    relation_names = [G[u][v]['relation'] for u, v in G.edges()]
    relation_frequency = Counter(relation_names)
    node_counter = {node:degree for node, degree in node_degrees.items()}
    node_frequency = list(node_counter.values())

    return node_frequency


def aggregate_relation_embeddings_to_list(G, transe_save, relation_id):
    model = torch.load(transe_save)
    node_aggregated_embeddings = {}
    non_zero_shape = None

    # First pass: aggregate embeddings and find the shape of non-zero embeddings
    for node in G.nodes():
        embeddings_sum = None

        for neighbor in G.neighbors(node):
            relation = G[node][neighbor]['relation']
            if isinstance(relation, np.ndarray):
                relation = relation.item()

            en_rel = relation_id.get(relation)
            if en_rel is None:
                continue

            relation_embedding = model.relation_representations[0](
                torch.tensor(en_rel, dtype=torch.long)).cpu().detach().numpy()

            if embeddings_sum is None:
                embeddings_sum = np.zeros_like(relation_embedding)

            embeddings_sum += relation_embedding

        node_aggregated_embeddings[node] = embeddings_sum

        if embeddings_sum is not None and non_zero_shape is None:
            non_zero_shape = embeddings_sum.shape

    # Second pass: create the list of lists with zero padding where needed
    embeddings_list = []
    for node in G.nodes():
        embeddings = node_aggregated_embeddings[node]
        if embeddings is None or not np.any(embeddings):
            # Create a list of zeros with the determined shape
            embeddings_list.append([0] * non_zero_shape[0])
        else:
            embeddings_list.append(embeddings.tolist())

    return embeddings_list


# Generate weighted Adjacency matrix

def create_weighted_adjacency_matrix(G):
    # Calculate the frequency of each relation
    relation_names = [G[u][v]['relation'] for u, v in G.edges()]
    relation_frequency = Counter(relation_names)
    inverse_relation_frequency = {relation: 1 / frequency for relation, frequency in relation_frequency.items()}

    # Create a mapping from node identifiers to indices
    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    # Initialize adjacency matrix
    num_nodes = len(G.nodes())
    weighted_adj_matrix = torch.zeros((num_nodes, num_nodes))

    # Assign weights to the adjacency matrix
    for u, v in G.edges():
        relation = G[u][v]['relation']
        weight = inverse_relation_frequency.get(relation, 1)  # Default weight is 1 if relation not found
        u_idx, v_idx = node_mapping[u], node_mapping[v]
        weighted_adj_matrix[u_idx, v_idx] = weight
        weighted_adj_matrix[v_idx, u_idx] = weight  # Assuming undirected graph

    return weighted_adj_matrix


"""Autoencoder"""


def generate_relation_type_adj(G, relation_id):
    nodes = list(G.nodes())

    # Create a dictionary that maps matrix indices to nodes.
    entity_node_id = {node: i for i, node in enumerate(nodes)}

    # G is your networkx graph
    # entity_id is a dictionary mapping entity names to ids
    # relation_id is a dictionary mapping relation names to ids

    # Get the number of nodes in the graph
    num_nodes = len(G.nodes())

    # Initialize the relation_types matrix with zeros
    relation_types = np.zeros((num_nodes, num_nodes), dtype=int)

    # Fill in the relation_types matrix
    for edge in G.edges(data=True):
        node_a_id = entity_node_id[edge[0]]  # Assuming your nodes are named as per entity_id
        node_b_id = entity_node_id[edge[1]]
        # print(node_a_id,node_b_id)
        relation_type = relation_id[edge[2]['relation']]  # Assuming the relation is stored under the 'relation' key
        relation_types[node_a_id, node_b_id] = relation_type
        # If the graph is undirected, also set the reverse direction
        if not G.is_directed():
            relation_types[node_b_id, node_a_id] = relation_type
        # Now relation_types matrix is filled with the relation types for each edge

    return (relation_types)


def calculate_diversity_penalty(reconstructed_adj, relation_types):
    unique_relations = torch.unique(relation_types)
    relation_frequencies = {rel_type.item(): (relation_types == rel_type).sum().item() for rel_type in unique_relations}

    # Compute inverse frequencies as weights
    edge_weights = {rel_type: 1.0 / (freq + 1e-6) for rel_type, freq in relation_frequencies.items()}

    diversity_penalty = 0.0
    for rel_type in unique_relations:
        rel_type = rel_type.item()
        # Mask to extract corresponding edges for this relation type
        edge_mask = (relation_types == rel_type)

        # Apply the mask to the reconstructed adjacency matrix
        reconstructed_probs = reconstructed_adj[edge_mask]

        # Apply the weight for this relation type
        weighted_penalty = edge_weights[rel_type] * reconstructed_probs.sum()
        diversity_penalty += weighted_penalty

    return diversity_penalty


""""test_cases for penalty"""


# test cases

def edge_probabilities_to_adj_matrix(edge_probabilities, edge_index, num_nodes, threshold=0.2):
    # Initialize a square adjacency matrix with zeros
    adj_matrix = torch.zeros((num_nodes, num_nodes))

    # Fill in the probabilities
    for i, (source, target) in enumerate(edge_index.transpose(0, 1)):
        adj_matrix[source, target] = edge_probabilities[i]

    return adj_matrix


def calculate_relation_type_frequencies(reconstructed_probabilities, relation_types, edge_index, num_nodes):
    # Convert edge probabilities to adjacency matrix
    reconstructed_adj = edge_probabilities_to_adj_matrix(reconstructed_probabilities, edge_index, num_nodes)

    # Calculate frequencies
    unique_types = torch.unique(relation_types)
    frequencies = torch.zeros(len(unique_types))

    for i, rel_type in enumerate(unique_types):
        frequencies[i] = torch.sum(reconstructed_adj[relation_types == rel_type])

    return frequencies


def get_edge_types(edge_types, num_relations):
    return list(map(lambda x: x % num_relations, edge_types))


def get_top_k_frequent_types(edge_types, k):
    counts = Counter(edge_types.tolist())
    top_k_values = {item[0]: i for i, item in enumerate(counts.most_common(k))}
    return top_k_values



def get_typed_adj_matrix(adj, edge_index, edge_types, num_types):
    typed_adj = torch.zeros((num_types, adj.shape[0], adj.shape[1]))
    for edge, type in zip(edge_index.t(), edge_types):
        typed_adj[type, edge[0], edge[1]] = adj[edge[0], edge[1]]
    return typed_adj


def get_typed_adj_matrix_top_k(adj, edge_index, edge_types, k, top_k=None):
    typed_adj = torch.zeros((k + 1, adj.shape[0], adj.shape[1]))
    if top_k == None:
        top_k = get_top_k_frequent_types(edge_types, k)
    for edge, type in zip(edge_index.t(), edge_types):
        if type in top_k:
            typed_adj[top_k[type], edge[0], edge[1]] = adj[edge[0], edge[1]]
        else:
            typed_adj[k, edge[0], edge[1]] = adj[edge[0], edge[1]]
    return typed_adj


def group_edge_types(edge_attr, k):
    # Calculate frequencies
    frequency = Counter(edge_attr)
    # Sort edge types by frequency
    sorted_edge_types = sorted(frequency, key=frequency.get)

    # Identify k-1 least frequent edge types
    least_frequent = set(sorted_edge_types[:k-1])

    # Identify the least frequent type among the rest
    least_frequent_among_rest = sorted_edge_types[k-1]

    # Replace non-least-frequent edge types with the least frequent among the rest
    new_edge_type = [edge if edge in least_frequent else least_frequent_among_rest for edge in edge_attr]

    return new_edge_type