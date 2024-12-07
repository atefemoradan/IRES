from functools import lru_cache
import rdflib.term
from collections import Counter
import pandas as pd
import networkx as nx
import netlsd
import utils
import random
import numpy as np
import torch


@lru_cache
def graph_stat(G):
    # Create a dictionary that maps matrix indices to nodes.
    nodes = list(G.nodes())
    entity_node_id = {node: i for i, node in enumerate(nodes)}
    relation_names = [G[u][v]['relation'] for u, v in G.edges()]
    relation_frequency = Counter(relation_names)
    relation_frequency = {relation: 1 / frequency for relation, frequency in relation_frequency.items()}
    return entity_node_id, relation_names, relation_frequency


@lru_cache(maxsize=None)
def import_top_summary(path, eid, i, k, targetentity):
    return utils.import_top_summary(path, eid, i, k, targetentity)




def calculate_neighbor_community_sizes(G, z, entity_node_id, k):
    """
    Calculate sizes of top k neighboring communities for each node in the graph.

    Args:
        G (NetworkX Graph): The graph containing the nodes and edges.
        z (torch.Tensor): A tensor where each row represents the community probabilities for a node.
        entity_node_id (dict): A dictionary mapping graph node identifiers to indices in the tensor z.
        k (int): Number of top communities to consider for each node.

    Returns:
        dict: A dictionary where keys are node identifiers and values are dictionaries.
              Each value dictionary maps a community label to the count of neighboring nodes belonging to that community,
              considering the top k community assignments based on scores.
    """

    community_sizes = {}
   
     
    for node in list(G.nodes()):
        
        
        #target_node='<http://data.linkedmdb.org/resource/film/12398>'
        neighbor_nodes = list(G.neighbors(node))
    
        #print("TAMAM")
        # This will store counts of neighbors belonging to their top k communities
        community_counts = {}

        for neighbor in neighbor_nodes:
            
            # Get the index of the node in the tensor z
            idx = entity_node_id[neighbor]
            # Find the top k communities based on the probabilities (scores)
            #print(neighbor)
            top_k_communities = torch.topk(z[idx], k, largest=True).indices.tolist()
            #print(top_k_communities)
           
            #print(neighbor)
            #print(top_k_communities)
            #print(top_k_communities)
            for community in top_k_communities:
                if community in community_counts:
                    community_counts[community] += 1
                else:
                    community_counts[community] = 1

        community_sizes[node] = community_counts
   
        #print("TAMAM")
        
        #print(community_counts)
    #print(community_sizes)
    return community_sizes


def calculate_entropy(z):
    """
    Calculate the entropy of community assignments for each node.

    Args:
        z (torch.Tensor): Tensor of community probabilities for each node.

    Returns:
        np.array: Array of entropy values for each node.
    """
    z = torch.sigmoid(z) 
    # Avoid log(0) by replacing zero probabilities with a very small number
    z_safe = torch.where(z == 0, torch.tensor(1e-10), z)
    entropy = -torch.sum(z_safe * torch.log(z_safe), dim=1)
    return entropy

def average_precision(ground_truth_summary,predicted_summary):
    
  
     # Calculate average precision
    relevant_count = 0
    cumulative_precision = 0

    # Iterate over each predicted summary
    for i, summary in enumerate(predicted_summary):
        if summary in ground_truth_summary:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            cumulative_precision += precision_at_i

     # Calculate average precision for this entity
    if relevant_count > 0:
        average_precision = cumulative_precision / len(ground_truth_summary)
        average_precision=average_precision
    else:
        average_precision=0
    
    return average_precision




   
def evaluate(path, entity_dataset, G, k, node_weights, z):
    """
    Evaluate by finding summaries for entities in a dataset and calculating F-score for the summaries.
    Ensure the summary set has exactly k members if possible.
    
    Parameters:
        G (networkx.Graph): The graph of nodes and edges.
        z (np.array): The n*k community score matrix.
        path (str): Path for importing summaries.
        entity_dataset (pd.DataFrame): Dataset containing entity information.
        k (int): Number of top summary edges to find.
    
    Returns:
        tuple: Mean F-scores for entities with and without considering relation.
    """
    
    
    entity_node_id, relation_names, relation_frequency = graph_stat(G)
    
    #entropy_values = calculate_entropy(z) 

    results = []
    results_norel = []
    
    for index, row in entity_dataset.iterrows():
        targetentity = f'<{row["euri"]}>'
        node_id = entity_node_id.get(targetentity)
        if node_id is None:
            continue
        out_edges = list(G.out_edges(targetentity, data=True))
        
        random.shuffle(out_edges)
        summary_edges = []
        added_communities = set()
        #target_communities = torch.topk(z[node_id], 1).indices.tolist()
        
        # Sort edges based on the community size of the neighbor nodes' primary community
        #out_edges.sort(key=lambda x: neighbor_community_sizes[x[1]].get(torch.argmax(z[entity_node_id[x[1]]]).item(), float('inf')))
        # Step: Modify the sort criteria for out_edges to prioritize nodes whose top k community sizes are smaller.
        #out_edges.sort(key=lambda x: sum(neighbor_community_sizes[x[1]].values()), reverse=True)
        #out_edges.sort(key=lambda x: entropy_values[entity_node_id[x[1]]], reverse=True)
        #out_edges = list(G.out_edges(targetentity, data=True))
        
        
        all_edges = []
        # Collect out-edges with weights
        for edge in out_edges:
            other_node = edge[1] if edge[0] == targetentity else edge[0]
            weight = node_weights[node_id][entity_node_id[other_node]].item()
            # Optionally, adjust weight by relation frequency if needed
            # weight *= relation_frequency[edge[2]['relation']]
            all_edges.append((edge, weight))
            
        all_edges.sort(key=lambda x: x[1], reverse=True)
        weight_portion=all_edges[-1][1]
        #print(all_edges[:k])   
        
        for edge in G.in_edges(targetentity, data=True):
            other_node = edge[1] if edge[0] == targetentity else edge[0]
            weight = node_weights[node_id][entity_node_id[other_node]].item()+weight_portion
            all_edges.append((edge, weight))

        # Sort all_edges by weight in descending order
        all_edges.sort(key=lambda x: x[1], reverse=True)
        #print(all_edges)
        
        
        # Extract just the edges from the sorted all_edges list
        out_edges = [edge for edge, weight in all_edges]
        
        #visualization_dict={}
        #visualization_dict[targetentity]=target_communities
        for edge in out_edges:
            
            """if len(summary_edges) >= k:
                break"""
            other_node = edge[1]
            other_node_id = entity_node_id[other_node]
            other_communities = torch.topk(z[other_node_id],1).indices.tolist()
            
            #edge_key = (edge[0], edge[1])  # Assuming edge is a tuple (node1, node2, attr_dict)
            # Store both community data and the edge's attributes in the value
            """visualization_dict[edge_key] = {
                'communities': other_communities,
                'attributes': edge[2]  # This assumes the third element in the tuple is the attribute dict
            }"""
         
            

            #print(torch.topk(z[other_node_id],1).indices.tolist())
            #print(torch.topk(z[other_node_id],1).values.tolist())
            #not set(other_communities).issubset(added_communities):
            #not set(other_communities).issubset(added_communities)
            
            if not set(other_communities).issubset(added_communities) :
                summary_edges.append(edge)
                added_communities.update(set(other_communities))
        #print(visualization_dict)      
        if len(summary_edges) < k:
            for edge in out_edges:
                if len(summary_edges) >= k:
                    break
                # Edge is added without additional checks if it's not already in summary_edges
                if edge not in summary_edges:
                    summary_edges.append(edge)

        # Ensure the number of summary edges is exactly k, if possible
        summary_edges = summary_edges[:k]
        
     
        #print("yes")
        #print(visualization_dict)
        
        pval = []
        for edge in summary_edges:
            relation = edge[2]['relation']
            tsumm = (edge[0], relation, edge[1])
            pval.append(tuple(ts.n3() if type(ts) == rdflib.term.URIRef else ts for ts in tsumm))
        
        #print(len(pval))
        for i in range(6):
            tval = import_top_summary(path, index, i, k, targetentity)
            fscore = utils.fmeasure_score(tval, set(pval))
            Ap=average_precision(tval, set(pval))
            
            #print('average:', Ap)
            results.append({'eid': index, 'euri': targetentity, 'fmeasure': fscore,'ave_precision':Ap, 'Top': k})
        
        pval_norel = {(item[0], item[-1]) for item in pval}
        for i in range(6):
            tval = import_top_summary(path, index, i, k, targetentity)
            tval_norel = {(item[0], item[-1]) for item in tval}
            fscore = utils.fmeasure_score(tval_norel, set(pval_norel))
            Ap_norel=average_precision(tval_norel, set(pval_norel))
            results_norel.append({'eid': index, 'euri': targetentity, 'fmeasure_norel': fscore,'ave_precision_norel':Ap_norel, 'Top': k})
    
    results_df = pd.DataFrame(results)
    results_norel_df = pd.DataFrame(results_norel)
    
    averages = results_df.groupby('euri')['fmeasure'].mean()
    averages_norel = results_norel_df.groupby('euri')['fmeasure_norel'].mean()
    
    averages_ap= results_df.groupby('euri')['ave_precision'].mean()
    averages_norel_ap = results_norel_df.groupby('euri')['ave_precision_norel'].mean()
    
    return (averages.mean(), averages_norel.mean(),averages_ap.mean(),averages_norel_ap.mean())

# Note: Helper functions like graph_stat, calculate_neighbor_community_sizes, import_top_summary, and utils.fmeasure_score
# would need to be defined to make this function work properly.




def evaluate_old(path, entity_dataset, G, k, node_weights, adj_syn):
    entity_node_id, relation_names, relation_frequency = graph_stat(G)
     
    
    #new_adj_syn = adj_syn.clone()

    # adj_syn=adj_syn_temp
    # node_weights=weights_temp
    # node_weights= weights * adj.float()

    restuls = []
    restulsnorel = []
    top_k_edges=[]
    for index, row in entity_dataset.iterrows():
        pval = []
        eid = index
        #print(eid)
        target_temp = row['euri']
        targetentity = f'<{target_temp}>'
        node_id = entity_node_id[targetentity]
        #new_adj_syn[node_id, :] = 0
        #new_adj_syn[:, node_id] = 0

        #print(node_id)

        # in_edges = list(G.in_edges(targetentity, data=True))
        out_edges = list(G.out_edges(targetentity, data=True))
        
      
                
                
        all_edges = []
        random.shuffle(out_edges)
        # Collect out-edges with weights
        for edge in out_edges:
            other_node = edge[1] if edge[0] == targetentity else edge[0]
            weight = node_weights[node_id][entity_node_id[other_node]].item()
            #weight=weight*relation_frequency[edge[2]['relation']]
            all_edges.append((edge, weight))
            
            all_edges.sort(key=lambda x: x[1], reverse=True)
            
            #print(all_edges)
            weight_portion=all_edges[-1][1]
                        
        """for edge in G.in_edges(targetentity, data=True):
            other_node = edge[1] if edge[0] == targetentity else edge[0]
            weight = node_weights[node_id][entity_node_id[other_node]].item()+weight_portion
            all_edges.append((edge, weight))"""
        
        
        
       # all_edges.sort(key=lambda x: x[1], reverse=True)
        
       
        #top_k_edges = all_edges[:k]
        #print(top_k_edges)
        # Select top k edges
        """if k % 2 ==0:
            top_k_edges = all_edges[:k-int(k/2)]
            top_k_edges.extend(all_edges[-(k-int(k/2)):])
        else:
            top_k_edges = all_edges[:k-int(k/2)]
            top_k_edges.extend(all_edges[-2:])"""
        
        #print(len(top_k_edges))
        #top_k_edges .append(all_edges[-1:k-k/2])
        # Find the lowest weight in the top k edges
        #print(len(top_k_edges))
        #top_k_edges.sort(key=lambda x: x[1], reverse=False)
        #print(len(top_k_edges))
        
        """lowest_weight_in_top_k = top_k_edges[-1][1]
        
        # Collect edges with the same weight as the lowest in top k
        equal_weight_edges = [(edge, weight) for edge, weight in all_edges if weight == lowest_weight_in_top_k]

        for i in range(len(top_k_edges)):
            if top_k_edges[i][1] > lowest_weight_in_top_k:
                continue
            else:
                current_edge = top_k_edges[i]
                current_relation = current_edge[0][2]['relation'] if len(current_edge[0]) > 2 and 'relation' in \
                                                                     current_edge[0][2] else None

                for edge, weight in equal_weight_edges:
                    if len(edge) > 2 and 'relation' in edge[2]:
                        other_relation = edge[2]['relation']
                        if edge != current_edge[0] and weight * relation_frequency[other_relation] > weight * \
                                relation_frequency[current_relation]:
                            new_edge = (edge, weight)
                            if new_edge not in top_k_edges:
                                top_k_edges[i] = new_edge
                                break"""

        # Ensure that top_k_edges only contains k elements
        top_k_edges = all_edges[:k]
        #print(top_k_edges)
        print
        
        pval = []
        for edge, _ in top_k_edges[:k]:
            relation = edge[2]['relation']
            tsumm = (edge[0], relation, edge[1])
            pval.append(tuple(ts.n3() if type(ts) == rdflib.term.URIRef else ts for ts in tsumm))

        for i in range(6):
            tval = import_top_summary(path, eid, i, k, targetentity)
            fscore = utils.fmeasure_score(tval, set(pval))
            resultsirow = {'eid': eid, 'euri': targetentity, 'fmeasure': fscore, 'Top': k}
            restuls.append(resultsirow)
        

        for i in range(6):
            tval = import_top_summary(path, eid, i, k, targetentity)
            tval_norel = {(item[0], item[-1]) for item in tval}
            pval_norel = {(item[0], item[-1]) for item in pval}
            fscore = utils.fmeasure_score(tval_norel, set(pval_norel))
            resultsirow1 = {'eid': eid, 'euri': targetentity, 'fmeasure_norel': fscore, 'Top': k}
            restulsnorel.append(resultsirow1)

    
    resultsidf = pd.DataFrame(restuls, columns=['eid', 'euri', 'fmeasure', 'Top'])
    resultsidf1 = pd.DataFrame(restulsnorel, columns=['eid', 'euri', 'fmeasure_norel', 'Top'])
    
    averages = resultsidf1.groupby('euri')['fmeasure_norel'].mean()
    averages_df1 = averages.reset_index()

    averages = resultsidf.groupby('euri')['fmeasure'].mean()
    averages_df = averages.reset_index()

    return (averages_df['fmeasure'].mean(), averages_df1['fmeasure_norel'].mean())


def netlsd_signiture_distance(sparse_adj, G):
    sparse_adj = sparse_adj.detach().numpy()


    G_undirected = G.to_undirected()
    GS = nx.Graph()
    for i in range(len(sparse_adj)):
        for j in range(len(sparse_adj[i])):
            if sparse_adj[i][j] != 0:
                GS.add_edge(i, j, weight=sparse_adj[i][j])

    desc1 = netlsd.heat(GS)
    desc2 = netlsd.heat(G_undirected)

    distance = netlsd.compare(desc1, desc2)
    return distance
