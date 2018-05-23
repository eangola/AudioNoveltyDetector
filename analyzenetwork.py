#!/usr/bin/env python
"""
Created : 09-05-2018
Last Modified : Thu 10 May 2018 02:01:32 PM EDT
Created By : Enrique D. Angola
"""
import networkx as nx
import csv
import numpy as np
import matplotlib.pyplot as plt
import community

def read_csv(filename):
    """
    reads csv output from novelty detector library
    """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile,delimiter=' ')
        data = [line for line in reader]

    return data

def find_novel_nodes(novels=None):
    """
    finds and returns indices for nodes that present novelty
    """
    indices = np.where(novels != 0)[0]
    return indices

def find_weights(indices):
    """
    computes and returns weights for the edges
    """
    k = indices
    weights = [1/(k[i+1]-k[i]) for i in range(len(k)-1)]
    return weights

def build_edges(indices=None,weights=None):
    """
    builds and return edges to insert intro graph
    """
    k = indices
    w = weights
    edges = [(k[i],k[i+1],{'weight':w[i]}) for i in range(len(k)-1)]
    return edges

def get_best_partition(G,nodes):
    """
    return best partitions of the graph nodes which maximises
    the modularity
    """
    p = community.best_partition(G)
    clusters = [(node,p[node]) for node in nodes]
    return clusters

def build_network(novelties):
    """
    build the network
    """
    G = nx.Graph()
    for i in range(len(novelties)):
        G.add_node(i)

    indices = find_novel_nodes(novelties)
    w = find_weights(indices)
    edges = build_edges(indices,w)

    G.add_edges_from(edges)
    nodes = list(np.where(novelties!=0)[0])

    return G,nodes

def compute_scores(clusters,novelties):
    """
    compute novelty scores based on clusters
    """
    score = {}
    lastCluster = clusters[0][1]
    tmp = 0
    #iterate through elements in clusters
    for element in clusters:
        node = element[0]
        cluster = element[1]
        #check if still in same cluster, and add novelty to counter
        if cluster == lastCluster:
            tmp += novelties[node]
        else: #if cluster changed, reset tmp and set lastCluster to current
            score[lastCluster] = tmp
            tmp = novelties[node]
            lastCluster = cluster

    score[lastCluster] = tmp

    return score

def write_csv(filename,samples): #pragma: no cover
    """
    Export results to a csv file

    Parameters
    ----------
    filename: str
        path to file to export
    samples: list

    Returns
    -------
    None

    """
    import csv
    with open(filename,'w') as csvfile:
        writer = csv.writer(csvfile, delimiter =' ', quotechar = '|')
        for sample in samples:
            values = [(key,sample[key]) for key in sample.keys()]
            writer.writerow(values)


if __name__ == '__main__':

    filename = 'novelres.csv'
    datanovel = read_csv(filename)
    results = []
    for sample in datanovel:
        goodNovel = np.asarray(sample).astype(float)
        G,nodes = build_network(goodNovel)
        plt.figure()
        nx.draw(G,with_labels=True,node_size=100,font_weight='bold',nodelist=nodes)
        plt.show()
        clusters = get_best_partition(G,nodes)
        scores = compute_scores(clusters,goodNovel)
        results.append(scores)
        write_csv('networkresults.csv',results)


