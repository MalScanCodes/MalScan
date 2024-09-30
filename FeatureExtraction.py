import os.path

import networkx as nx
import time
import argparse
import csv
from multiprocessing import Pool as ThreadPool
from functools import partial
import glob
import numpy as np
import igraph

def parseargs():
    parser = argparse.ArgumentParser(description='Malware Detection with centrality.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains benign and malware.', required=True, type=str)
    parser.add_argument('-o', '--output', help='The dir_path of output', required=True, type=str)
    # parser.add_argument('-c', '--centrality', help='The type of centrality: degree, katz, closeness, harmonic', required=True, type=str)
    args = parser.parse_args()
    return args

def obtain_sensitive_apis(file):
    sensitive_apis = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            else:
                sensitive_apis.append(line.strip())
    return sensitive_apis


def degree_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.gml')[0]
    try:
        g = igraph.Graph.Read_GML(file)
        nodes = g.vcount()
        node_centrality = dict(zip(g.vs['label'], np.array(g.degree())/(nodes-1)))

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)
        return (sha256, vector)

    except:
        return (sha256, None)


def katz_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.gml')[0]
    try:
        CG = nx.read_gml(file)
        node_centrality = nx.katz_centrality(CG)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        return (sha256, vector)

    except:
        return (sha256, None)


def closeness_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.gml')[0]
    try:
        CG = nx.read_gml(file)
        node_centrality = nx.closeness_centrality(CG)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        return (sha256, vector)

    except:
        return (sha256, None)


def harmonic_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.gml')[0]
    try:
        CG = nx.read_gml(file)
        node_centrality = nx.harmonic_centrality(CG)
        # g = igraph.Graph.Read_GML(file)
        # node_centrality = dict(zip(g.vs['label'], g.harmonic_centrality()))
        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        return (sha256, vector)

    except:
        return (sha256, None)


def pagerank_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.gml')[0]
    try:
        g = igraph.Graph.Read_GML(file)
        node_centrality = dict(zip(g.vs['label'], g.pagerank()))

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        return (sha256, vector)

    except:
        return (sha256, None)


def eigenvector_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.gml')[0]
    try:
        g = igraph.Graph.Read_GML(file)
        node_centrality = dict(zip(g.vs['label'], g.eigenvector_centrality()))

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        return (sha256, vector)

    except:
        return (sha256, None)


def authority_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.gml')[0]
    try:
        g = igraph.Graph.Read_GML(file)
        node_centrality = dict(zip(g.vs['label'], g.authority_score()))
        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        return (sha256, vector)

    except:
        return (sha256, None)




def obtain_dataset(dataset_path, centrality_type, sensitive_apis):
    Vectors = []
    Labels = []

    if dataset_path[-1] == '/':
        apps_b = glob.glob(dataset_path + 'benign/*.gml')
        apps_m = glob.glob(dataset_path + 'malware/*.gml')
    else:
        apps_b = glob.glob(dataset_path + '/benign/*.gml')
        apps_m = glob.glob(dataset_path + '/malware/*.gml')

    pool_b = ThreadPool(15)
    pool_m = ThreadPool(15)
    if centrality_type == 'degree':
        vector_b = pool_b.map(partial(degree_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(degree_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    elif centrality_type == 'katz':
        vector_b = pool_b.map(partial(katz_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(katz_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    elif centrality_type == 'closeness':
        vector_b = pool_b.map(partial(closeness_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(closeness_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    elif centrality_type == 'harmonic':
        vector_b = pool_b.map(partial(harmonic_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(harmonic_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    elif centrality_type == 'pagerank':
        vector_b = pool_b.map(partial(pagerank_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(pagerank_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    elif centrality_type == 'eigenvector':
        vector_b = pool_b.map(partial(eigenvector_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(eigenvector_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    elif centrality_type == 'authority':
        vector_b = pool_b.map(partial(authority_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(authority_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    else:
        print('Error Centrality Type!')

    Vectors.extend(vector_b)
    Labels.extend([0 for i in range(len(vector_b))])

    Vectors.extend(vector_m)
    Labels.extend([1 for i in range(len(vector_m))])

    return Vectors, Labels  

def main():
    tic = time.time()
    sensitive_apis_path = 'sensitive_apis_426.txt'
    sensitive_apis = obtain_sensitive_apis(sensitive_apis_path)

    args = parseargs()
    dataset_path = args.dir
    folder = os.path.exists(args.output)
    if not folder:
        os.makedirs(args.output)
    # cetrality_type = args.centrality
    types = ['degree', 'closeness', 'harmonic', 'katz', 'eigenvector', 'pagerank', 'authority']
    for cetrality_type in types:
        Vectors, Labels = obtain_dataset(dataset_path, cetrality_type, sensitive_apis)
        feature_csv = [[] for i in range(len(Labels)+1)]
        feature_csv[0].append('SHA256')
        feature_csv[0].append('Label')
        feature_csv[0].extend(sensitive_apis)

        for i in range(len(Labels)):
            (sha256, vector) = Vectors[i]
            if vector is None:
                print(sha256, 'error')
            else:
                feature_csv[i+1].append(sha256)
                feature_csv[i+1].append(Labels[i])
                feature_csv[i+1].extend(vector)
        if args.output[-1] == '/':
            csv_path = args.output + cetrality_type + '_features.csv'
        else:
            csv_path = args.output + '/' + cetrality_type + '_features.csv'

        with open(csv_path, 'w', newline='') as f:
            csvfile = csv.writer(f)
            csvfile.writerows(feature_csv)
    print(time.time()-tic)

if __name__ == '__main__':
    main()
