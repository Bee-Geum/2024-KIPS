import pickle
import numpy as np
import algorithms
import osmnx as ox
import multiprocessing


def get_sample(cities):
    samples = [1000, 2000, 4000, 8000]
    sample_data = {}

    for city in cities:
        for num in samples:
            with open(f'../sample_data/{city}_sample_{num}.pkl', 'rb') as file:
                data = pickle.load(file)

            sample_data[f'{city}_sample_{num}'] = data
            
    return sample_data


def get_graph(cities):
    graphs = {}

    for city in cities:
        with open(f'../graph_data/{city}_graph.pkl', 'rb') as file:
            G = pickle.load(file)
            graphs[city] = G
    
    return graphs


def get_cities():
    cities = []

    with open('../txt/intl.txt', 'r') as txt_file:
        for line in txt_file:
            city = line.strip().split(',')[0]
            cities.append(city)

    return cities


def get_nodes(graphs, cities):
    nodes_data = {}

    for city in cities:
        G     = graphs[city]
        nodes = ox.graph_to_gdfs(G, edges=False)

        nodes_data[city] = nodes

    return nodes_data


def get_dict_graph(graphs, cities):
    dict_G = {}

    for city in cities:
        G = graphs[city]
        graph_dict = algorithms.get_nested_dict_graph(G)

        dict_G[city] = graph_dict

    return dict_G


def get_simple_path(tup):

    city, num, nodes, samples, graph = tup

    print(f"Processing sample number = {num} in {city}")

    results = []
    paths   = []

    for sample_tup in samples:
        source = sample_tup[0]
        target = sample_tup[1]

        simple_dist, simple_time, path = algorithms.simple_path_dijkstra(graph, nodes, source, target)

        simple_time /= 60

        result = [simple_dist, simple_time]
        results.append(result)
        paths.append(path)
    
    results = np.array(results)
    results = np.round(results, 2)

    with open(f'../result/{city}_simple_{num}.pkl', 'wb') as file1, \
        open(f'simple_paths/{city}_SimplePath_{num}.pkl', 'wb') as file2:
        pickle.dump(results, file1)
        pickle.dump(paths, file2)


def main():
    cities  = get_cities()
    graphs  = get_graph(cities)
    samples = get_sample(cities)
    nodes   = get_nodes(graphs, cities)
    dict_G  = get_dict_graph(graphs, cities)

    file_infos = []

    sample_num = [1000, 2000, 4000, 8000]
    for city in cities:
        graph      = dict_G[city]
        nodes_data = nodes[city]

        for num in sample_num:
            sample_list = samples[f'{city}_sample_{num}']
            file_infos.append((city, num, nodes_data, sample_list, graph))

    with multiprocessing.Pool(processes=10) as pool:
        pool.map(get_simple_path, file_infos)


if __name__ == '__main__':
    main()