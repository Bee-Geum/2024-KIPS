import pickle
import numpy as np
import multiprocessing
import algorithms


def calculate(tup):
    city, num, sample_list, graph = tup

    print(f"Processing sample number = {num} in {city}")

    results = []

    for sample_tup in sample_list:
        source = sample_tup[0]
        target = sample_tup[1]

        spd, spt = algorithms.shortest_path_dijkstra(graph, source, target)
        qpd, qpt = algorithms.quickest_path_dijkstra(graph, source, target)
        
        spt /= 60
        qpt /= 60

        result = [spd, spt, qpd, qpt]
        
        results.append(result)

    results = np.array(results)
    results = np.round(results, 2)

    with open(f'../result/{city}_sq_{num}.pkl', 'wb') as file:
        pickle.dump(results, file)


def get_graph(cities):
    graphs = {}

    for city in cities:
        with open(f'../graph_data/{city}_graph.pkl', 'rb') as file:
            G          = pickle.load(file)
            graph_dict = algorithms.get_nested_dict_graph(G)

        graphs[city] = graph_dict
    
    return graphs


def get_sample(cities):
    sample_num  = [1000, 2000, 4000, 8000]
    sample_data = {}

    for city in cities:
        for num in sample_num:
            with open(f'../sample_data/{city}_sample_{num}.pkl', 'rb') as file:
                data = pickle.load(file)

            sample_data[f'{city}_sample_{num}'] = data

    return sample_data


def get_cities():
    cities = []

    with open('../txt/intl.txt', 'r', encoding='UTF-8') as txt_file:
        for line in txt_file:
            city = line.strip().split(',')[0]
            cities.append(city)

    return cities


def main():
    cities  = get_cities()
    samples = get_sample(cities)
    graphs  = get_graph(cities)

    file_infos = []

    sample_num = [1000, 2000, 4000, 8000]
    for city in cities:
        graph = graphs[city]

        for num in sample_num:
            sample_list = samples[f'{city}_sample_{num}']
            file_infos.append((city, num, sample_list, graph))

    with multiprocessing.Pool(processes=10) as pool:
        pool.map(calculate, file_infos)


if __name__ == '__main__':
    main()