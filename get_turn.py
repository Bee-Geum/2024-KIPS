import pickle
import numpy as np
import algorithms
import multiprocessing


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


def get_dict_graph(graphs, cities):
    dict_G = {}

    for city in cities:
        G = graphs[city]
        graph_dict = algorithms.get_nested_dict_graph(G)

        dict_G[city] = graph_dict

    return dict_G


def get_paths(cities):
    samples = [1000, 2000, 4000, 8000]
    paths_data = {}

    for city in cities:
        for num in samples:
            with open(f'simple_paths/{city}_SimplePath_{num}.pkl', 'rb') as file:
                data = pickle.load(file)

            paths_data[f'{city}_sample_{num}'] = data
            
    return paths_data


def get_turn(tup):
    city, num, paths, G = tup

    print(f"Processing sample number = {num} in {city}")

    threshold = 0.5 # cos60

    turns = []
    
    for path in paths:
        turn = 0

        for i in range(len(path)-2):
            current_vec = G[path[i]][path[i+1]]['vector']
            next_vec    = G[path[i+1]][path[i+2]]['vector']

            if np.dot(current_vec, next_vec) < threshold:
                turn += 1
        
        turns.append(turn)
    
    turns = np.array(turns)

    with open(f'../result/{city}_turns_{num}.pkl', 'wb') as file:
        pickle.dump(turns, file)



def main():
    cities     = get_cities()
    graphs     = get_graph(cities)
    dict_G     = get_dict_graph(graphs, cities)
    paths_data = get_paths(dict_G)

    file_infos = []

    sample_num = [1000, 2000, 4000, 8000]
    for city in cities:
        G = dict_G[city]

        for num in sample_num:
            paths = paths_data[f'{city}_sample_{num}']
            file_infos.append((city, num, paths, G))


    with multiprocessing.Pool(processes=10) as pool:
        pool.map(get_turn, file_infos)



if __name__ == '__main__':
    main()