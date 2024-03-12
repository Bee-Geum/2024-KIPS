import pickle
from haversine import haversine
import osmnx as ox
import numpy as np


def get_drone(cities, graphs, sample_data):
    samples = [1000, 2000, 4000, 8000]

    for city in cities:
        G     = graphs[city]
        nodes = ox.graph_to_gdfs(G, edges=False)

        for num in samples:
            lat_lons = []
            datas    = sample_data[f'{city}_sample_{num}']

            for data in datas:
                source = data[0]
                target = data[1]
                s_lat  = nodes.iloc[source]['y']
                s_lon  = nodes.iloc[source]['x']
                t_lat  = nodes.iloc[target]['y']
                t_lon  = nodes.iloc[target]['x']

                lat_lons.append([(s_lat, s_lon), (t_lat, t_lon)])
                
            drone_dists = []
            for lat_lon in lat_lons:
                dist = haversine(lat_lon[0], lat_lon[1], unit='m')
                drone_dists.append(dist)

            drone_dists = np.array(drone_dists)
            drone_dists = np.round(drone_dists, 2)

            with open(f'../result/{city}_drone_{num}.pkl', 'wb') as file:
                pickle.dump(drone_dists, file)


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


def main():
    cities      = get_cities()
    graphs      = get_graph(cities)
    sample_data = get_sample(cities)
    get_drone(cities, graphs, sample_data)


if __name__ == '__main__':
    main()