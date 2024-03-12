import osmnx as ox
import networkx as nx
import pickle
import os
import numpy as np


def get_graph(city):

    G = ox.graph_from_place(city, network_type='drive')
    G = ox.add_edge_bearings(G)

    return G


def direction_unit_vector(lat1, lon1, lat2, lon2):
    """
    Calculate the direction unit vector between two points given their latitudes and longitudes.
    If the points are the same, return a default value or an error.

    Args:
    lat1, lon1: Latitude and longitude of the first point.
    lat2, lon2: Latitude and longitude of the second point.

    Returns:
    A unit vector representing the direction from the first point to the second point, or a default/error value if points are the same.
    """
    # Convert latitudes and longitudes to radians for computation
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calculate the differences in coordinates
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    # Check if the vector is zero
    if delta_lon == 0 and delta_lat == 0:
        # Return a default value or raise an error
        return np.array([0, 0])  # or raise ValueError("Points are the same.")

    # Calculate the vector
    vector = np.array([delta_lon, delta_lat])

    # Normalize the vector to get the unit vector
    unit_vector = vector / np.linalg.norm(vector)

    return unit_vector


def get_unitvec(G):
    node_dict = {}

    for node, data in G.nodes(data=True):
        node_dict[node] = data

    for source, target, data in G.edges(data=True):
        source_lat = node_dict[source]['y']
        source_lon = node_dict[source]['x']
        target_lat = node_dict[target]['y']
        target_lon = node_dict[target]['x']

        vector = direction_unit_vector(source_lat, source_lon, target_lat, target_lon)
        data['vector'] = vector

    return G


def node_ascending_order(graph):
    # 노드 번호 0부터 오름차순으로 바꾸기
    new_node_id_mapping = {}
    new_id = 0
    for old_node_id in graph.nodes():
        new_node_id_mapping[old_node_id] = new_id
        new_id += 1

    new_graph = nx.relabel_nodes(graph, new_node_id_mapping)

    return new_graph


def add_time(graph):
    # travl_time 추가 위해 필수

    # Fallback 속도를 30km/h로 설정 ( add_edge_speeds가 데이터가 없을 때, default로 속도 30으로 설정 )
    fallback_speed = 30

    # add_edge_speeds 함수 호출
    graph = ox.add_edge_speeds(graph, fallback=fallback_speed)
    graph = ox.add_edge_travel_times(graph)

    # 각 노드에서의 대기시간 더해주기
    w = 20
    k = 30
    node_wait_times = {}

    for node, degree in graph.degree():
        if degree <= 2:
            wait_time = w
        else:
            wait_time = w + k * (degree - 2)

        node_wait_times[node] = wait_time

    # multidigraph에서 digraph로 변환
    for s, t, key, data in graph.edges(keys=True, data=True):
        # 도착 vertex만 할당, 나중에 path에서 계산할때는 끝 vertex 시간 빼주기. 시작점과 끝점 시간은 빼주기로 했음
        wait_time = node_wait_times[t]
        data["travel_time"] += wait_time

    return graph


def save_graph(graph, city):
    city = city.split(',')[0]

    with open(os.path.join(f'../graph_data/{city}_graph.pkl'), 'wb') as file:
        pickle.dump(graph, file)


def type_fix(G):
    # to save memory
    for u, v, data in G.edges(data=True):
        data['length']      = np.float16(data['length'])
        data['travel_time'] = np.float16(data['travel_time'])

    return G


def main():
    cities = []

    with open('../txt/intl.txt', 'r', encoding='UTF-8') as file:
        for line in file:
            city = line.strip()
            cities.append(city)

    for city in cities:
        print(city)

        G          = get_graph(city)
        graph      = node_ascending_order(G)
        t_graph    = add_time(graph)
        f16_graph  = type_fix(t_graph)
        vec_graph  = get_unitvec(f16_graph)

        save_graph(vec_graph, city)


if __name__ == "__main__":
    main()