import heapq
import numpy as np


def quickest_path_dijkstra(graph, source, target):
    travel_times = {node: float('inf') for node in graph}
    lengths      = {node: float('inf') for node in graph}
    travel_times[source] = 0
    lengths[source]      = 0

    priority_queue = [(0, source)]

    while priority_queue:
        current_time, current_node = heapq.heappop(priority_queue)

        if current_node == target: 
            return lengths[target], travel_times[target]

        if current_time > travel_times[current_node]:
            continue

        for neighbor, edge_info in graph[current_node].items():
            length      = edge_info['length']
            travel_time = edge_info['travel_time']

            new_time   = current_time + travel_time
            new_length = lengths[current_node] + length

            if new_time < travel_times[neighbor]:
                travel_times[neighbor] = new_time
                lengths[neighbor]      = new_length
                heapq.heappush(priority_queue, (new_time, neighbor))

    if travel_times[target] != float('inf'):
        return lengths[target], travel_times[target]
    else:
        return -1, -1  # 대상까지 경로가 없는 경우


def shortest_path_dijkstra(graph, source, target):
    distances    = {node: float('inf') for node in graph}
    travel_times = {node: float('inf') for node in graph}
    distances[source]    = 0
    travel_times[source] = 0

    priority_queue = [(0, source)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == target:  # 대상 노드에 도달하면 중지하고 결과 반환
            return distances[target], travel_times[target]

        if current_distance > distances[current_node]:
            continue

        for neighbor, edge_info in graph[current_node].items():
            length      = edge_info['length']
            travel_time = edge_info['travel_time']

            new_distance    = current_distance + length
            new_travel_time = travel_times[current_node] + travel_time

            if new_distance < distances[neighbor]:
                distances[neighbor]    = new_distance
                travel_times[neighbor] = new_travel_time
                heapq.heappush(priority_queue, (new_distance, neighbor))

    if distances[target] != float('inf'):
        return distances[target], travel_times[target]
    else:
        return -1, -1  # 대상까지 경로가 없는 경우


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


def simple_path_dijkstra(graph, nodes, source, target):
    dot_products = {node: float('inf') for node in graph}
    distances    = {node: float('inf') for node in graph}
    travel_times = {node: float('inf') for node in graph}
    parents      = {node: None for node in graph}
    distances[source]    = 0
    travel_times[source] = 0
    dot_products[source] = 0

    s_lat = nodes.loc[source]['y']
    s_lon = nodes.loc[source]['x']
    t_lat = nodes.loc[target]['y']
    t_lon = nodes.loc[target]['x']

    std_vec = direction_unit_vector(s_lat, s_lon, t_lat, t_lon)

    priority_queue = [(0, source)]

    while priority_queue:
        current_dot_product, current_node = heapq.heappop(priority_queue)

        if current_node == target:
            break

        if dot_products[current_node] < current_dot_product:
            continue

        for neighbor, edge_info in graph[current_node].items():
            inter_vec   = edge_info['vector']
            distance    = edge_info['length']
            travel_time = edge_info['travel_time']

            new_distance    = distances[current_node] + distance
            new_travel_time = travel_times[current_node] + travel_time

            # 활성화 함수 처럼 y = -x + 1 적용
            # 유닛 벡터 내적값은 -1 ~ 1 사이기 떄문에, 값이 작을수록 weight를 크게 줘야하기 때문에 이렇게 함.
            dot_product = -np.dot(std_vec, inter_vec) + 1
            final_vec   = current_dot_product + dot_product

            if final_vec < dot_products[neighbor]:
                dot_products[neighbor] = final_vec
                distances[neighbor]    = new_distance
                travel_times[neighbor] = new_travel_time
                parents[neighbor]      = current_node
                heapq.heappush(priority_queue, (final_vec, neighbor))

    path = []
    current_node = target
    while current_node is not None:
        path.append(current_node)
        current_node = parents[current_node]
    path.reverse()


    if dot_products[target] != float('inf'):
        return distances[target], travel_times[target], path
    else:
        return -1, -1, []  # 대상까지 경로가 없는 경우


def get_nested_dict_graph(G):
    # 그래프를 딕셔너리로 변환
    graph_dict = {}

    for node in G.nodes():
        edges_data = {}

        for neighbor in G.successors(node):
            length_time_vec = {}

            edge_data = G.get_edge_data(node, neighbor)
            edge_data = list(edge_data.values())[0]

            length = edge_data['length']
            travel_time = edge_data['travel_time']
            vector = edge_data['vector']

            length_time_vec['length'] = length
            length_time_vec['travel_time'] = travel_time
            length_time_vec['vector'] = vector

            edges_data[neighbor] = length_time_vec

        graph_dict[node] = edges_data

    return graph_dict