import pickle
import random


def sampling(G, city):
    sample_num = [1000, 2000, 4000, 8000]
    nodes      = list(G.nodes())

    for num in sample_num:
        attempts     = 0
        sample_nodes = []

        while attempts < num:
            random_nodes = tuple(random.sample(nodes, 2))
            if random_nodes not in sample_nodes:
                sample_nodes.append(random_nodes)
                attempts += 1

        # 체크용
        print(len(sample_nodes))
        
        with open(f'../sample_data/{city}_sample_{num}.pkl', 'wb') as file:
            pickle.dump(sample_nodes, file)


def get_graph(city):
    with open(f'../graph_data/{city}_graph.pkl', 'rb') as file:
        G = pickle.load(file)

    return G


def main():
    cities = []

    with open('../txt/intl.txt', 'r', encoding='UTF-8') as txt_file:
        for line in txt_file:
            city = line.strip().split(',')[0]
            cities.append(city)

    for city in cities:
        G = get_graph(city)
        sampling(G, city)


if __name__ == '__main__':
    main()