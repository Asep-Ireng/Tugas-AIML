import random
import math
import matplotlib.pyplot as plt
import networkx as nx

# Membuat peta Rumania sebagai graf dengan jarak antar kota
def create_romania_map():
    romania_map = {
        'Oradea': {'Zerind': 71, 'Sibiu': 151},
        'Zerind': {'Oradea': 71, 'Arad': 75},
        'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
        'Timisoara': {'Arad': 118, 'Lugoj': 111},
        'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
        'Mehadia': {'Lugoj': 70, 'Dobreta': 75},
        'Dobreta': {'Mehadia': 75, 'Craiova': 120},
        'Craiova': {'Dobreta': 120, 'Pitesti': 138, 'Rimnicu Vilcea': 146, 'Giurgiu': 101},
        'Rimnicu Vilcea': {'Craiova': 146, 'Pitesti': 97, 'Sibiu': 80},
        'Sibiu': {'Arad': 140, 'Oradea': 151, 'Rimnicu Vilcea': 80, 'Fagaras': 99},
        'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
        'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
        'Giurgiu': {'Craiova': 101, 'Bucharest': 90},
        'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
        'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
        'Hirsova': {'Urziceni': 98, 'Eforie': 86},
        'Eforie': {'Hirsova': 86},
        'Vaslui': {'Urziceni': 142, 'Iasi': 92},
        'Iasi': {'Vaslui': 92, 'Neamt': 87},
        'Neamt': {'Iasi': 87}
    }
    return romania_map

# Jarak garis lurus (heuristik) ke Bucharest
heuristic_to_bucharest = {
    'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Dobreta': 242,
    'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151,
    'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234,
    'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193,
    'Sibiu': 253, 'Timisoara': 329, 'Urziceni': 80,
    'Vaslui': 199, 'Zerind': 374
}

# Mencari rute acak dari start ke goal
def find_random_path(graph, start, goal):
    path = [start]
    current = start

    while current != goal:
        neighbors = list(graph[current].keys())
        if not neighbors:
            return None  # Tidak ada rute yang tersedia

        # Saring kota yang belum dikunjungi jika memungkinkan
        unvisited = [n for n in neighbors if n not in path]
        if not unvisited and goal not in neighbors:
            # Jika semua tetangga telah dikunjungi dan goal tidak langsung bisa dicapai
            return None

        # Pilih kota berikutnya
        if goal in neighbors:
            next_city = goal
        elif unvisited:
            next_city = random.choice(unvisited)
        else:
            next_city = random.choice(neighbors)

        path.append(next_city)
        current = next_city

    return path

# Menghitung panjang rute
def path_length(graph, path):
    if not path:
        return float('inf')

    length = 0
    for i in range(len(path) - 1):
        length += graph[path[i]][path[i+1]]
    return length

# Algoritma Hill Climbing
def hill_climbing(graph, start, goal, max_iterations=1000):
    # Mulai dengan solusi acak
    current_path = find_random_path(graph, start, goal)
    if not current_path:
        return None, 0

    current_length = path_length(graph, current_path)
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Hasilkan solusi tetangga dengan mencoba mengganti segmen
        best_neighbor = None
        best_neighbor_length = current_length

        for i in range(1, len(current_path) - 1):
            for neighbor in graph[current_path[i-1]]:
                if neighbor != current_path[i] and neighbor in graph and goal in graph[neighbor]:
                    # Coba ganti segmen tertentu
                    new_path = current_path[:i] + [neighbor, goal]
                    new_length = path_length(graph, new_path)

                    if new_length < best_neighbor_length:
                        best_neighbor = new_path
                        best_neighbor_length = new_length

        # Jika tidak ditemukan tetangga yang lebih baik, maka mencapai optima lokal
        if best_neighbor is None or best_neighbor_length >= current_length:
            break

        # Pindah ke tetangga terbaik
        current_path = best_neighbor
        current_length = best_neighbor_length

    return current_path, iterations

# Algoritma Simulated Annealing
def simulated_annealing(graph, start, goal, initial_temp=10, cooling_rate=0.8, max_iterations=1000):
    # Mulai dengan solusi acak
    current_path = find_random_path(graph, start, goal)
    if not current_path:
        return None, 0

    current_length = path_length(graph, current_path)
    best_path = current_path.copy()
    best_length = current_length

    temperature = initial_temp
    iterations = 0

    while temperature > 0.1 and iterations < max_iterations:
        iterations += 1

        # Hasilkan solusi tetangga secara acak
        i = random.randint(1, len(current_path) - 2)  # Memilih kota acak dalam rute
        neighbors = list(graph[current_path[i-1]].keys())
        alternative_cities = [
            n for n in neighbors
            if n != current_path[i] and n != current_path[i+1]
        ]

        if not alternative_cities:
            continue

        new_city = random.choice(alternative_cities)

        # Periksa apakah rute dapat disambung melalui kota ini
        can_complete_path = False
        for next_city in graph[new_city]:
            if next_city in current_path[i+1:]:
                can_complete_path = True
                break

        if not can_complete_path:
            continue

        # Buat rute baru
        new_path = current_path[:i] + [new_city]

        # Cari cara untuk menyambung ke goal
        current = new_city
        while current != goal:
            next_options = [
                n for n in graph[current]
                if n not in new_path or n == goal
            ]
            if not next_options:
                break
            next_city = min(
                next_options,
                key=lambda x: heuristic_to_bucharest.get(x, float('inf'))
            )
            new_path.append(next_city)
            current = next_city
            if current == goal:
                break

        if current != goal:
            continue

        new_length = path_length(graph, new_path)

        # Hitung delta energi (perubahan panjang rute)
        delta_e = new_length - current_length

        # Terima solusi baru berdasarkan probabilitas penerimaan
        if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
            current_path = new_path
            current_length = new_length

            # Perbarui solusi terbaik jika solusi saat ini lebih baik
            if current_length < best_length:
                best_path = current_path.copy()
                best_length = current_length

        # Pendinginan suhu
        temperature *= cooling_rate

    return best_path, iterations

# Eksekusi algoritma
romania_map = create_romania_map()
start_city = 'Oradea'
goal_city = 'Bucharest'

# Hill Climbing
hc_path, hc_iterations = hill_climbing(romania_map, start_city, goal_city)
hc_length = path_length(romania_map, hc_path) if hc_path else float('inf')
print(f"Hill Climbing: {hc_path} dengan panjang {hc_length} (iterasi: {hc_iterations})")

# Simulated Annealing
sa_path, sa_iterations = simulated_annealing(romania_map, start_city, goal_city)
sa_length = path_length(romania_map, sa_path) if sa_path else float('inf')
print(
    f"Simulated Annealing: {sa_path} dengan panjang {sa_length} "
    f"(iterasi: {sa_iterations})"
)
