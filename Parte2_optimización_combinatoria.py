import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from itertools import permutations
from matplotlib.animation import FuncAnimation

# Definimos las ciudades y sus coordenadas en el mapa 
cities = {
    'Palmira': (3, 2),
    'Pasto': (4, 5),
    'Tuluá': (1, 4),
    'Bogota': (2, 6),
    'Pereira': (3, 3),
    'Armenia': (2, 3),
    'Manizales': (1, 5),
    'Valledupar': (6, 1),
    'Montería': (5, 2),
    'Soledad': (7, 4),
    'Cartagena': (6, 4),
    'Barranquilla': (8, 5),
    'Medellín': (1, 6),
    'Bucaramanga': (2, 8),
    'Cúcuta': (3, 9)
}

# Calcular la matriz de distancias entre ciudades
dist_matrix = distance_matrix(list(cities.values()), list(cities.values()))

# Algoritmo genético para encontrar la mejor ruta
def genetic_algorithm(num_generations, population_size, crossover_rate, mutation_rate):
    num_cities = len(cities)
    population = np.array([np.random.permutation(num_cities) for _ in range(population_size)])

    for generation in range(num_generations):
        fitness = np.array([calculate_fitness(route) for route in population])
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]

        # Seleccionar padres
        parents = population[:int(population_size * crossover_rate)]

        # Cruzar padres para crear hijos
        offspring = crossover(parents, population_size)

        # Aplicar mutación
        mutate(offspring, mutation_rate)

        # Reemplazar población antigua con la nueva generación
        population = np.vstack((parents, offspring))

    best_route = population[0]
    return best_route

# Función de costo (a ajustar según los costos reales)
def calculate_fitness(route):
    total_distance = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    return total_distance

# Operador de cruce (crossover)
def crossover(parents, population_size):
    num_parents = len(parents)
    num_offspring = population_size - num_parents
    offspring = np.empty((num_offspring, len(parents[0])), dtype=int)

    for i in range(num_offspring):
        parent1, parent2 = np.random.choice(num_parents, size=2, replace=False)
        crossover_point = np.random.randint(1, len(parents[0]) - 1)
        offspring[i, :crossover_point] = parents[parent1, :crossover_point]
        offspring[i, crossover_point:] = [city for city in parents[parent2] if city not in offspring[i, :crossover_point]]

    return offspring

# Operador de mutación
def mutate(offspring, mutation_rate):
    for child in offspring:
        if np.random.rand() < mutation_rate:
            mutation_indices = np.random.choice(len(child), size=2, replace=False)
            child[mutation_indices[0]], child[mutation_indices[1]] = child[mutation_indices[1]], child[mutation_indices[0]]

# Función para visualizar el recorrido en el mapa de Colombia
def plot_route(route):
    x = [list(cities.values())[city][0] for city in route]
    y = [list(cities.values())[city][1] for city in route]

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', linestyle='-', color='b')
    ax.plot(x[0], y[0], marker='o', color='r', label='Inicio/Final')
    ax.legend()
    ax.set_title('Mejor Ruta')
    plt.show()

# Animación del proceso de optimización
def animate_optimization(best_routes):
    fig, ax = plt.subplots()
    plt.scatter(*zip(*list(cities.values())), color='red')  # Marcadores de ciudades
    line, = ax.plot([], [], linestyle='-', color='blue', marker='o')

    def update(frame):
        route = best_routes[frame]
        x = [cities[city][0] for city in route]
        y = [cities[city][1] for city in route]
        line.set_data(x, y)
        return line,

    ani = FuncAnimation(fig, update, frames=len(best_routes), interval=500, blit=True)
    plt.show()

# Parámetros del algoritmo genético
num_generations = 100
population_size = 100
crossover_rate = 0.6
mutation_rate = 0.02

# Ejecutar el algoritmo genético
best_route = genetic_algorithm(num_generations, population_size, crossover_rate, mutation_rate)

# Visualizar la mejor ruta
plot_route(best_route)

# Visualizar animación del proceso de optimización
all_routes = list(permutations(cities.keys()))
best_routes = [genetic_algorithm(1, 100, 0.6, 0.02)[0] for _ in range(50)]  # 50 generaciones para la animación
animate_optimization(best_routes)


