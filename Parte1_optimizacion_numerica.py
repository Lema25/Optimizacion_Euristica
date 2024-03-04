import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution, minimize, dual_annealing
from scipy.optimize import basinhopping, shgo
from matplotlib.animation import FuncAnimation


# Parte 1: optimización numérica

# 1. Escoja dos funciones de prueba

# Seleccionamos Función de Rosenbrock y Función de Rastrigin y las definimos

def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    return 10 * len(x) + sum(x**2 - 10 * np.cos(2 * np.pi * x))



# 2. Optimice las funciones en dos y tres dimensiones usando un método de descenso por gradiente con condición inicial aleatoria

# Definimos el metodo

# Método de descenso por gradiente
def gradient_descent(func, dim):
    x0 = np.random.rand(dim)
    result = minimize(func, x0, method='L-BFGS-B')
    return result.x, result.fun

# Optimización en 2 dimensiones

dim_2 = 2

rosenbrock_gd_2 = gradient_descent(rosenbrock, dim_2)
rastrigin_gd_2 = gradient_descent(rastrigin, dim_2)

# Optimización en 3 dimensiones

dim_3 = 3

rosenbrock_gd_3 = gradient_descent(rosenbrock, dim_3)
rastrigin_gd_3 = gradient_descent(rastrigin, dim_3)



# 3. Optimice las funciones en dos y tres dimensiones usando: algoritmos evolutivos, optimización de partículas y evolución diferencial

# Definimos los metodos

# Algoritmos evolutivos
def evolutionary_algorithm(func, dim):
    bounds = [(0, 1)] * dim
    result = differential_evolution(func, bounds)
    return result.x, result.fun

# Optimización de partículas
def particle_optimization(func, dim):
    bounds = [(0, 1)] * dim
    result = basinhopping(func, np.random.rand(dim), niter=100, minimizer_kwargs={"bounds": bounds})
    return result.x, result.fun

# Evolución diferencial
def differential_evolution_optimization(func, dim):
    bounds = [(0, 1)] * dim
    result = differential_evolution(func, bounds)
    return result.x, result.fun

# Optimización en 2 dimensiones

dim_2 = 2

rosenbrock_ea_2 = evolutionary_algorithm(rosenbrock, dim_2)
rastrigin_ea_2 = evolutionary_algorithm(rastrigin, dim_2)

rosenbrock_po_2 = particle_optimization(rosenbrock, dim_2)
rastrigin_po_2 = particle_optimization(rastrigin, dim_2)

rosenbrock_de_2 = differential_evolution_optimization(rosenbrock, dim_2)
rastrigin_de_2 = differential_evolution_optimization(rastrigin, dim_2)


# Optimización en 3 dimensiones

dim_3 = 3

rosenbrock_ea_3 = evolutionary_algorithm(rosenbrock, dim_3)
rastrigin_ea_3 = evolutionary_algorithm(rastrigin, dim_3)

rosenbrock_po_3 = particle_optimization(rosenbrock, dim_3)
rastrigin_po_3 = particle_optimization(rastrigin, dim_3)

rosenbrock_de_3 = differential_evolution_optimization(rosenbrock, dim_3)
rastrigin_de_3 = differential_evolution_optimization(rastrigin, dim_3)


# Imprimimos resultados

print("\n","Resultados en 2 dimenciones","\n")

print("Rosenbrock - Gradiente Descendente (2D):", rosenbrock_gd_2)
print("Rastrigin - Gradiente Descendente (2D):", rastrigin_gd_2)

print("Rosenbrock - Algoritmo Evolutivo (2D):", rosenbrock_ea_2)
print("Rastrigin - Algoritmo Evolutivo (2D):", rastrigin_ea_2)

print("Rosenbrock - Optimización de Partículas (2D):", rosenbrock_po_2)
print("Rastrigin - Optimización de Partículas (2D):", rastrigin_po_2)

print("Rosenbrock - Evolución Diferencial (2D):", rosenbrock_de_2)
print("Rastrigin - Evolución Diferencial (2D):", rastrigin_de_2)

print("\n","Resultados en 3 dimenciones","\n")

print("Rosenbrock - Gradiente Descendente (3D):", rosenbrock_gd_3)
print("Rastrigin - Gradiente Descendente (3D):", rastrigin_gd_3)

print("Rosenbrock - Algoritmo Evolutivo (3D):", rosenbrock_ea_3)
print("Rastrigin - Algoritmo Evolutivo (3D):", rastrigin_ea_3)

print("Rosenbrock - Optimización de Partículas (3D):", rosenbrock_po_3)
print("Rastrigin - Optimización de Partículas (3D):", rastrigin_po_3)

print("Rosenbrock - Evolución Diferencial (3D):", rosenbrock_de_3)
print("Rastrigin - Evolución Diferencial (3D):", rastrigin_de_3)


# 4. Represente con un gif animado o un video el proceso de optimización de descenso por gradiente y el proceso usando el método heurístico.

# Función de prueba (Rosenbrock)
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Gradiente de la función de prueba (Rosenbrock)
def rosenbrock_gradient(x):
    gradient = np.zeros_like(x)
    gradient[0] = -400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1 - x[0])
    gradient[1:-1] = 200.0 * (x[1:-1] - x[:-2]**2) - 400.0 * x[1:-1] * (x[2:] - x[1:-1]**2) - 2.0 * (1 - x[1:-1])
    gradient[-1] = 200.0 * (x[-1] - x[-2]**2)
    return gradient

# Métodos heurísticos
def evolutionary_algorithm(func, dim):
    bounds = [(0, 1)] * dim
    result = differential_evolution(func, bounds)
    return result.x, result.fun

def particle_optimization(func, dim):
    bounds = [(0, 1)] * dim
    result = basinhopping(func, np.random.rand(dim), niter=100, minimizer_kwargs={"bounds": bounds})
    return result.x, result.fun

def differential_evolution_optimization(func, dim):
    bounds = [(0, 1)] * dim
    result = differential_evolution(func, bounds)
    return result.x, result.fun

# Inicialización de datos
x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(np.array([X, Y]))

# Método de descenso por gradiente
def gradient_descent_animation():
    x0 = np.array([-1.5, 2.0])  # Condición inicial
    iterations = [x0]

    def update(frame):
        nonlocal x0
        gradient = rosenbrock_gradient(x0)
        x0 = x0 - 0.01 * gradient  # Tasa de aprendizaje
        iterations.append(x0.copy())
        ax.clear()
        ax.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap="viridis")
        ax.plot([point[0] for point in iterations], [point[1] for point in iterations], marker='o', color='red')

# Método heurístico (Differential Evolution)
def differential_evolution_animation():
    result = differential_evolution(rosenbrock, bounds=[(-2, 2), (-1, 3)])
    iterations = [result.x]

    def update(frame):
        nonlocal iterations
        ax.clear()
        ax.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap="viridis")
        ax.plot([point[0] for point in iterations], [point[1] for point in iterations], marker='o', color='blue')

# Método heurístico (Algoritmos Evolutivos)
def evolutionary_algorithm_animation():
    bounds = [(0, 1), (0, 1)]
    result = differential_evolution(rosenbrock, bounds=bounds)
    iterations = [result.x]

    def update(frame):
        nonlocal iterations
        ax.clear()
        ax.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap="viridis")
        ax.plot([point[0] for point in iterations], [point[1] for point in iterations], marker='o', color='green')

# Método heurístico (Optimización de Partículas)
def particle_optimization_animation():
    bounds = [(0, 1), (0, 1)]
    result = basinhopping(rosenbrock, np.random.rand(2), niter=100, minimizer_kwargs={"bounds": bounds})
    iterations = [result.x]

    def update(frame):
        nonlocal iterations
        ax.clear()
        ax.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap="viridis")
        ax.plot([point[0] for point in iterations], [point[1] for point in iterations], marker='o', color='purple')

# Crear la animación combinada
fig, ax = plt.subplots()
ax.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap="viridis")

# Animación de descenso por gradiente
gradient_descent_animation()

# Animación de evolución diferencial
differential_evolution_animation()

# Animación de algoritmos evolutivos
evolutionary_algorithm_animation()

# Animación de optimización de partículas
particle_optimization_animation()

plt.show()


# Discuta ¿Qué aportaron los métodos de descenso por gradiente y qué aportaron los métodos heurísticos? 
# Para responder a esta pregunta considere el valor final de la función objetivo y el número de evaluaciones 
# de la función objetivo. Para responder a esta pregunta es posible que se requiera hacer varias corridas de los algoritmos.