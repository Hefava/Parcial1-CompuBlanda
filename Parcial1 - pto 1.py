import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(0)  # Semilla

# Parámetros
b = np.array([63, 28, 24, 85, 53, 94, 44, 67, 59, 64, 56, 100, 35, 26, 63, 19, 83, 31, 30, 31])  # Beneficios de los ítems
p = np.array([67, 32, 92, 85, 50, 70, 36, 85, 67, 13, 84, 22, 60, 52, 95, 12, 18, 59, 35, 93])  # Pesos de los ítems
d = len(b)  # Número de ítems
C = 563.5  # Capacidad máxima de la mochila
N = 100  # Tamaño de la población
generations = 500  # Número de generaciones

# Parámetros adicionales
cross_rate = 0.8  # Porcentaje de cruzamiento
mutation_rate = 0.02  # Tasa de mutación

# Inicializar la población
pop = np.random.randint(0, 2, (N, d))  # Población inicial

# Función de adaptación - Penalización
def fitness(xi, b, p, C):
    wi = xi.dot(p)
    bi = xi.dot(b)
    if wi > C:
        return bi - (wi - C) * 0.5  # Penalización para soluciones inviables
    return bi

# Evaluar el fitness de cada individuo
fos = np.array([fitness(ind, b, p, C) for ind in pop])

# Selección por ruleta
def roulette_selection(fos):
    fitness_sum = np.sum(fos)
    probs = fos / fitness_sum
    idx = np.random.choice(range(len(fos)), p=probs)
    return idx

# Función de cruce (crossover)
def crossover(p1, p2, cross_rate):
    if np.random.rand() < cross_rate:
        crossover_point = np.random.randint(1, len(p1))
        child = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
        return child
    return p1.copy()

# Función de mutación
def mutate(xi, mutation_rate):
    if np.random.rand() < mutation_rate:
        pos = np.random.randint(0, len(xi))
        xi[pos] = 1 - xi[pos]

# Ciclo principal - generaciones
best_in_gen = []
incumbent_fitness = -np.inf  # Inicializamos con un valor muy bajo
incumbent_weight = 0
incumbent_individual = np.zeros(d)
population_data = []

for gen in range(generations):
    new_population = []
    
    for _ in range(N):
        # Seleccionar padres mediante ruleta
        parent1_idx = roulette_selection(fos)
        parent2_idx = roulette_selection(fos)
        p1 = pop[parent1_idx]
        p2 = pop[parent2_idx]

        # Realizar cruzamiento para producir hijos
        child = crossover(p1, p2, cross_rate)

        # Aplicar mutación a los hijos
        mutate(child, mutation_rate)

        # Evaluar el fitness de los hijos
        child_fitness = fitness(child, b, p, C)

        # Agregar el hijo a la nueva población
        new_population.append((child, child_fitness))
    
    # Reemplazar la población con los nuevos individuos
    pop = np.array([ind for ind, fit in new_population])
    fos = np.array([fit for ind, fit in new_population])

    # Encontrar el mejor de la generación actual
    best_idx = np.argmax(fos)
    best_individual = pop[best_idx]
    best_fitness = fos[best_idx]
    best_weight = best_individual.dot(p)
    
    # Si el mejor de la generación actual es mejor que el incumbente, actualizamos el incumbente
    if best_fitness > incumbent_fitness:
        incumbent_fitness = best_fitness
        incumbent_weight = best_weight
        incumbent_individual = best_individual
    
    best_in_gen.append(incumbent_fitness)

    population_data.append({
        'Generación': gen + 1,
        'Incumbente Fitness': incumbent_fitness,
        'Peso Asociado': incumbent_weight,
        'Incumbente Individuo': incumbent_individual
    })

    # Graficar el incumbente en cada generación
    if gen % 50 == 0:
        plt.scatter(gen, incumbent_fitness, c='b')

# Mostrar el gráfico final del incumbente
plt.xlabel('Generaciones')
plt.ylabel('Incumbente Fitness')
plt.title('Evolución del Incumbente')
plt.show()

print("Mejor solución (incumbente) encontrada:")
print(f"Individuo Incumbente: {incumbent_individual}")
print(f"Incumbente fitness: {incumbent_fitness}")
print(f"Peso asociado: {incumbent_weight}")

# Mostrar la tabla de evolución del incumbente
df = pd.DataFrame(population_data)
print(df)