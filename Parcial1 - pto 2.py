import numpy as np
import matplotlib.pyplot as plt
import time  

np.random.seed(0)  # Semilla

# Función para generar instancias del problema de la mochila
def knapsack_instances(n, w_range=(1, 20), b_range=(10, 100), alpha=0.5):
    """
    Genera una instancia del problema de la mochila.
    Parámetros:
    - n (int): Número de ítems.
    - w_range (tuple): Rango de pesos de los ítems.
    - b_range (tuple): Rango de beneficios de los ítems.
    - alpha (float): Factor para definir la capacidad de la mochila relativa a la suma de pesos.
    
    Retorna:
    - p (np.array): Pesos de los ítems.
    - b (np.array): Beneficios de los ítems.
    - C (float): Capacidad de la mochila.
    """
    p = np.random.randint(w_range[0], w_range[1] + 1, n)  # Pesos
    b = np.random.randint(b_range[0], b_range[1] + 1, n)  # Beneficios
    C = alpha * np.sum(p)  # Capacidad de la mochila
    
    # Imprimir los parámetros generados
    print(f"Instancia generada con {n} ítems:")
    print(f"Pesos (p): {p}")
    print(f"Beneficios (b): {b}")
    print(f"Capacidad de la mochila (C): {C}\n")
    
    return p, b, C

# Función de evaluación (fitness) con penalización
def fitness(xi, b, p, C):
    wi = xi.dot(p)  # Peso total
    bi = xi.dot(b)  # Beneficio total
    if wi > C:
        return bi - (wi - C) * 0.5  # Penalización por violar la capacidad
    return bi

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

# Función de reparación para soluciones no factibles
def repair(xi, b, p, C, d):
    wi = xi.dot(p)
    while wi > C:
        pos = np.random.randint(0, d)
        xi[pos] = 0
        wi = xi.dot(p)
    bi = xi.dot(b)
    return xi, wi, bi

# Algoritmo genético con penalización y selección por ruleta
def genetic_algorithm_penalty(n_items, alpha=0.5, N=100, generations=500, cross_rate=0.8, mutation_rate=0.02):
    # Medir el tiempo de inicio
    start_time = time.time()

    # Generar instancia
    p, b, C = knapsack_instances(n_items, alpha=alpha)
    
    # Inicializar la población
    pop = np.random.randint(0, 2, (N, len(b)))
    
    # Evaluar la población inicial
    fos = np.array([fitness(ind, b, p, C) for ind in pop])
    
    # Variables para almacenar el mejor resultado
    best_in_gen = []
    incumbent_fitness = -np.inf  # Fitness incumbente
    incumbent_individual = np.zeros(len(b))  # Individuo incumbente
    
    # Ciclo de generaciones
    for gen in range(generations):
        new_population = []
        
        for _ in range(N):
            # Selección de padres por ruleta
            parent1_idx = roulette_selection(fos)
            parent2_idx = roulette_selection(fos)
            p1 = pop[parent1_idx]
            p2 = pop[parent2_idx]

            # Cruce
            child = crossover(p1, p2, cross_rate)

            # Mutación
            mutate(child, mutation_rate)

            # Evaluación del hijo
            child_fitness = fitness(child, b, p, C)
            new_population.append((child, child_fitness))
        
        # Actualizar población y función objetivo
        pop = np.array([ind for ind, fit in new_population])
        fos = np.array([fit for ind, fit in new_population])

        # Encontrar el mejor de la generación
        best_idx = np.argmax(fos)
        best_individual = pop[best_idx]
        best_fitness = fos[best_idx]

        # Actualizar incumbente si se mejora
        if best_fitness > incumbent_fitness:
            incumbent_fitness = best_fitness
            incumbent_individual = best_individual

        best_in_gen.append(incumbent_fitness)

    # Mostrar resultados finales
    print("Mejor solución encontrada (Penalización):")
    print(f"Beneficios: {incumbent_fitness}")
    print(f"Peso: {incumbent_individual.dot(p)}")
    print(f"Individuo: {incumbent_individual}")
    
    # Medir el tiempo de fin y calcular duración
    end_time = time.time()
    duration = end_time - start_time
    print(f"Tiempo de ejecución: {duration:.2f} segundos\n")
    
    # Graficar evolución
    plt.plot(best_in_gen)
    plt.title(f'Evolución de la mejor solución - Penalización ({n_items} ítems)')
    plt.xlabel('Generaciones')
    plt.ylabel('Beneficio incumbente')
    plt.show()

# Algoritmo genético con reparación y selección por torneo
def genetic_algorithm_repair(n_items, alpha=0.5, N=100, generations=500, cross_rate=0.8, mutation_rate=0.02):
    # Medir el tiempo de inicio
    start_time = time.time()

    # Generar instancia
    p, b, C = knapsack_instances(n_items, alpha=alpha)
    
    # Inicializar la población
    pop = np.random.randint(0, 2, (N, len(b)))
    fos = pop.dot(b)  # Beneficios iniciales
    ws = pop.dot(p)  # Pesos iniciales

    # Reparar población inicial
    for i in range(N):
        if ws[i] > C:
            pop[i], ws[i], fos[i] = repair(pop[i], b, p, C, len(b))
    
    # Variables para almacenar el mejor resultado
    best_in_gen = []
    idx_best = np.argmax(fos)

    # Ciclo de generaciones
    for gen in range(generations):
        # Selección por torneo
        idxps = np.random.choice(N, 4, replace=False)
        idxp1 = idxps[0] if fos[idxps[0]] > fos[idxps[1]] else idxps[1]
        idxp2 = idxps[2] if fos[idxps[2]] > fos[idxps[3]] else idxps[3]

        # Cruce
        child = crossover(pop[idxp1], pop[idxp2], cross_rate)

        # Reparar hijo
        child, wc, bc = repair(child, b, p, C, len(b))

        # Mutación
        mutate(child, mutation_rate)
        child, wc, bc = repair(child, b, p, C, len(b))  # Reparar de nuevo si es necesario

        # Reemplazo del peor en la población
        idx_worst = np.argmin(fos)
        if bc > fos[idx_worst]:
            pop[idx_worst] = child
            fos[idx_worst] = bc
            ws[idx_worst] = wc

        # Actualizar incumbente
        idx_best_gen = np.argmax(fos)
        if fos[idx_best_gen] > fos[idx_best]:
            idx_best = idx_best_gen

        best_in_gen.append(fos[idx_best])

    # Mostrar resultados finales
    print("Mejor solución encontrada (Reparación):")
    print(f"Beneficios: {fos[idx_best]}")
    print(f"Peso: {pop[idx_best].dot(p)}")
    print(f"Individuo: {pop[idx_best]}")
    
    # Medir el tiempo de fin y calcular duración
    end_time = time.time()
    duration = end_time - start_time
    print(f"Tiempo de ejecución: {duration:.2f} segundos\n")

    # Graficar evolución
    plt.plot(best_in_gen)
    plt.title(f'Evolución de la mejor solución - Reparación ({n_items} ítems)')
    plt.xlabel('Generaciones')
    plt.ylabel('Beneficio incumbente')
    plt.show()

# Correr los experimentos
genetic_algorithm_penalty(200, alpha=0.5)
genetic_algorithm_repair(200, alpha=0.5)
genetic_algorithm_penalty(500, alpha=0.5)
genetic_algorithm_repair(500, alpha=0.5)
genetic_algorithm_repair(750, alpha=0.5)
genetic_algorithm_penalty(1000, alpha=0.5)
