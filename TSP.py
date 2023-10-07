
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Load the cities data
cities = np.loadtxt('cities1000.txt', dtype=int, delimiter=',')
num_cities = len(cities)

# Calculate the distance matrix
distances = euclidean_distances(cities)

# Define the fitness function
def fitness(individual):
    return np.sum(distances[np.arange(num_cities), individual])

# Define the selection function
def selection(population):
    return random.choices(population, k=2)

# Define the crossover function
def crossover(parents):
    crossover_point = random.randint(1, num_cities - 1)
    offspring = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
    return offspring

# Define the mutation function
def mutation(individual):
    mutation_point1 = random.randint(0, num_cities - 1)
    mutation_point2 = random.randint(0, num_cities - 1)
    individual[mutation_point1], individual[mutation_point2] = individual[mutation_point2], individual[mutation_point1]
    return individual

# Define the genetic algorithm function
def genetic_algorithm(population_size, num_generations, mutation_rate):
    population = [random.sample(range(num_cities), num_cities) for _ in range(population_size)]
    best_individual = min(population, key=fitness)
    best_fitness = fitness(best_individual)

    for generation in range(num_generations):
        new_population = []

        for _ in range(population_size):
            parents = selection(population)
            offspring = crossover(parents)
            offspring = mutation(offspring)
            new_population.append(offspring)

        population = new_population
        current_individual = min(population, key=fitness)
        current_fitness = fitness(current_individual)

        if current_fitness < best_fitness:
            best_individual = current_individual
            best_fitness = current_fitness

    return best_individual, best_fitness

# Run the genetic algorithm
population_size = 200
num_generations = 1000
mutation_rate = 0.1
best_individual, best_fitness = genetic_algorithm(population_size, num_generations, mutation_rate)

print(f"Best individual: {best_individual}")
print(f"Shortest Path: {best_fitness}")

# Plot the best individual
plt.figure(figsize=(10, 10))
plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='Cities')
plt.plot(cities[best_individual, 0], cities[best_individual, 1], c='red', label='Best individual')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Traveling Salesman Problem - Genetic Algorithm')
plt.legend()
plt.show()