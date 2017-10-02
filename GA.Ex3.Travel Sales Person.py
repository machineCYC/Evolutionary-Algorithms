"""Genetic Algorithm Example3, Travel Sales Person
Visualize Genetic Algorithm to find the shortest route for travel sales problem.

Key points
1. Define DNA
Consider have 5 cities, the DNA is defined like [4, 2, 1, 0, 3]
2. Define fitness
The basic idea is the route has the shortest distance that has the good fitness
"""

import numpy as np
import matplotlib.pyplot as plt

N_CITIES = 20
POP_SIZE = 500
CROSS_RATE = 0.8
MUTATION_RATE = 0.02
N_GENERATIONS = 500

# generate the initial population DNA
POP = np.vstack([np.random.permutation(N_CITIES) for i in range(POP_SIZE)])

# the coordinate of each city
CITY_POSITION = np.random.rand(N_CITIES, 2)

# convert the DNA-route to corresponding coordinate
def translateDNA2xy(CITY_POSITION, POP):
    line_x = CITY_POSITION[:, 0][POP]  # the x coordinates of city route for each DNA
    line_y = CITY_POSITION[:, 1][POP]  # the y coordinates of city route for each DNA
    return line_x, line_y

# calculate the value of fitness and total distance
def get_fitness(line_x, line_y, N_CITIES):
    total_distance = np.sum(np.sqrt(np.square(np.diff(line_x)) + np.square(np.diff(line_y))), axis=1)
    fitness = np.exp(2 * N_CITIES / total_distance)  # use np.exp is to enhance the different with each distance
    return fitness, total_distance

# choose the good DNA from population
def select(fitness, POP, POP_SIZE):
    idx = np.random.choice(range(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / fitness.sum())
    return POP[idx]

# generate child from parent
def crossover(parent_F, POP, POP_SIZE, N_CITIES):
    if np.random.rand() < CROSS_RATE:
        idy = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, size=N_CITIES).astype(np.bool)  # choose crossover points
        keep_city = parent_F[~cross_points]
        swap_city = POP[idy, ~np.in1d(POP[idy, ], keep_city)]
        parent_F[:] = np.concatenate((keep_city, swap_city))
    return parent_F

# mutation in the child
def mutate(child, N_CITIES, MUTATION_RATE):
    for idz in range(N_CITIES):
        if np.random.rand() < MUTATION_RATE:
            idw = np.random.randint(0, N_CITIES, size=1)
            change_cityA, change_cityB = child[idw], child[idz]
            child[idw], child[idz] = change_cityB, change_cityA
    return child

# dynamic plot
plt.ion()

for g in range(N_GENERATIONS):
    line_X, line_Y = translateDNA2xy(CITY_POSITION, POP)
    fitness, total_distance = get_fitness(line_X, line_Y, N_CITIES)
    POP = select(fitness, POP, POP_SIZE)

    # plotting
    plt.cla()
    plt.scatter(CITY_POSITION[:, 0], CITY_POSITION[:, 1], c="k", s=200)
    best_route = np.argmax(fitness)
    plt.plot(line_X[best_route], line_Y[best_route], c="r")
    plt.text(0.1, 1.1, "Generation: {0}".format(g))
    plt.text(0.4, 1.1, "Total distance: {0}".format(np.round(total_distance[best_route], 3)))
    plt.xlim((0, 1.0))
    plt.ylim((0, 1.2))
    plt.pause(0.05)

    POP_copy = POP.copy()
    for parent_F in POP:
        child = crossover(parent_F, POP_copy, POP_SIZE, N_CITIES)
        child = mutate(child, N_CITIES, MUTATION_RATE)
        parent_F[:] = child

plt.ioff()
plt.show()
