"""Microbial Genetic Algorithm Example1
Visualize Microbial Genetic Algorithm to find the biggest value for target function.

The difference between MGA and GA is MGA(without replacement) choose two DNA and use good DNA to improve bad DNA,
but GA(with replacement) through "select function" to choose the good DNA.

Key points
1. Define DNA
Consider the DNA size is 5, the DNA is defined like [0, 1, 1, 0, 1]
2. Define fitness
The basic idea is the DNA has the biggest value of target function that has the good fitness
"""

import numpy as np
import matplotlib.pyplot as plt

POP_SIZE = 20
DNA_SIZE = 10
N_GENERATION = 200
CROSS_RATE = 0.8
MUTATION_RATE = 0.01
X_BOUND = [0, 10]

POP = np.random.randint(0, 2, size=(POP_SIZE, DNA_SIZE))

# target function
def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

# translate the binary DNA to range(0, 10)
def translateDNA2x(pop):
    return pop.dot(2 ** np.arange((DNA_SIZE-1), -1, -1)) / (2 ** DNA_SIZE - 1) * X_BOUND[1]

# calculate the value of fitness
def get_fitness(value):
    return value

# improved loser DNA with winner DNA
def crossover(loser_winner, cross_rate, DNA_SIZE):
    cross_idx = np.empty(DNA_SIZE).astype(np.bool)
    for c in range(DNA_SIZE):
        cross_idx[c] = True if np.random.rand() < cross_rate else False

    loser_winner[0, cross_idx] = loser_winner[1, ][cross_idx]  # assign winners genes to loser
    return loser_winner

# mutation for loser
def mutate(loser_winner, DNA_SIZE, mutation_rate):
    mutate_idx = np.empty(DNA_SIZE).astype(np.bool)
    for m in range(DNA_SIZE):
        mutate_idx[m] = True if np.random.rand() < mutation_rate else False
    loser_winner[0, mutate_idx] = ~loser_winner[0, mutate_idx].astype(np.bool)  # flip values in mutation points
    return loser_winner

# dynamic plot
plt.ion()
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))

for g in range(N_GENERATION):

    if "plot_points" in globals():
        plot_points.remove()
        plot_text.remove()

    plot_points = plt.scatter(translateDNA2x(POP), F(translateDNA2x(POP)), c="r", s=100, alpha=0.5)
    plot_text = plt.text(0, -10, "Generation: %i" % g)
    plt.pause(0.05)

    for _ in range(5):
        sub_pop_idx = np.random.choice(range(POP_SIZE), size=2, replace=False)
        sub_pop = POP[sub_pop_idx]
        sub_fitness = get_fitness(F(translateDNA2x(sub_pop)))
        loser_winner = sub_pop[np.argsort(sub_fitness)]  # the first is loser and second is winner
        loser_winner = crossover(loser_winner, CROSS_RATE, DNA_SIZE)
        loser_winner = mutate(loser_winner, DNA_SIZE, MUTATION_RATE)
        POP[sub_pop_idx] = loser_winner

plt.ioff()
plt.show()
