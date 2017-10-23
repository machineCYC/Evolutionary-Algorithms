"""Evolution Strategy Example1
Visualize Evolution Strategy to find the biggest value for target function.
The difference between ES and GA is ES use population to generate the kids. Define the
new population base on the top POP_SIZE fitness of mix population and kids. but GA through
choose two DNA from population to generate kid that will replace one of the parents.

Key points
1. Define DNA
Consider the DNA size is 2 and DNA is real number, like [1.221, 2.14]
2. Define fitness
The basic idea is the DNA has the biggest value of target function that has the good fitness
"""

import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1
POP_SIZE = 30
DNA_BOUND = [0, 10]
N_GENERATION = 100
KID_SIZE = 5

# initial population
POP = dict(DNA=DNA_BOUND[1] * np.random.rand(POP_SIZE, DNA_SIZE),
           DNA_Var=np.random.rand(POP_SIZE, DNA_SIZE))

# target function
def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(value):
    return value.flatten()

# generation kids(new DNA) from population base on crossover and mutate
def make_kids(n_kids, POP):
    # generate the empty kids holder
    kids = dict(DNA=np.empty((n_kids, DNA_SIZE)),
                DNA_Var=np.empty((n_kids, DNA_SIZE)))

    for kd, kv in zip(kids["DNA"], kids["DNA_Var"]):
        # crossover
        p_F, p_M = np.random.choice(range(POP_SIZE), size=2, replace=False)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)

        kd[cross_points] = POP["DNA"][p_F, cross_points]
        kd[~cross_points] = POP["DNA"][p_M, ~cross_points]

        kv[cross_points] = POP["DNA_Var"][p_F, cross_points]
        kv[~cross_points] = POP["DNA_Var"][p_M, ~cross_points]

        # mutate
        kv[:] = np.maximum(kv + (np.random.rand(*kv.shape) - 0.5), 0.)  # Variance must > 0
        kd += kv * np.random.randn(*kd.shape)
        kd[:] = np.clip(kd, *DNA_BOUND)
    return kids

# mix POP and kids and select the top POP_SIZE DNA
def kill_kids(kids, POP):
    # make pop and kids together
    for key in ["DNA", "DNA_Var"]:
        POP[key] = np.vstack((POP[key], kids[key]))

    fitness = get_fitness(F(POP["DNA"]))
    fitness_idx = np.argsort(fitness)  # return the indices from small to big

    # select the top POP_SIZE fitness
    for key in ["DNA", "DNA_Var"]:
        POP[key] = POP[key][fitness_idx[-POP_SIZE:]]
    return POP

# dynamic plot
plt.ion()
x = np.linspace(*DNA_BOUND, 200)
plt.plot(x, F(x))
plt.xlim(-1, 11)
plt.ylim(-20, 21)

for g in range(N_GENERATION):

    if "plot_points" in globals():
        plot_points.remove()
        plot_text.remove()

    plot_points = plt.scatter(POP["DNA"], F(get_fitness(POP["DNA"])), c="r", s=200, alpha=0.5)
    plot_text = plt.text(0, 10, "Generation: %d" % g)
    plt.pause(0.05)

    kids = make_kids(KID_SIZE, POP)
    POP = kill_kids(kids, POP)

plt.ioff()
plt.show()
