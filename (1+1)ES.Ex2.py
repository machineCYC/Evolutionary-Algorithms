"""(1+1) Evolution Strategy Example2
Visualize (1+1) Evolution Strategy to find the biggest value for target function.
(1+1) Evolution Strategy is a variation of Evolution Strategy

(1+1) Evolution Strategy only has a DNA in population called parent, use parent
to generate kid through DNA_var.

Key points
1. Define DNA
Consider the DNA size is 2 and DNA is real number, like [1.221, 2.14]
2. Define fitness
The basic idea is the DNA has the biggest value of target function that has the good fitness
"""

import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1
DNA_BOUND = [0, 10]
N_GENERATIONS = 150
DNA_var = 10.

np.random.seed(0)

# initial population
PARENT = DNA_BOUND[1] * np.random.rand(DNA_SIZE)

# target function
def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

# calculate the value of fitness
def get_fitness(value):
    return value

# generation kids(new DNA) from parent base on mutate
def make_kids(parent):
    # mutate
    kid = parent + DNA_var * np.random.randn(DNA_SIZE)
    kid = np.clip(kid, *DNA_BOUND)
    return kid

# from parent and kid to select the bigger value of fitness to become a new parent
def kill_kids(kid, parent):
    global DNA_var
    f_k = get_fitness(F(kid))
    f_p = get_fitness(F(parent))

    # 1/5 successful rule
    p_target = 1/5
    if f_k > f_p:
        p_s = 1
        parent = kid
    else:
        p_s = 0
    DNA_var *= np.exp((1/3)*(p_s - p_target)/(1 - p_target))
    return parent

# dynamic plot
plt.ion()
x = np.linspace(*DNA_BOUND, 200)
plt.plot(x, F(x))
plt.xlim(-1, 11)
plt.ylim(-20, 21)

for g in range(N_GENERATIONS):

    kid = make_kids(PARENT)
    PARENT = kill_kids(kid, PARENT)

    if "plot_points_k" in globals():
        plot_points_k.remove()
        plot_points_p.remove()
        plot_text_g.remove()
        plot_text_v.remove()

    plot_points_k = plt.scatter(kid, F(kid), c="r", s=200, alpha=0.5)  # kid
    plot_points_p = plt.scatter(PARENT, F(PARENT), c="blue", s=200, alpha=0.5)  # parent
    plot_text_g = plt.text(0, 15, "Generation: %i" % g)
    plot_text_v = plt.text(0, 13, "DNA_var: %f" % DNA_var)
    plt.pause(0.1)

plt.ioff()
plt.show()
