"""Nature Evolution Strategy Example1
Natural Evolution Strategies (NES) are a family of evolution strategies which iteratively
update a search distribution by using an estimated gradient on its distribution parameters.

Here we use the normal distribution as the setting.
Finally, we through Visualization to understand the process.

Reference paper: http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf

key points:
1. Define fitness
2. Choose the distribution
"""

import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
from tensorflow.contrib.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1
POP_SIZE = 5
LR = 0.1
N_GENERATION = 40

tf.set_random_seed(1)

def F(x):
    return x**2 - 2*x + 1

def get_fitness(value):
    return -value

mean = tf.Variable(tf.constant(-30.), dtype=tf.float32)
sigma = tf.Variable(tf.constant(1.), dtype=tf.float32)
N_dist = Normal(loc=mean, scale=sigma)
make_kids = N_dist.sample([POP_SIZE])

tfkids = tf.placeholder(tf.float32, [POP_SIZE, DNA_SIZE])
tfkids_fit = tf.placeholder(tf.float32, [POP_SIZE])
loss = -tf.reduce_mean(N_dist.log_prob(tfkids) * tfkids_fit)
train_op = tf.train.GradientDescentOptimizer(LR).minimize(loss)

x = np.linspace(-70, 70, 100)
plt.plot(x, F(x))
plt.xlim(-70, 70)
plt.ylim(-100, 1000)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

plt.ion()
for g in range(N_GENERATION):
    kids = sess.run(make_kids)
    kids_fit = get_fitness(F(kids))
    sess.run(train_op, feed_dict={tfkids: np.reshape(kids, [-1, 1]), tfkids_fit: kids_fit})

    if "plot_points" in globals():
        plot_points.remove()
        plot_text1.remove()
        plot_text2.remove()
        plot_text3.remove()

    plot_points = plt.scatter(np.clip(kids, -70, 70), np.clip(F(kids), -100, 1000),
                              s=100, c="k", alpha=0.5)
    plot_text1 = plt.text(-60, 200, "Generation:{}".format(g))
    plot_text2 = plt.text(-60, 100, "Mean:{}".format(sess.run(tf.round(mean))))
    plot_text3 = plt.text(-60, 0, "Sigma:{}".format(sess.run(tf.round(sigma))))
    plt.pause(0.5)

plt.ioff()
plt.show()
