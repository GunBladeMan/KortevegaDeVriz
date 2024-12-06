import tensorflow as tf
import keras
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

model = keras.layers.TFSMLayer("model_5l_50n", call_endpoint="serving_default")

L = 10.0  # Domain [-L, L]
T = 10.0  # Time range [0, T]
N_x = 100  # Number of spatial points
N_t = 100  # Number of temporal points

x = tf.linspace(-L, L, N_x)
t = tf.linspace(0.0, T, N_t)
X, T = tf.meshgrid(x, t)
x_train = tf.reshape(X, (-1, 1))
t_train = tf.reshape(T, (-1, 1))

y = model(tf.concat([x_train, 0.1 * tf.ones_like(t_train)], axis=1))
y_train = tf.reshape(y["output_0"], (-1))

fig = plt.figure()
ax = plt.axes(xlim=(-L, L), ylim=(-1, 5))
line, = ax.plot([], [], 'o', lw=3)

def init():
    line.set_data([], [])
    return line,
def animate(i):
    i = i * 0.005
    y = model(tf.concat([x_train,  i * tf.ones_like(t_train)], axis=1))
    y_train = tf.reshape(y["output_0"], (-1))
    line.set_data(x_train, y_train)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=250, interval=10, blit=True)

plt.show()