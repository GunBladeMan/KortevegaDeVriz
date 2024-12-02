import tensorflow as tf
import keras
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

L = 10.0  # Domain [-L, L]
T = 5.0  # Time range [0, T]


#loaded_model = keras.saving.load_model("weights.keras")

fig = plt.figure()
ax = plt.axes(xlim=(-L, L), ylim=(-1, 5))

line1, = ax.plot([], [], lw=3)
line2, = ax.plot([], [], lw=3)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2,
def animate(i):
    i = i * 0.002
    x = tf.linspace(-L, L, 1000)
    y = 2 / ((tf.cosh(x - 4 * i)) ** 2)
    line1.set_data(x, y)
    y = i * x
    line2.set_data(x, y)
    return line1, line2,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=250, interval=10, blit=True)


plt.show()
