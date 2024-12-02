#Made by B23-215 group by Allabergenov Maksat and Denisov Aleksei

import tensorflow as tf
import keras
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up parameters
L = 10.0  # Domain [-L, L]
T = 5.0  # Time range [0, T]
N_x = 100  # Number of spatial points
N_t = 100  # Number of temporal points

# Initial condition function f(x)
def initial_condition(x):
    res = 2 / (tf.cosh(x) ** 2)
    return res

def createModel():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((2, )),
            tf.keras.layers.Dense(units = 50, activation = 'tanh'),
            tf.keras.layers.Dense(units = 50, activation = 'tanh'),
            tf.keras.layers.Dense(units = 50, activation = 'tanh'),
            tf.keras.layers.Dense(units = 1),
        ]
    )
    return model

# Prepare training data
x = tf.linspace(-L, L, N_x)
t = tf.linspace(0.0, T, N_t)
X, T = tf.meshgrid(x, t)
x_train = tf.reshape(X, (-1, 1))
t_train = tf.reshape(T, (-1, 1))

# Define the neural network
class PINN(tf.Module):
    def __init__(self, model) -> None:
        super(PINN, self).__init__()
        self._model = model
        self._train_loss = None
        self._model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.001), loss="mean_squared_error")

    def built(self):
        pass

    def __call__(self, x):
        return self._model(x)

    def loss_fn(self, x, t):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            tape1.watch(t)
        
            # Compute u
            u = self._model(tf.concat([x, t], axis=1))

            # Compute first derivatives
            u_x = tape1.gradient(u, x)
            u_t = tape1.gradient(u, t)

            # Second and third derivatives
            u_xx = tape1.gradient(u_x, x)
            u_xxx = tape1.gradient(u_xx, x)

        # PINN loss (KdV equation residual)
        loss_pde = tf.reduce_mean(tf.square(u_t + 6 * u * u_x + u_xxx))

        # Initial condition loss
        u0_pred = self._model(tf.concat([x, tf.zeros_like(t)], axis=1))
        u0_actual = initial_condition(x)
        loss_ic = tf.reduce_mean(tf.square(u0_pred - u0_actual))

        # Boundary conditions loss
        u_L_t = self._model(tf.concat([-L * tf.ones_like(t), t], axis=1))
        u_R_t = self._model(tf.concat([L * tf.ones_like(t), t], axis=1))
        loss_bc = tf.reduce_mean(tf.square(u_L_t)) + tf.reduce_mean(tf.square(u_R_t))

        # Total loss
        loss = loss_pde + loss_ic + loss_bc
        del tape1
        return loss

    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(x_train, t_train)
            gradients = tape.gradient(loss, self._model.trainable_variables)
            self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
            return loss

model = PINN(model = createModel())
# Training loop
loss_record = list([])
epochs = 1000
pbar = tqdm(range(epochs))
for epoch in pbar:
    loss = model.train_step()
    loss_record.append(loss.numpy())
    if (epoch % 10 == 0):
        pbar.set_description("Epoch: %d | Loss: %.6f | Training Progress" % (epoch, loss.numpy()))

y = model(tf.concat([x_train, 0.1 * tf.ones_like(t_train)], axis=1))
y_train = tf.reshape(y, (-1))

fig = plt.figure()
ax = plt.axes(xlim=(-L, L), ylim=(-1, 5))
line, = ax.plot([], [], 'o', lw=3)

def init():
    line.set_data([], [])
    return line,
def animate(i):
    i = i * 0.005
    y = model(tf.concat([x_train,  i * tf.ones_like(t_train)], axis=1))
    y_train = tf.reshape(y, (-1))
    line.set_data(x_train, y_train)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=250, interval=10, blit=True)

plt.show()
