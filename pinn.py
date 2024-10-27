import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set up parameters
L = 1.0  # Domain [-L, L]
T = 1.0  # Time range [0, T]
N_x = 100  # Number of spatial points
N_t = 100  # Number of temporal points

# Define the neural network
class PINN(tf.keras.Model):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='tanh') for layer in layers]
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

# Initialize the network with 3 hidden layers of 50 neurons each
layers = [50, 50, 50]
pinn = PINN(layers)

# Define the loss function
def loss_fn(x, t):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(x)
        tape1.watch(t)
        
        # Compute u
        u = pinn(tf.concat([x, t], axis=1))
        
        # Compute first derivatives
        u_x = tape1.gradient(u, x)
        u_t = tape1.gradient(u, t)
    
        # Second and third derivatives
        u_xx = tape1.gradient(u_x, x)
        u_xxx = tape1.gradient(u_xx, x)
    
    # PINN loss (KdV equation residual)
    loss_pde = tf.reduce_mean(tf.square(u_t + 6 * u * u_x + u_xxx))
    
    # Initial condition loss
    u0_pred = pinn(tf.concat([x, tf.zeros_like(t)], axis=1))
    u0_actual = initial_condition(x)
    loss_ic = tf.reduce_mean(tf.square(u0_pred - u0_actual))
    
    # Boundary conditions loss
    u_L_t = pinn(tf.concat([-L * tf.ones_like(t), t], axis=1))
    u_R_t = pinn(tf.concat([L * tf.ones_like(t), t], axis=1))
    loss_bc = tf.reduce_mean(tf.square(u_L_t)) + tf.reduce_mean(tf.square(u_R_t))
    
    # Total loss
    loss = loss_pde + loss_ic + loss_bc
    return loss

# Initial condition function f(x)
def initial_condition(x):
    return tf.sin(np.pi * x)  # For example, a sine wave

# Prepare training data
x = tf.linspace(-L, L, N_x)
t = tf.linspace(0.0, T, N_t)
X, T = tf.meshgrid(x, t)
x_train = tf.reshape(X, (-1, 1))
t_train = tf.reshape(T, (-1, 1))

# Training function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss = loss_fn(x_train, t_train)
        gradients = tape.gradient(loss, pinn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pinn.trainable_variables))
        return loss

# Training loop
epochs = 1000
for epoch in range(epochs):
    loss = train_step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
