import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.stats import norm
from functools import partial
from scipy.optimize import root_scalar
from lifelines import WeibullAFTFitter
from lifelines import KaplanMeierFitter
# Define Git function
def Git(t, mu, y, u):
    return norm.cdf((t * mu + y) / np.sqrt(t)) - np.exp(-2 * y * mu) * norm.cdf((mu * t - y) / np.sqrt(t)) - u

# Set parameters
N = 1000
p = 2
q = 2
beta = np.ones(p)
alpha = -np.ones(q)
# Generate x and z
x = np.column_stack((np.ones(N), np.random.randint(0, 2, N)))
z = np.column_stack((np.ones(N), np.random.randint(0, 2, N)))

mu = x @ alpha
y = np.exp(z @ beta)
t = []
U = []
for i in range(N):
    u1 = np.random.uniform()
    Generate_fixed = partial(Git, mu=mu[i], y=y[i],u=u1)
    result=root_scalar(Generate_fixed, bracket=[0.1, 10000], method='brentq')
    t1=result.root
    t.append(t1)
    U.append(u1)
censor = np.random.uniform(5, 100000000000, N)
Y = np.minimum(censor, t)
C = (np.array(t) <= censor).astype(int)
data = pd.DataFrame({
    'time': t,
    'censor': C,
    'xs.2': x[:, 1],
    'zs.2': z[:, 1],
    'mu': mu,
    'y': y
})

data['xs'] = data['xs.2'].astype('category')
data['zs'] = data['zs.2'].astype('category')

# Split data
train_indices = np.random.choice(N, size=int(0.8 * N), replace=False)
test_indices = np.setdiff1d(np.arange(N), train_indices)

train = data.iloc[train_indices]
test = data.iloc[test_indices]

y_true = train[['time', 'censor']]
y_pred = train[['mu', 'y']]

import tensorflow as tf

def TR_loglik_continuous(y_true, y_pred):
    t = y_true[:, 0]
    c = y_true[:, 1]
    mu = y_pred[:, 0]
    y0 = y_pred[:, 1]
    s1 = (mu * t + y0) / tf.sqrt(t)
    s2 = (mu * t - y0) / tf.sqrt(t)
    f = tf.math.log(y0 + tf.keras.backend.epsilon()) - 1.5 * tf.math.log(t) - tf.math.square(y0 + mu * t) / (2 * t)
    S = tf.math.exp(-2 * y0 * mu)
    return tf.reduce_mean(-1 * (c * f + (1 - c) * S))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers

# Define custom activation functions
def act_exp(x):
    return tf.math.exp(x)

len_mu = 1
len_y0 = 1

# Define model
input_mu = Input(shape=(len_mu,), dtype='float32', name='input_mu')
input_y0 = Input(shape=(len_y0,), dtype='float32', name='input_y0')

output_mu = Dense(
    units=10,
    activation='tanh',
    kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05),
    bias_initializer=initializers.Constant(0)
)(input_mu)
output_mu = Dense(
    units=1,
    activation='linear',
    name='mu',
    kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05),
    bias_initializer=initializers.Constant(0)
)(output_mu)

output_y0 = Dense(
    units=1,
    activation=act_exp,
    name='y0',
    kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05),
    bias_initializer=initializers.Constant(1)
)(input_y0)

output_tr = Concatenate()([output_mu, output_y0])

tr_model = Model(inputs=[input_mu, input_y0], outputs=output_tr)

# Display model summary
tr_model.summary()

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.01,
    patience=10,
    restore_best_weights=True,
    verbose=0
)

# Compile the model
tr_model.compile(
    loss=TR_loglik_continuous,
    optimizer=Nadam(learning_rate=0.002)
)

# Prepare training data
x_train_mu = np.expand_dims(train['mu'].values, axis=1)
x_train_y0 = np.expand_dims(train['y'].values, axis=1)

# Train the model
history = tr_model.fit(
    [x_train_mu, x_train_y0],
    y_true.values,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Plot the training and validation loss
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


