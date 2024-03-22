# This is a ridiculous attempt to train a model to recognize if a game is winnable
# from the initial board deal.  There's a number of reasons why this almost certainly won't work
# so I'm expecting an accuracy of around 50%.  
#
# It will use 10K+ games played by the solver as training/testing data.

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2
EPOCHS = 10


gameData = pd.read_csv('./games.csv') # setup a pandas dataFrame on the csv


results = gameData.pop('result') # pop off the results column, as that's our traing target


gameData = np.array(gameData)
print(gameData)

# normalizer = tf.keras.layers.Normalization(axis=-1)
# normalizer.adapt(tensor)

normalize = layers.Normalization()
normalize.adapt(gameData)

model = tf.keras.Sequential([
	normalize,
	layers.Dense(64),
	layers.Dense(1)
])

model.compile(loss = tf.keras.losses.MeanSquaredError(),
					  optimizer = tf.keras.optimizers.Adam(),metrics=['accuracy'])


history = model.fit(gameData, results, epochs=EPOCHS,validation_split=0.2)

# plot the accuracy of the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('solverModel.keras')