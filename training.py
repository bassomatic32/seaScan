import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib


data_dir = pathlib.Path('imgs').with_suffix('')

print(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Image Count: ",image_count)


batch_size = 32
img_height = 90
img_width = 90

# I hate these names.  Training data-set
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Note: image size constrains the size of the image to what is given
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print('Class Names')
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))


# uncomment the following to see sample images
# for images, labels in train_ds.take(1):
# 	for i in range(9):
# 		print(class_names[labels[i]])
# 		ax = plt.subplot(3, 3, i + 1)
# 		plt.imshow(images[i].numpy().astype("uint8"))
# 		plt.title(class_names[labels[i]])
# 		plt.axis("off")

# plt.show()

# shape of the training tensors
for image_batch, labels_batch in train_ds:
	print(image_batch.shape)
	print(labels_batch.shape)
	break


# adding caching and prefetching on datasets to improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# normalize the layers to values between 0..255
# normalization_layer = layers.Rescaling(1./255) # we are configuring a keras.layer ( see import ).  

# test the normalization.  Create a normalzied data set using our created normalization layer ( we don't need this, so its getting commented out).  Instead, we'll put it in the model which is a better apporach
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds)) # get the first batch
# first_image = image_batch[0] # and the first item of that batch
# # Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

num_classes = len(class_names) # I hate snake case

# Funny name, data augmentation.  We are basically adding some noise to the input images to reduce over-fitting.  In a way, gives us more training data.
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model = Sequential([
    # layers.RandomRotation(0.1),
    # layers.RandomZoom(0.1),
	
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), # normalize the RGB values to 0..1 instead of 0..255
	layers.Conv2D(16, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(64 , 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Dropout(0.2),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(num_classes)
])

model.compile(optimizer='adam',
			  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			  metrics=['accuracy'])

model.summary()


epochs=15
# train the model with the training data set and the validation data set
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# plot the accuracy of the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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

model.save('seaModel.keras')

