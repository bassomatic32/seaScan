#  Given a file screenshot of a seahaven game, this script
# will pull out all the rank and suit images, and ask the user to identify each one
# these images will then be added to a categorical folder


import cv2
import Cards
import solver
import sys
import tensorflow.keras as keras
import tensorflow as tf
import cv2
import pathlib
import jsonpickle


data_dir = pathlib.Path('imgs').with_suffix('')

# I'm a bit surprised that the saved Tensorflow model doesn't save the classifcation labels.  So, we are going to just
# recreate them here
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(50, 50),
  batch_size=32)


classNames = train_ds.class_names
print(classNames)

# read the image as presented on the command line
img = cv2.imread(sys.argv[1])

# Create the board from the image.  This will isolate the cards in the image AND determine their semantic location
# on the board.  ( that is, is the card in the goals, the cells, or the tableau, etc.)
cardImages = Cards.findCards(img)

# load the Keras model
model = keras.models.load_model("seaModel.keras")

def detectImage(img) :

	#Resize to respect the input_shape
	inp = cv2.resize(img, (90 , 90 ))

	#Convert img to RGB
	rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

	rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
	#Add dims to rgb_tensor
	rgb_tensor = tf.expand_dims(rgb_tensor , 0)

	predictions = model.predict(rgb_tensor, steps=1,verbose=0)
	
	return classNames[predictions.argmax(axis=1)[0]]


# detect the rank and suit of each card, and set them into the card value
for c in cardImages:
	
	img = c.rankImage()	
	rank = detectImage(img)

	img = c.suitImage()
	suit = detectImage(img)

	c.setValue((rank,suit))
	c.printLocation()

print("Done Detecting")


# now, create the board, and put the cards on it
board = solver.Board()
Cards.placeCards(board,cardImages)

# and finally play the game
solver.playGameWithBoard(board)





