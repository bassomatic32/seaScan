import numpy as np
import cv2
import time
import math
from typing import List
from solver import Board,CardRank,CardSuit,Card


IMAGE_STANDARD_WIDTH = 300

CARD_MAX_AREA = 50000
CARD_MIN_AREA = 500

font = cv2.FONT_HERSHEY_SIMPLEX

def showImage(image):

	cv2.imshow("Display window", image)
	k = cv2.waitKey(0) # Wait for a keystroke in the window
	return chr(k)


def preProcessImage(image):
	"""Returns a grayed, blurred, and thresholded image."""

	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(3,3),0)
	retVal,thresh = cv2.threshold(blur,178,255,cv2.THRESH_BINARY)
	
	return thresh




class CardImage:

	def __init__(self,loc,img):
		self.loc = loc
		self.img = img
		self.verticalBucket = 0
		self.horizontalBucket = 0

	def rankImage(self):
		"""Return the Rank part of the card image"""

		center = int( self.img.shape[1] / 2 )
		return self.img[:,:center]

	def suitImage(self):
		"""Return the Suit part of the card image"""
		center = int (self.img.shape[1] / 2)
		return self.img[:,center:]
	
	def setValue(self,value):
		self.rank = value[0]
		self.suit = value[1]

	def __getstate__(self):
		return self.rank+self.suit
	
	def __str__(self):
		return self.rank+self.suit
	
	def printLocation(self):
		print(self.rank+self.suit,self.loc)
	
	
		

	

def placeCards(board:Board,cards:List[CardImage]) -> None:
	"""Place an unsorted list of cards into their appropriate structurual representations on the game board"""

	# We are going to do analysis of the X and Y values to get a list of cardinal bucket values for the location
	xValues = list(map(lambda c:c.loc[0],cards))
	yValues = list(map(lambda c:c.loc[1],cards))
	print('yValues',yValues)

	minY = np.min(yValues)
	topPartLimit = ((np.max(yValues) - minY) * .2) + minY 
	# topPartLimit is our border between the goal/cell area and the tableau

	tableauCards = sorted(filter(lambda c: c.loc[1] > topPartLimit,cards),key=lambda c:c.loc[1]) # tableu cards are sorted on the y-axis.  This will come into play later
	topCards = sorted(filter(lambda c: c.loc[1] <= topPartLimit,cards),key=lambda c:c.loc[0])
	# now we have segregated the cards into cards on the top, and cards in the tableu

	xMedian = np.median(xValues) # our midpoint along the x-axis

	goals:List[CardImage] = []
	cells:List[CardImage] = []
	tableaus:List[CardImage] = []
	
	# goals are on left of mid-point, and cells on right.  Order doesn't matter here se we just segregate these as we find them
	for c in topCards:
		if c.loc[0] < xMedian:
			goals.append(c)
		else:
			cells.append(c)	


	
	yMedian = np.average(list(filter(lambda y: y > topPartLimit,yValues))) # since tableau stacks are 5x2, we can segregate along the horizontal midpoint		
	print('yMedian',yMedian)
	# setup our tableu colums		
	tabColumns = []
	for i in range(5):
		tabColumns.append([])

	xValues = sorted(np.unique(xValues)) 
	nextX = xValues[1]
	index = 1
	while (nextX - xValues[0] < 15):
		index += 1
		nextX = xValues[index]

	xStep = nextX - xValues[0]
	xBase = xValues[0]+(xStep/2)

	# print('xValues',xValues)
	# print('xStep',xStep)
	# print('xBase',xBase)
	
	# use a bit of math to find the appropriate column for our tableu cards.  Since cards are already sorted along y-axis, the will go in order bottom to top.
	for c in tableauCards:
		index = math.ceil((c.loc[0] - xBase) / xStep)	
		# print(index)		
		tabColumns[index].append(c)		
	# each full column is now ordered.  We now need to segregate along mid-point
	

	for col in tabColumns:
		tableaus.append(list(filter(lambda c: c.loc[1] < yMedian,col)))
		tableaus.append(list(filter(lambda c: c.loc[1] >= yMedian,col)))


	# now the board has fully organized images in their understood locations.	
	# let's convert them to proper Cards and load the given board with them
	goals = list(map(lambda c:Card.cardFromStr(c.rank,c.suit),goals))
	cells = list(map(lambda c:Card.cardFromStr(c.rank,c.suit),cells))
	stacks = []
	for stack in tableaus:
		stacks.append(list(map(lambda c:Card.cardFromStr(c.rank,c.suit),stack)))

	board.loadCards(cells,goals,stacks)
		
		


def findCards(originalImage) -> List[CardImage]:
	"""Finds all card-sized contours in a thresholded  image.
	Returns the filtered contours"""

	# resize image to a standard size, maintaining aspect
	r =  IMAGE_STANDARD_WIDTH / originalImage.shape[1]
	dim = (IMAGE_STANDARD_WIDTH, int(originalImage.shape[0] * r))
	originalImage = cv2.resize(originalImage,dim)

	
	image = preProcessImage(originalImage)
	# Find contours and sort their indices by contour size
	contours,hier = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# If there are no contours, do nothing
	if len(contours) == 0:
		return [], []
	

	# for i in index_sort:
		# cv2.convexHull(contours[i], contours[i], False, True )
		# sortedContours.append(contours[i])

	filteredContours = []		
	for c in contours:
		cv2.convexHull(c, c, False, True )
		size = cv2.contourArea(c)
		# print("size; ",size)
		if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)):
			filteredContours.append(c)

	boxes = list(map(lambda c: np.int0(cv2.boxPoints(cv2.minAreaRect(c))),filteredContours))


	dst = originalImage.copy()

	cv2.drawContours(dst,boxes, -1, (0,255,0), 3)

	# showImage(dst)

	cards:CardImage = []

	# create a list of cards with their image and location.  We also clip off most of the card as we only need the portion with the rank and suit
	yLimit = min(map(lambda box:max(map(lambda b:b[1],box))-min(map(lambda b:b[1],box)), boxes))
	for box in boxes:
		minY = min(map(lambda b:b[1],box))
		maxY = max(map(lambda b:b[1],box))-minY
		maxY = min(maxY,yLimit)
		minX = min(map(lambda b:b[0],box))
		maxX = max(map(lambda b:b[0],box))
		
		boxImage = originalImage[minY:minY+maxY,minX:maxX]
		card = CardImage((minX,minY),boxImage)
		cards.append(card)

	return cards


# def createBoard(img) -> Board:
	
# 	# showImage(image)
# 	cards = findCards(img)

# 	board = Board()
# 	board.placeCards(cards) # put the cards into their semantic location on the board

# 	return board



