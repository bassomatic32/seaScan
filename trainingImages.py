#  Given a file screenshot of a seahaven game, this script
# will pull out all the rank and suit images, and ask the user to identify each one
# these images will then be added to a categorical folder


import cv2
import Cards
import sys
import os

img = cv2.imread(sys.argv[1])

from pathlib import Path

p = Path('imgs')
p.mkdir(exist_ok=True)




cards = Cards.findCards(img)

def getCategory(img) -> chr: 

	validCategories = ['a','2','3','4','5','6','7','8','9','t','j','q','k','h','d','s','c']
	while True:
		category = Cards.showImage(img)
		if category in validCategories :
			return category

def saveImageToCategory(img,category):
	category = category.upper()
	p = Path('imgs/'+category)
	p.mkdir(exist_ok=True)

	fileCount = len(list(Path('imgs/'+ category).iterdir()))
	fileName = 'imgs/'+category+'/'+str(fileCount+1)+'.jpg'
	print('Saving to file:',fileName)
	cv2.imwrite(fileName,img)


for c in cards:
	
	
	img = c.rankImage()
	category = getCategory(img)
	saveImageToCategory(img,category)

	img = c.suitImage()
	category = getCategory(img)
	saveImageToCategory(img,category)
