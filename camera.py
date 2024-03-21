# import the opencv library 
import cv2 
import numpy as np
  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
	  
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 
	
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# blur = cv2.GaussianBlur(gray,(3,3),0)
	retVal,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
	contours,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
	cv2.drawContours(frame,contours, -1, (0,255,0), 2)	

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
	# Display the resulting frame 
	cv2.imshow('frame', frame) 
	  
	# the 'q' button is set as the 
	# quitting button you may use any 
	# desired button of your choice 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 