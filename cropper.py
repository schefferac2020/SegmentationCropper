import numpy as np 
import cv2
from PIL import Image
import imutils
import os


def order_points_old(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def get_width_height(rect, area):
	#find the width
	width = rect[1][0] - rect[0][0] # set width to be diff of top of rect
	if (rect[1][0] - rect[2][0] > width):
		width = rect[1][0] - rect[2][0]

	height = rect[3][1] - rect[1][1]
	if (rect[2][1] - rect[1][1] > height):
		height = rect[2][1] - rect[1][1]

	return (int(width), int(height))





#frame = np.array(Image.open("/Users/drewscheffer/Downloads/red.jpeg"))
frame = cv2.imread("./output/seg_imp.jpg.jpg")

# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV

'''
LIGHT BLUE (Right bottom of leg):
---------------------------------
lower_blue = np.array([90, 100, 150])
upper_blue = np.array([110, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

DARK BLUE (Right top of leg)
---------------------------------
lower_blue = np.array([110, 150, 150])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

RED (Torso)
----------------------------------
mask1 = cv2.inRange(hsv, (0,100,120), (5,255,255))
mask2 = cv2.inRange(hsv, (175,100,120), (180,255,255))

## Merge the mask and crop the red regions
mask = cv2.bitwise_or(mask1, mask2 )

DARK GREEN (Left top of leg)
-----------------------------------
lower_blue = np.array([40, 160, 160])
upper_blue = np.array([80, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

LIGHT GREEN (Left bottom of leg)
-----------------------------------
lower_blue = np.array([30, 20, 200])
upper_blue = np.array([60, 130, 240])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

LIGHT PURPLE (right forearm)
-----------------------------------
lower_blue = np.array([145, 60, 20])
upper_blue = np.array([165, 250, 240])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

ORANGE (right upper arm)
-----------------------------------
lower_blue = np.array([5, 100, 75])
upper_blue = np.array([15, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

YELLOW (right hand and left upper arm)
-----------------------------------
lower_blue = np.array([20, 150, 25])
upper_blue = np.array([30, 255, 255])

HOT PINK (left forearm)
-----------------------------------
lower_blue = np.array([140, 150, 25])
upper_blue = np.array([160, 255, 255])

CYAN BLUE (left hand)
-----------------------------------
lower_blue = np.array([80, 100, 25])
upper_blue = np.array([95, 255, 255])


''' 
lower_blue = np.array([80, 100, 25])
upper_blue = np.array([95, 255, 255])
# Threshold the HSV image to get only blue colors

mask = cv2.inRange(hsv, lower_blue, upper_blue)



bluecnts = cv2.findContours(mask.copy(),
						cv2.RETR_EXTERNAL,
						cv2.CHAIN_APPROX_SIMPLE)[-2]

if len(bluecnts)>0:
	#blue_area = max(bluecnts, key=cv2.contourArea)
	#(xg,yg,wg,hg) = cv2.boundingRect(blue_area)
	#cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)
	colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))


	for (i, c) in enumerate(bluecnts):

		if cv2.contourArea(c) < 25:
			continue

		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

		print("Object #{}:".format(i + 1))
		rect = order_points_old(box)


		# show the re-ordered coordinates
		print(rect.astype("int"))
		print("")

		# loop over the original points and draw them
		for ((x, y), color) in zip(rect, colors):
			cv2.circle(frame, (int(x), int(y)), 5, color, -1)

		cv2.putText(frame, "Object #{}".format(i + 1),
			(int(rect[0][0] - 15), int(rect[0][1] - 15)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


		#This is where we use this information to crop out the body
		#parts from the real image before the segmentation process.

		#find the right file name
		filename = "";
		for fname in os.listdir("./input/"):
			print(fname)
			if "imp" in fname:
				filename = fname;
		orig_image = cv2.imread("./input/" + filename)

		w, h = get_width_height(rect, cv2.contourArea(c))
		y = int(rect[0][1])
		x = int(rect[0][0])
		print(x)
		print(y)
		## TODO - ACTUALLY CROP THE IMAGE HERE
		cropped_image = orig_image[y:y+h, x:x+w]

		#Write the cropped image to a file

		cv2.imwrite('./final/' + filename, cropped_image)



cv2.imshow('frame',frame)
cv2.imshow('mask',mask)


while True:
	k = cv2.waitKey(5) 
	if k == 27:
		break

