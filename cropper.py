import numpy as np 
import cv2
from PIL import Image
import imutils


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







#frame = np.array(Image.open("/Users/drewscheffer/Downloads/red.jpeg"))
frame = cv2.imread("/Users/drewscheffer/Desktop/output/seg_COCO_val2014_000000043816.jpg.jpg")

# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([100,50,50])
upper_blue = np.array([130,255,255])
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



cv2.imshow('frame',frame)
cv2.imshow('mask',mask)


while True:
	k = cv2.waitKey(5) 
	if k == 27:
		break

