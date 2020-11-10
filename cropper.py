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


def get_width_height(rect, h, w):
	
	for i in range(0, 4, 1):
		if rect[i][0] < 0:
			rect[i][0] = 0;
		if rect[i][1] < 0:
			rect[i][1] = 0;
		if rect[i][0] > w -1:
			rect[i][0] = w -1
		if rect[i][1] > h -1:
			rect[i][1] = h -1


	furthest_top = rect[0][1]
	if (rect[1][1] < furthest_top):
		furthest_top = rect[1][1]

	furthest_bot = rect[2][1]
	if (rect[3][1] > furthest_bot):
		furthest_bot = rect[3][1]

	furthest_right = rect[1][0]
	if (rect[2][0] > furthest_right):
		furthest_right = rect[2][0]

	furthest_left = rect[0][0]
	if (rect[3][0] < furthest_left):
		furthest_left = rect[3][0]

	height = furthest_bot - furthest_top
	width = furthest_right - furthest_left


	return (int(furthest_left), int(furthest_top), int(width), int(height))
	
	
filters = [("right-bot-leg", np.array([90, 100, 150]), np.array([110, 255, 255])), \
		   ("right-top-leg", np.array([110, 150, 150]), np.array([130, 255, 255])), \
		   ("left-leg-top", np.array([40, 160, 160]), np.array([80, 255, 255])), \
		   ("left-leg-bot", np.array([30, 20, 200]), np.array([60, 130, 240])), \
		   ("right-forearm", np.array([145, 60, 20]), np.array([165, 250, 240])), \
		   ("right-upper-arm", np.array([5, 100, 75]), np.array([15, 255, 255])), \
		   ("right-hand-left-arm", np.array([20, 150, 25]), np.array([30, 255, 255])), \
		   ("left-forearm", np.array([140, 150, 25]), np.array([160, 255, 255])), \
		   ("left-hand", np.array([80, 100, 25]), np.array([95, 255, 255])), \
		   ("torso", np.array((0,100,120)), np.array((5,255,255)), np.array((175,100,120)), np.array((180,255,255)))]

#---------------------------------------------------Create Directories---------------------------------------------------#
path = os.getcwd()
print ("The current working directory is %s" % path)

path = "./final"
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)


for i in range(0, len(filters), 1):
	folder_name = filters[i][0]
	folder_path = path + "/" + folder_name
	try:
	    os.mkdir(folder_path)
	except OSError:
	    print ("Creation of the directory %s failed" % folder_path)
	else:
	    print ("Successfully created the directory %s " % folder_path)



for fname in os.listdir("./output/"):
	if ".jpg" in fname:
		filename = fname;
	else:
		continue
	frame = cv2.imread("./output/" + filename)

	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#---------------------------------------------------Apply Filters to Images---------------------------------------------------#
	for i in range(0, len(filters), 1):
		folder_name = filters[i][0]
		lower_blue = filters[i][1]
		upper_blue = filters[i][2]

		# Threshold the HSV image to get only specific color
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
				print("THE CHOPPED OFF STRING IS: " + filename[4:len(filename)-4])
				orig_image = cv2.imread("./input/" + filename[4:len(filename)-4])
				
				height, width = frame.shape[:2]

				x, y, w, h = get_width_height(rect, height, width)
				print(x)
				print(y)
				print(w)
				print(h)
				## TODO - ACTUALLY CROP THE IMAGE HERE
				cropped_image = orig_image[y:y+h, x:x+w]

				#Write the cropped image to a file
				try:
					cv2.imwrite('./final/' + folder_name + "/" + str(i) + "_" + filename[:len(filename)-4], cropped_image)
				except:
					print("Tried to print but failed")



		cv2.imshow('frame',frame)
		cv2.imshow('mask',mask)


		while True:
			k = cv2.waitKey(5) 
			if k == 27:
				break









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








