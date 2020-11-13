import argparse
import numpy as np 
import progressbar
import cv2
from PIL import Image
from time import sleep
import imutils
import os


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--path", required=True,
		help="base path to directory")
	ap.add_argument("-b", "--base", required=True,
		help="what you want the files to look like")
	args = vars(ap.parse_args())

	directory = args["path"]
	base = args["base"]
	
	for index, fname in enumerate(os.listdir(directory)):
		if ".DS" not in fname:
			os.rename(os.path.join(directory, fname), os.path.join(directory, ''.join([base + str(index), '.jpg'])))


if __name__ == '__main__':
	main()