# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def find_mask(image, betamap):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the face parts individually
		mask = np.zeros(image.shape, dtype=image.dtype)
		noseMask = np.zeros(image.shape, dtype=image.dtype)
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)

			if(betamap):
				if(name=='right_eyebrow' or name=='left_eyebrow'):
					continue
				if(name=='jaw'):
					continue
			else:
				if(name=='jaw' or name=='nose' or name=='left_eyebrow' or name=='right_eyebrow'):
					continue
			pts = shape[i:j]
			hull = cv2.convexHull(pts)
			if(name=='nose'):
				cv2.drawContours(noseMask, [hull], -1, (255,255,255), -1)
			else:
				cv2.drawContours(mask, [hull], -1, (255,255,255), -1)

		if(betamap):
			kernel = np.ones((5,5),np.uint8)
			dilation = cv2.dilate(mask,kernel,iterations = 4)
			gradient = cv2.morphologyEx(noseMask, cv2.MORPH_GRADIENT, kernel)
			gradient = cv2.dilate(gradient,kernel,iterations = 2)
			#cv2.imshow("Mask", mask)
			#cv2.imshow("Dilated Mask", dilation)
			#cv2.imshow("Nose Mask", gradient)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()
			mask = dilation+gradient


		return mask

if __name__ == '__main__':
	image = cv2.imread('subject.jpg', 1)
	image = imutils.resize(image, width=500)
	find_mask(image, False)