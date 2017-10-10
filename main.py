import cv2
import numpy as np
import flood
import faces

img = cv2.imread('subject.jpg',1)
fs = faces.boundingrect(img)
for (x,y,w,h) in fs:
        img2 = img[y:y+h, x:x+w]
        flood.floodfill(img2)

