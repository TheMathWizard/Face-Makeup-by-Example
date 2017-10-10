# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import skin_detector
import imutils
import dlib
import cv2
import manual_select

def draw_delaunay(img, subdiv, delaunay_color) :
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList :
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

gmin = -1
def findtop(img, coord):
    global gmin
    mask = skin_detector.process(img)
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    x, y = coord
    y = y-10
    if(gmin == -1):
        gmin = y+1
    else:
        if(gmin+20<y):
            y = gmin+20
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    while(1):
        #img[y:y+1,x:x+1] = np.zeros((1,1),'uint8')
        if(y==0):
            break
        y-=1
        if(abs(l.item(y,x)-l.item(y+1,x))>=15 or mask.item(y,x)==0):
            break
    gmin = y
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    return (x,y)


def triangulate(image):
    global gmin

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # load the input image, resize it, and convert it to grayscale
    image = imutils.resize(image, width=500)
    img = image.copy()  # used in manual editing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cx = int(.15*w)
        cy = int(.5*h)
        #cv2.rectangle(image, (x-cx, y-cx), (x+w+cx, y+h+cx), (0, 255, 0), 2)
        #cv2.imshow('x', image)
        #cv2.waitKey(0)

        # show the face number
        #cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #rect = (0, 0, image.shape[1], image.shape[0])
        subdiv = cv2.Subdiv2D((max(x-cx,0), max(y-cy,0), min(w+x+cx,image.shape[1]), min(h+y+cx,image.shape[0])))
        forehead = []
        gmin = -1
        for num, (x, y) in enumerate(shape):


            if((num>=17 and num<=27)):
                forehead.append(findtop(image, (x,y)))

            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            #cv2.imshow('show', image)
            #cv2.waitKey(0)

        for item in forehead:
            shape = np.vstack((shape, item))
            #print(item)

        #manual_select.edit_points(img, shape)


        for (x, y) in shape:
            subdiv.insert((int(x),int(y)))
        #draw_delaunay(image, subdiv, (255, 255, 255))
        #cv2.imshow("Output", image)
        #cv2.waitKey(0)

        return shape, subdiv.getTriangleList()

if __name__ == '__main__':
    image = cv2.imread('target.jpg')
    triangulate(image)