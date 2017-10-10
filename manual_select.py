import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils


def draw_points(img, shape):
    image = img.copy()
    for num, (x, y) in enumerate(shape):
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        txt = ("(" + str(num) + ")")
        cv2.putText(image, txt, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
    return image

def get_curr_location(event, x, y, flags, param):
    shape, ind = param
    if event == cv2.EVENT_LBUTTONUP:
        shape[ind] = x, y

#This function returns the changed value of shape
def edit_points(img, shape):
    print("Help:\n\t(*)type \'exit\' to stop editing\n\t(*)\'del\' to remove a point")
    while True:
        cv2.namedWindow('Edit')
        cv2.imshow("Edit", draw_points(img, shape))
        cv2.waitKey(200)
        cmd = input("Enter command: ")
        cv2.destroyAllWindows()
        if (cmd == 'exit'):
            cv2.destroyAllWindows()
            break
        elif (cmd == 'del'):
            ind = int(input("Enter index to delete: "))
            shape[ind] = (0, 0)
            cv2.destroyAllWindows()
            cv2.namedWindow('Edit')
            cv2.imshow("Edit", draw_points(img, shape))
            cv2.setMouseCallback("Edit", get_curr_location, param=(shape, ind))
            print("Now select the point by clicking on the image and press \'esc\' to confirm")
            while True:
                cv2.imshow("Edit", draw_points(img, shape))
                if cv2.waitKey(20) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    break


    return shape

if __name__ == '__main__':
    subject = cv2.imread('subject.jpg', 1)
    target = cv2.imread('target.jpg', 1)
    subject = imutils.resize(subject, width=500)
    target = imutils.resize(target, width=500)
    gray_sub = cv2.cvtColor(subject, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # detect faces in the grayscale image
    rects = detector(gray_sub, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray_sub, rect)
        shape = face_utils.shape_to_np(shape)
        new_shape = edit_points(subject, shape)

        cv2.imshow("Final", draw_points(subject, new_shape))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

