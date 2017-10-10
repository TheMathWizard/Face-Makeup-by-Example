import cv2
import imutils
import numpy as np
import warp
import makeup

def overlay(orig, makeup, mask):

    blur_mask = cv2.blur(mask, (20, 20))
    new = makeup.copy()
    for y in range(0, orig.shape[0]):
        for x in range(0, orig.shape[1]):
            w = blur_mask[y][x]/255
            if (w > 0.6):
                w = (w - 0.6) / 0.4
            else:
                w = 0
            new[y][x] = makeup[y][x]*w + orig[y][x]*(1 - w)


    return new


if __name__ == '__main__':
    subject = cv2.imread('subject.jpg', 1)
    target = cv2.imread('bluelip2.jpg', 1)
    subject = imutils.resize(subject, width=500)
    target = imutils.resize(target, width=500)
    sub, warped_tar = makeup.warp_target(subject, target)
    zeros = np.zeros(warped_tar.shape, dtype=warped_tar.dtype)
    ones = np.ones(warped_tar.shape, dtype=warped_tar.dtype)
    face_mask = np.where(warped_tar==[0,0,0], zeros, ones*255)
    res = cv2.imread('res.jpg', 1)

    new = overlay(subject, res, face_mask[:,:,0])
    cv2.imshow('new', new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




