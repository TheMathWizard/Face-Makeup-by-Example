import cv2
import facial_landmark
import imutils
import numpy as np

def warp(src, dst):
    src_points, src_triangles = facial_landmark.triangulate(src)
    dst_points, dst_triangles = facial_landmark.triangulate(dst)
    warped_image = np.zeros(src.shape, dtype=np.uint8)

    for i,t in enumerate(dst_triangles):
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        first, second, third = -1,-1,-1
        for i2, (x,y) in enumerate(dst_points):
            if(x==t[0] and y==t[1]):
                #print('triangle '+(i)+' has first point at '+str(i2))
                #print(x,y)
                #cv2.circle(src,(x,y), 1, (i*2,0,i*3), -1)
                first = i2
            if(x==t[2] and y==t[3]):
                #print('triangle '+str(i)+' has second point at '+str(i2))
                #print(x,y)
                #cv2.circle(src,(x,y), 1, (i*2,0,i*3), -1)
                second = i2
            if(x==t[4] and y==t[5]):
                #print('triangle '+str(i)+' has third point at '+str(i2))
                #print(x,y)
                #cv2.circle(src,(x,y), 1, (i*2,0,i*3), -1)
                third = i2
        if(first>=0 and second>=0 and third>=0):
            x1,y1 = src_points[first]
            x2,y2 = src_points[second]
            x3,y3 = src_points[third]
            dx1,dy1 = dst_points[first]
            dx2,dy2 = dst_points[second]
            dx3,dy3 = dst_points[third]

            #creating mask in destination image
            mask = np.zeros(src.shape, dtype=np.uint8)
            roi_corners = np.array([[dx1,dy1],[dx2,dy2],[dx3,dy3]], dtype=np.int32)
            cv2.fillPoly(mask, [roi_corners], (255,255,255))

            #warping src image to destination image
            pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3]])
            pts2 = np.float32([[dx1,dy1],[dx2,dy2],[dx3,dy3]])
            M = cv2.getAffineTransform(pts1,pts2)
            rows,cols,ch = src.shape
            res = cv2.warpAffine(src,M,(cols,rows))

            warped_image = cv2.bitwise_or(warped_image,cv2.bitwise_and(mask,res))
            #cv2.imshow('masked_image', warped_image)
            #cv2.waitKey(0)
            '''
            cv2.line(src, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA, 0)
            cv2.line(src, pt2, pt3, (255, 255, 255), 1, cv2.LINE_AA, 0)
            cv2.line(src, pt3, pt1, (255, 255, 255), 1, cv2.LINE_AA, 0)
            '''
    #cv2.imshow('res',warped_image)
    #cv2.waitKey(0)
    return warped_image

if __name__ == '__main__':
    src = cv2.imread('example_01.jpg', 1)
    dst = cv2.imread('example_02.jpg', 1)
    src = imutils.resize(src, width=500)
    dst = imutils.resize(dst, width=500)