import cv2
import numpy as np
import skin_detector

'''
def edit(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print('button down detected')
        fill(x, y, 1)
def reverse(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        fill(x, y, -1)

def edit(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
'''


def floodfill(img):
    def fill(x, y,dir):
        stack = []
        prev = [-1,-1,-1]
        stack.append((x,y,prev))
        while(len(stack)!=0):
            x,y,prev = stack.pop()
            if(y>=img.shape[0] or x>=img.shape[1] or y<0 or x<0):
                continue
            if(checked.item(y,x)==1):
                #print('checked')
                continue
            l = lab.item(y,x,0)
            a = lab.item(y,x,1)
            b = lab.item(y,x,2)
            #print(str(l) + " " + str(a) + " " + str(b))
            if(prev[0]!=-1):
                grad = (abs(prev[0]-l))
                if(grad > 5):
                    continue
                #print(grad)
            if(dir==1):
                mask[y:y+1,x:x+1] = np.ones((1,1),'uint8')*255
            #else:
            #    img[y:y+1,x:x+1] = np.ones((1,1,3),'uint8')*255
            checked.itemset((y,x), 1)
            #print(checked.item(y,x))
            stack.append((x,y+1,(l,a,b)))
            stack.append((x,y-1,(l,a,b)))
            stack.append((x+1,y,(l,a,b)))
            stack.append((x-1,y,(l,a,b)))

    mask = skin_detector.process(img)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    checked = np.zeros((img.shape[0], img.shape[1]))
    cv2.namedWindow('img')
    for y in range(0,img.shape[0]):
        for x in range(0,img.shape[1]):
            if(mask.item(y,x)==255):
                fill(x,y,1)
    #cv2.setMouseCallback('img', reverse)
    while(1):
        s = cv2.bitwise_and(img, img, mask = mask)
        cv2.imshow("img", s)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    '''
    l,a,b = cv2.split(lab)
    cv2.imshow('b', b)
    b2 = cv2.bilateralFilter(b, 9,100,75)
    cv2.imshow('a', b2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    img = cv2.imread('10.jpg', 1)
    floodfill(img)

