import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
def getCoords(image):
    img= cv2.imread(image)
    #cv2.imshow('try',img)
    edges = cv2.Canny(img, 60, 120)
    imgray=~edges
    ret, thresh= cv2.threshold(imgray,10,255,0)
    #cv2.imshow('thresh',thresh)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgray,contours,-1,(0,55,0),1)
    #cv2.imshow('contours',imgray)
    contours.pop(0)
    all_x = []
    cord = dict()
    all_y = []
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            all_x.append(contours[i][j][0][0])
            all_y.append(contours[i][j][0][1])
            if (contours[i][j][0][1] in cord.keys()):
                temp = cord[contours[i][j][0][1]]
                temp.append(contours[i][j][0][0])
                cord[contours[i][j][0][1]] = temp
            else:
                cord[contours[i][j][0][1]] = [contours[i][j][0][0]]
    final = {}
    for keyy in cord:
        temp = (max(cord[keyy]) + min(cord[keyy])) / 2
        final[keyy] = temp
    from collections import OrderedDict

    finall = OrderedDict(sorted(final.items()))
    # plt.plot(list(finall.values()),list(finall.keys()))

    draw_x = list(finall.values())
    draw_y = list(finall.keys())
    px = []
    py = []
    for i in range(0, len(draw_x),20):
        px.append(draw_x[i])
        py.append(draw_y[i])

    yhat = savgol_filter(py, 11, 3)  # window size 51, polynomial order 3
    draw_points = (np.asarray([px, yhat]).T).astype(np.int32)  # needs to be int32 and transposed
    #cmg = np.zeros((600, 600))
    #cmg = cv2.polylines(cmg, [draw_points], False, (225, 0, 0), thickness=1)
    #cv2.imshow(dest, cmg)
    #plt.plot(px, yhat)
    #plt.show()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return(px,yhat)

X,Y=getCoords('smoothened.png')
print(Y)
plt.plot(X,Y)
def getSlopes(X,Y):
    slope=[]
    for i in range(len(X) - 1):
        if (X[i] == X[i + 1]):
            slope.append((Y[i] - Y[i + 1]) * 1)
        else:
            slope.append((Y[i] - Y[i + 1]) / (X[i] - X[i + 1]))
    temp = slope[-1]
    slope.append(temp)
    return(slope)

def drawLines(X,Y,slopes):
    for i in range(0,len(X)):
        x=np.linspace(X[i]-2,X[i]+2,3)
        m=-1/slopes[i]
        plt.plot(x,m*np.array(x)+(Y[i]-X[i]*m))
drawLines(X,Y,getSlopes(X,Y))
M=-1/np.array(getSlopes(X,Y))

def getThreshold(X : list, Y: list ) -> float:
    height=max(Y)-min(Y)
    end_thoracic=height*12/17
    return(end_thoracic)

def classify(X,Y,M):
    Thoracic_slopes,Lumbar_slopes=[],[]
    threshold=getThreshold(X, Y)
    for i in range(len(Y)):
        if Y[i]>threshold:
            Thoracic_slopes.append(M[i])
        else:
            Lumbar_slopes.append(M[i])
    return Thoracic_slopes,Lumbar_slopes


def CAngle(m1: float, m2:float) -> float:
    from math import atan,pi
    angle= (180/pi)*(atan(m1)-atan(m2))/(1+atan(m1)*atan(m2))
    return (abs(angle)%90)



TSlopes,LSlopes=classify(X,Y,M)
print(CAngle(max(M),min(M)))
print(CAngle(M[0],min(TSlopes)))
print(CAngle(min(LSlopes),max(LSlopes)))

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
