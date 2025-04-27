import numpy as np
import os
import cv2
from matplotlib import pyplot as  plt

GT_dir=os.path.abspath(r"Cobb/GT")
ctrln_dir=os.path.abspath('Cobb/centerlines')
GTs=os.listdir(GT_dir)
print(GTs)
for gt in GTs:
    path=GT_dir+'/'+gt
    dest=ctrln_dir+'/'+gt
    print(path)
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(600,600))
    img=~img
    gray_filtered = cv2.bilateralFilter(img, 7, 50, 50) #removes bg noise

    # Applying the canny filter
    edges = cv2.Canny(img, 60, 120)
    #cv2.imshow('edges',edges)
    edges_filtered = cv2.Canny(gray_filtered, 60, 120)
    #cv2.imshow('fedges',edges_filtered)
    imgray=~edges_filtered
    #cv2.imshow('fedges',imgray)

    ret, thresh= cv2.threshold(imgray,10,255,0)
    #cv2.imshow('thresh',thresh)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours.pop(0)


    all_x=[]
    cord=dict()
    all_y=[]
    for i in range(len(contours)):
     #print(len(contours[i]))
        for j in range(len(contours[i])):
            all_x.append(contours[i][j][0][0])
            all_y.append(contours[i][j][0][1])
            if(contours[i][j][0][1] in cord.keys()):
                temp=cord[contours[i][j][0][1]]
                temp.append(contours[i][j][0][0])
                cord[contours[i][j][0][1]]=temp
            else:
                cord[contours[i][j][0][1]]=[contours[i][j][0][0]]


    #print(len(all_x),len(all_y))
    #print(cord)
    final={}
    for keyy in cord:
        temp=(max(cord[keyy])+min(cord[keyy]))/2
        final[keyy]=temp
    from collections import OrderedDict
    finall=OrderedDict(sorted(final.items()))
    #plt.plot(list(finall.values()),list(finall.keys()))


    draw_x=list(finall.values())
    draw_y=list(finall.keys())

    
    draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed

    cmg=np.zeros((600,600))

    cmg=cv2.polylines(cmg, [draw_points], False, (225,0,0), thickness=3)
    cv2.imshow("dest",cmg)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break


