#import rospy
import cv2 as cv
import numpy as np
from collections import Counter

def length(line):
    return(line[2] - line[0])

def heigth(line):
    return(line[3] - line[1])

def y(line):
    return line[1]

def x(line):
    return line[0]

def level(line):
    return line[2]

def imageProcessing(shelf):
    gray = cv.cvtColor(shelf, cv.COLOR_RGB2GRAY)
    bw = cv.adaptiveThreshold(~gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, -2) 

    return bw

def horizontalLines(shelf):
    erode_structure = cv.getStructuringElement(cv.MORPH_RECT,(9,1))
    erosion = cv.erode(shelf.copy(),erode_structure)
    dilate_structure = cv.getStructuringElement(cv.MORPH_RECT,(15,3))
    dilatation = cv.dilate(erosion.copy(), dilate_structure)
    canny_output = cv.Canny(dilatation, 50, 150, None, 3)
    #colored = cv.cvtColor(canny_output, cv.COLOR_GRAY2BGR)
    linesP = cv.HoughLinesP(canny_output, 1, np.pi / 180, 50, None, 50, 10)
    lines = [list(linesP[i].flatten()) for i in range(len(linesP))]    
    
    return canny_output, lines

def verticalLines(shelf):
    erode_structure = cv.getStructuringElement(cv.MORPH_RECT,(1,9))
    erosion = cv.erode(shelf.copy(),erode_structure)
    dilate_structure = cv.getStructuringElement(cv.MORPH_RECT,(3,15))
    dilatation = cv.dilate(erosion.copy(), dilate_structure)
    canny_output = cv.Canny(dilatation, 50, 150, None, 3)
    #colored = cv.cvtColor(canny_output, cv.COLOR_GRAY2BGR)
    linesP = cv.HoughLinesP(canny_output, 1, np.pi / 180, 50, None, 50, 10)
    lines = [list(linesP[i].flatten()) for i in range(len(linesP))]
    
    return canny_output, lines

def mLinesV(lines):
    main_linesV = []
    for line in lines:
        if length(line) > (length(lines[0]))*0:
            main_linesV.append(line)
    
    return main_linesV

def mLinesH(lines):
    main_linesH = []
    for line in lines:
        if length(line) > (length(lines[0]))*0:
            main_linesH.append(line)
    
    return main_linesH

def ordinating(lines):
    L = []
    for line in lines:
        if line[1] > line[3]:
            auxY = line[3]
            line[3] = line[1]
            line[1] = auxY
            auxX = line[2]
            line[2] = line[0]
            line[0] = auxX
        L.append(line)

    return L

def clusteringV(lines):
    clustered = []
    aux = [0,0,0,0]
    cont = 0
    for i in range((len(lines) - 1)):
        #print("xo" + str(lines[i][0]) + "..." + "yo" + str(lines[i][1]) + "..." + "xi" + str(lines[i][2]) + "..." + "yf" + str(lines[i][3]))
        if ((lines[i+1][0] - lines[i][0])>-30) and ((lines[i+1][0] - lines[i][0]) < 30):
            aux[0] += lines[i][0]
            aux[1] += lines[i][1]
            aux[2] += lines[i][2]
            aux[3] += lines[i][3]
            cont += 1
            #print(cont)
        else:
            cont += 1
            aux[0] += lines[i][0]
            aux[1] += lines[i][1]
            aux[2] += lines[i][2]
            aux[3] += lines[i][3]
            med = [(aux[0]/cont),(aux[1]/cont),(aux[2]/cont),(aux[3]/cont)]
            clustered.append(med)
            aux = [0,0,0,0]
            cont = 0 

    med = [(aux[0]/cont),(aux[1]/cont),(aux[2]/cont),(aux[3]/cont)]
    clustered.append(med)
    
    return clustered

def clusteringH(lines):
    clustered = []
    aux = [0,0,0,0]
    cont = 0
    for i in range((len(lines) - 1)):
        #print("xo" + str(lines[i][0]) + "..." + "yo" + str(lines[i][1]) + "..." + "xi" + str(lines[i][2]) + "..." + "yf" + str(lines[i][3]))
        if ((lines[i+1][1] - lines[i][1])>-17) and ((lines[i+1][1] - lines[i][1]) < 17):
            aux[0] += lines[i][0]
            aux[1] += lines[i][1]
            aux[2] += lines[i][2]
            aux[3] += lines[i][3]
            cont += 1
            #print("If  " + str(cont))
        else:
            cont += 1
            aux[0] += lines[i][0]
            aux[1] += lines[i][1]
            aux[2] += lines[i][2]
            aux[3] += lines[i][3]
            med = [(aux[0]/cont),(aux[1]/cont),(aux[2]/cont),(aux[3]/cont)]
            clustered.append(med)
            aux = [0,0,0,0]
            #print("Else before  " + str(cont))
            cont = 0
            #print("Else after  " + str(cont))

    med = [(aux[0]/cont),(aux[1]/cont),(aux[2]/cont),(aux[3]/cont)]
    clustered.append(med)
    
    return clustered

def extendingV(shelf,lines):
    L = []
    for line in lines:
        line[1] = 0
        line[2] = line[0] 
        line[3] = shelf.shape[1]
        L.append(line)

    return L 

def extendingH(shelf,lines):
    L = []
    for line in lines:
        line[0] = 0
        line[2] = shelf.shape[0]
        line[3] = line[1] 
        L.append(line)
    
    return L

def intersection(linesV,linesH):
    V = []
    H = []
    flag = False
    for i in range(0,(len(linesV))):
        for j in  range(0,len(linesH)):
            if j == 0:
                linesV[i][1] = linesH[j][1]

            if i == 0:
                linesH[j][0] = linesV[i][0]
            elif i == (len(linesV)-1):
                linesH[j][2] = linesV[i][2]
            
            if j == (len(linesH) - 1):
                linesV[i][3] = linesH[j][3]
            #print("xo" + str(linesH[i][0]) + " ... " + "yo" + str(linesH[i][1]) + " ... " + "xi" + str(linesH[i][2]) + " ... " + "yf" + str(linesH[i][3]))    
            if flag == False:
                H.append(linesH[j])
        flag = True
        V.append(linesV[i])
            
    return V, H

def drawLines(shelf,linesV,linesH):
    if linesV is not None:
        for i in range(0, len(linesV)):
            l = linesV[i]
            cv.line(shelf, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    if linesH is not None:
        for i in range(0, len(linesH)):
            l = linesH[i]
            cv.line(shelf, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    return shelf

def findingLevels(shelf):
    processed_image = imageProcessing(shelf.copy())

    vertical, linesV = verticalLines(processed_image.copy())
    horizontal, linesH = horizontalLines(processed_image.copy())

    cv.imshow("Vertical",vertical)
    cv.imshow("Horzontal",horizontal)

    linesV_sort = sorted(linesV,key=heigth,reverse=True)
    linesH_sort = sorted(linesH,key=length,reverse=True)

    main_linesV = mLinesV(linesV_sort)
    main_linesH = mLinesH(linesH_sort)

    ordinatedV = ordinating(main_linesV)
    ordinatedH = ordinating(main_linesH)

    clusterV = clusteringV(sorted(ordinatedV,key=x))
    clusterH = clusteringH(sorted(ordinatedH,key=y))

    extended_linesV = extendingV(shelf.copy(),clusterV)
    extended_linesH = extendingH(shelf.copy(),clusterH)

    interV, interH = intersection(extended_linesV,extended_linesH)
    final = drawLines(shelf.copy(), interV, interH)

    nLevels = len(interH)
    lines = []

    for line in interH:
        aux = [line[0],line[2],line[3]]
        lines.append(aux)
    
    lines = sorted(lines,key=level,reverse=1)

    return nLevels, lines, final

def getCenterBB(BoundingBox):
    center = (BoundingBox.minX + int(BoundingBox.width/2), BoundingBox.minY + int(BoundingBox.height/2))
    return center

def findingObjects(nLevels, lines, descriptions):
    object_type = []
    object_labels = []
    for i in range(len(lines) - 1):
        object_level = []
        for description in descriptions:
            center_bbx = getCenterBB(description.BoundingBox)
            if center_bbx[1] > lines[i][2] and center_bbx[1] < lines[i+1][2]:
                object_level.append(description.label_class)
        if len(object_level) == 0:
            object_type.append("empty")
            object_level.append("empty")
            object_labels.append(object_level)
        else:
            types = [e.split("/")[0] for e in object_type]
            types = Counter(types)
            object_type.append(types.most_common(1))
            object_labels.append(object_level)
    
    return object_type, object_labels
            
    


#nivel, linha, imagem = findingLevels(cv.imread("estante1.jpg",cv.IMREAD_COLOR))

#print(nivel)
#print(linha)
#cv.imshow("Resultado",imagem)
#cv.waitKey(0)
