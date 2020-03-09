#coding:utf-8
import numpy as np
from skimage import io, color
from PIL import Image as image #加载PIL包，用于加载创建图片
from sklearn.metrics import normalized_mutual_info_score

from DSMOC.dsmoc_image import computePBM


def getMockResult(filename):
    fr = open(filename)
    arraylines = fr.readlines()
    MockResult = []
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(' ')  # 用split会返回一个list
        MockResult.append(linestrlist[-1])
    return MockResult

def loadlabel(filepath):
    fr = open(filepath)
    arraylines = fr.readlines()
    label = []
    for line in arraylines:
        label.extend(line.split(" "))
    return label

def getsuperpixelData(filename):
    fr = open(filename)
    arraylines = fr.readlines()
    numOfLines = len(arraylines)
    attributeNum = 3
    returnMat = np.zeros((numOfLines, attributeNum))
    index = 0
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(' ')  # 用split会返回一个list
        returnMat[index, :] = linestrlist[0:]
        index += 1
    return  returnMat


if __name__ == '__main__':
    rgb = io.imread('../slic_segment/3000323.jpg')
    lab_arr = color.rgb2lab(rgb)
    pic_new = image.new("L", (lab_arr.shape[1], lab_arr.shape[0]))
    predict_matrix = np.zeros((lab_arr.shape[0], lab_arr.shape[1]))

    fo = open('../mockimagedata/clusters3000323.txt')
    clusterslines = fo.readlines()



    mocklabel = getMockResult('../mockk5numbersolution/3000323-6-17.solution')
    clusterArrList = []
    clusterArrList.extend(list(set(mocklabel)))
    for label in clusterArrList:
        index = 0
        for element in mocklabel:
            if element == label:
                clusterslinelist = clusterslines[index].split('s')
                if clusterslinelist[0] == '\n':
                    continue
                for pairlen in range(len(clusterslinelist)-1):
                    pairElement = clusterslinelist[pairlen].split(',')
                    p1 = int(pairElement[1].replace(')',''))
                    p0 = int(pairElement[0].replace('(',''))
                    pic_new.putpixel((p1, p0), int(256 / (int(label) + 1)))
                    predict_matrix[p0,p1] = label

            index += 1

    pic_new.save("../aftersegmentationimagenew/3000323mock-1000-10k5number5.jpg", "JPEG")

    predictList = []
    for i in range(lab_arr.shape[0]):
        for j in range(lab_arr.shape[1]):
            predictList.append(predict_matrix[i][j])
    regionslabel = loadlabel("../imagelabel/3000323.regions.txt")
    layerslabel = loadlabel("../imagelabel/3000323.layers.txt")
    surfaceslabel = loadlabel("../imagelabel/3000323.surfaces.txt")

    mockdata = getsuperpixelData("../mockimagedata/3000323mocksuperPixel.txt")
    pbmlabel = []
    pbmlabel.append(mocklabel)
    result,pbmvalue = computePBM(mockdata,pbmlabel)

    regionslabel = normalized_mutual_info_score(regionslabel, predictList)
    layerslabel = normalized_mutual_info_score(layerslabel, predictList)
    surfaceslabel = normalized_mutual_info_score(surfaceslabel, predictList)

    print "pbm值为为%s" % pbmvalue
    print "regionslabel为%s" % regionslabel
    print "layerslabel为%s" % layerslabel
    print "surfaceslabel为%s" % surfaceslabel