#coding:utf-8

import numpy as np
from PIL import Image as image #加载PIL包，用于加载创建图片
# from sklearn.cluster import  KMeans#加载Kmeans算法
from DSMOC.dsmoc_image import *
from DSMOC.moclenew import moclenew
from slic_segment.slic import SLICProcessor



picturenumber = '3000323'
def loadlabel(filepath):
    fr = open(filepath)
    arraylines = fr.readlines()
    label = []
    for line in arraylines:
        label.extend(line.split(" "))
    return label

regionslabel = loadlabel("../imagelabel/%s.regions.txt"%picturenumber)
layerslabel = loadlabel("../imagelabel/%s.layers.txt"%picturenumber)
surfaceslabel = loadlabel("../imagelabel/%s.surfaces.txt"%picturenumber)
pro = SLICProcessor('../slic_segment/%s.jpg'%picturenumber, 1000, 10)
superPixelMatrix, clusters = pro.getDataForDSMOCMulti()


# km=KMeans(n_clusters=3) #聚类获得每个像素所属的类别
# label =km.fit_predict(imgData)
label_list,pbmValue = moclenew(superPixelMatrix)
print '选择出来的图片的pbm值为%s'%pbmValue
print 'dsmoc运行结束'
doc = open('out.txt','w')
doc.write(str(pbmValue))
doc.close()
pic_new = image.new("L", (pro.image_width,pro.image_height))
predict_matrix = zeros((pro.image_height,pro.image_width))
clusterArrList = []
clusterArrList.extend(list(set(label_list)))
for label in clusterArrList:
    index2 = 0
    for element in label_list:
        if (label == element):
            for p in clusters[index2].pixels:
                pic_new.putpixel((p[1], p[0]), int(256 / (label+1)))
                predict_matrix[p[0]][p[1]] = label
        index2 += 1
pic_new.save("../aftersegmentationimagenew/%sPBM-MOCLE-(random(3-5))-1000-10-gen20MultiNomal0.jpg"%picturenumber, "JPEG")
predictList = []
for i in range(pro.image_height):
    for j in range(pro.image_width):
        predictList.append(predict_matrix[i][j])
regionslabel = normalized_mutual_info_score(regionslabel, predictList)
layerslabel = normalized_mutual_info_score(layerslabel, predictList)
surfaceslabel = normalized_mutual_info_score(surfaceslabel, predictList)

print "regionslabel为%s"%regionslabel
print "layerslabel为%s"%layerslabel
print "surfaceslabel为%s"%surfaceslabel

    # rgb_arr = color.lab2rgb(image_arr)
    # name = 'result{m}.png'.format(m=i)
    # io.imsave(name, rgb_arr)
print '运行结束'
# for index in range(len(label_list)):
#     label=label_list[index].reshape([p.image_height,p.image_width]) #创建一张新的灰度图以保存聚类后的结果
#     pic_new = image.new("L",(p.image_height,p.image_width)) #根据类别向图片中添加灰度值
#     for i in range(row):
#         for j in range(col):
#             pic_new.putpixel((i,j),int(256/(label[i][j]+1))) #以JPEG格式保存图像
#     pic_new.save("result-bull-%s.jpg"%index,"JPEG")
