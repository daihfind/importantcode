#coding:utf-8

import numpy as np
from PIL import Image as image #加载PIL包，用于加载创建图片
# from sklearn.cluster import  KMeans#加载Kmeans算法
from DSMOC.dsmoc_image import *

# def loadData(filePath):
#     f = open(filePath,'rb') #以二进制形式打开文件
#     data= []
#     img =image.open(f)#以列表形式返回图片像素值
#     m,n =img.size #获得图片大小
#     for i in range(m):
#         for j in range(n): #将每个像素点RGB颜色处理到0-1范围内
#             x,y,z =img.getpixel((i,j)) #将颜色值存入data内
#             data.append([x/256.0,y/256.0,z/256.0])
#             f.close() #以矩阵的形式返回data，以及图片大小
#     return np.array(data),m,n
#
# imgData,row,col =loadData('bull.jpeg')#加载数据
from slic_segment.slic import SLICProcessor

pro = SLICProcessor('../slic_segment/3000323.jpg', 1500, 10)
superPixelMatrix, clusters = pro.getDataForDSMOCMulti()


# km=KMeans(n_clusters=3) #聚类获得每个像素所属的类别
# label =km.fit_predict(imgData)
label_list = dsmoc(superPixelMatrix)
print 111
image_arr = np.copy(pro.data)
pic_new = image.new("L", (pro.image_width,pro.image_height))
pixel_label = np.zeros((pro.image_height, pro.image_width))
clusterArrList = []
for i in range(len(label_list)):
    clusterArrList.append(list(set(label_list[i])))

i = 0
while i<len(label_list):
    for label in clusterArrList[i]:
        index1 = 0
        index2 = 0
        count = 0
        sum_l = 0
        sum_a = 0
        sum_b = 0
        dict = {}
        for element in label_list[i]:
            if (label == element):
                sum_l += clusters[index1].l
                sum_a += clusters[index1].a
                sum_b += clusters[index1].b
                count += 1
            index1 += 1
        dict[label] = [sum_l/count, sum_a/count, sum_b/count]
        for element in label_list[i]:
            if (label == element):
                for p in clusters[index2].pixels:
                    image_arr[p[0]][p[1]][0] = dict[label][0]
                    image_arr[p[0]][p[1]][1] = dict[label][1]
                    image_arr[p[0]][p[1]][2] = dict[label][2]
                    pic_new.putpixel((p[1], p[0]), int(256 / (label + 1)))
            index2 += 1
    pic_new.save("../aftersegmentationnimage/3000323-(random2-5)-1500-10-gen20multi-%s.jpg" % i, "JPEG")
    # rgb_arr = color.lab2rgb(image_arr)
    # name = 'result{m}.png'.format(m=i)
    # io.imsave(name, rgb_arr)
    i = i + 1
print '运行结束'
# for index in range(len(label_list)):
#     label=label_list[index].reshape([p.image_height,p.image_width]) #创建一张新的灰度图以保存聚类后的结果
#     pic_new = image.new("L",(p.image_height,p.image_width)) #根据类别向图片中添加灰度值
#     for i in range(row):
#         for j in range(col):
#             pic_new.putpixel((i,j),int(256/(label[i][j]+1))) #以JPEG格式保存图像
#     pic_new.save("result-bull-%s.jpg"%index,"JPEG")
