#coding:utf-8


import numpy as np
from skimage import io, color
from PIL import Image as image #加载PIL包，用于加载创建图片

def genpixeldata(filepath):
    rgb = io.imread(filepath)
    lab_arr = color.rgb2lab(rgb)
    doc = open('mockpixeldata0103468.txt', 'w')
    for i in range(lab_arr.shape[0]):
        for j in range(lab_arr.shape[1]):
            x = lab_arr[i][j]
            doc.write(str(lab_arr[i][j][0]))
            doc.write(' ')
            doc.write(str(lab_arr[i][j][1]))
            doc.write(' ')
            doc.write(str(lab_arr[i][j][2]))
            doc.write(' ')
            doc.write('1')
            doc.write("\n")

    doc.close()
    print '数据写入完成'

if __name__ == '__main__':
    genpixeldata('../slic_segment/0103468.jpg')