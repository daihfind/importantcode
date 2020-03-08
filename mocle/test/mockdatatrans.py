#coding:utf-8

from numpy import *

#默认第一列为数据所属类别
def getdata(filename):
    fr = open(filename)
    arraylines = fr.readlines()
    datalabelArr = []
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(' ')  # 用split会返回一个list
        datalabelArr.append(linestrlist[0])
    return datalabelArr

def trans(filename):
    fr = open(filename)
    arraylines = fr.readlines()
    datafile = open('%trans_segmentation.txt', 'w')
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(' ')  # 用split会返回一个list
        tmplabel = linestrlist[0]
        linestrlist[0] = linestrlist[-1]
        linestrlist[-1] = tmplabel
        for attribute in linestrlist:
            datafile.write(attribute+" ")
        datafile.write('\n')
    datafile.close()
    return arraylines

def transsuperPixel(filenpath):
    fr = open(filenpath)
    arraylines = fr.readlines()
    datafile = open('9004581mocksuperPixel.txt', 'w')
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(' ')  # 用split会返回一个list
        linestrlist = linestrlist[0:-1]
        for attribute in linestrlist:
            datafile.write(attribute+" ")
        datafile.write('\n')
    datafile.close()
    return arraylines
if __name__ == '__main__':
    transsuperPixel('../mockimagedata/mockdata9004581.txt')