#coding:utf-8

from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
from numpy import *
def getMockResult(filename):
    fr = open(filename)
    arraylines = fr.readlines()
    MockResult = []
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(' ')  # 用split会返回一个list
        MockResult.append(linestrlist[-1])
    return MockResult

def loadDataset(filename):
    fr = open(filename)
    arraylines = fr.readlines()
    numOfLines = len(arraylines)
    returnMat = zeros((numOfLines, 4))
    classlabelVector = []
    index = 0
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(',')  # 用split会返回一个list
        returnMat[index, :] = linestrlist[1:5]
        classlabelVector.append(linestrlist[0])
        index += 1
    return returnMat, classlabelVector

def computeAri(result,realLabels):
    ari = adjusted_rand_score(realLabels,result)
    return ari


if __name__ == '__main__':
    datamat,labels = loadDataset('../dataset/balance-scale.data')
    result = getMockResult('../analyzeMock/balance-scale/10-63.solution')
    ari = computeAri(result,labels)
    print 'ari值为：%s'%ari
    nmi = normalized_mutual_info_score(result,labels)
    print 'nmi值为:%s'%nmi