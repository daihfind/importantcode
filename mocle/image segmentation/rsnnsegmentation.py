# coding:utf-8
'''
Created on 2018年3月23日

@author: David
'''
from numpy import *
import numpy as np
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import AgglomerativeClustering
from PIL import Image as image

import tables
from sklearn import preprocessing
import array
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
#######数据集########
#coding:utf-8
'''
Created on 2019年1月14日

@author: David
'''
import numpy as np
from PIL import Image as image
#加载PIL包，用于加载创建图片
from sklearn.cluster import  KMeans#加载Kmeans算法
from sklearn.metrics import normalized_mutual_info_score

from DSMOC.dsmoc_image import computePBM


def loadData(filePath):
    f = open(filePath,'rb') #以二进制形式打开文件
    data= []
    img =image.open(f)#以列表形式返回图片像素值
    m,n =img.size     #获得图片大小
    for i in range(m):
        for j in range(n):
            #将每个像素点RGB颜色处理到0-1范围内
            x,y,z =img.getpixel((i,j))
            #将颜色值存入data内
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    #以矩阵的形式返回data，以及图片大小
    return np.mat(data),m,n

def loadlabel(filepath):
    fr = open(filepath)
    arraylines = fr.readlines()
    label = []
    for line in arraylines:
        label.extend(line.split(" "))

    return label
regionslabel = loadlabel("../imagelabel/0103468.regions.txt")
layerslabel = loadlabel("../imagelabel/0103468.layers.txt")
surfaceslabel = loadlabel("../imagelabel/0103468.surfaces.txt")
imgData,row,col =loadData('../slic_segment/0103468.jpg')#加载数据

km=KMeans(n_clusters=5)
#聚类获得每个像素所属的类别
originallabel =km.fit_predict(imgData)
label=originallabel.reshape([row,col])
#创建一张新的灰度图以保存聚类后的结果
pic_new = image.new("L",(row,col))
#根据类别向图片中添加灰度值
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j),int(256/(label[i][j]+1)))
pbmlabel = []
pbmlabel.append(originallabel)
result,pbmValue = computePBM(imgData,pbmlabel)
print 'pbm值为%s'%pbmValue

regionslabel = normalized_mutual_info_score(regionslabel, originallabel)
layerslabel = normalized_mutual_info_score(layerslabel, originallabel)
surfaceslabel = normalized_mutual_info_score(surfaceslabel, originallabel)

print "regionslabel为%s"%regionslabel
print "layerslabel为%s"%layerslabel
print "surfaceslabel为%s"%surfaceslabel
#以JPEG格式保存图像
pic_new.save("../aftersegmentationimagenew/0103468kmeans9.jpg","JPEG")

########################################################################

def data_sample(dataset,rate,ensembleSize):
    length = len(dataset)
    num = round(length*rate)#一个数据集里要采样的数据数量
    allIndex = [] #全部重采样出来的数据
    #赋值
    for i in range(length):
        allIndex.append(i)
    sampledData = [] #重采样出来的全部数据
    remainedData = [] #全部的除去采样出来的数据的其他数据
    sampledIndex = [] #全部重采样出来的数据的索引值
    remainedIndex = [] #全部除去采样出来的数据的其他数据的索引值
    for i in range(ensembleSize):
        sampledDataOne = []  # 一次重采样的数据
        remainedDataOne = [] #一次除去采样出来的数据的其他数据
        sampledIndexOne = []  # 一次重采样出来的数据的索引值
        remainedIndexOne = [] #一次除去采样出来的数据的其他数据的索引值
        for j in range(int(num)):
            sampleI = random.randint(0,length)
            sampledIndexOne.append(sampleI)
        sampledIndexOne = list(set(sampledIndexOne))#采样出来的数据的索引值,去重后
        remainedIndexOne = (list(set(allIndex).difference(set(sampledIndexOne))))
        for j in range(len(sampledIndexOne)):
            sampledDataOne.append(dataset[sampledIndexOne[j]])
        for j in range(len(remainedIndexOne)):
            remainedDataOne.append(dataset[remainedIndexOne[j]])
        sampledData.append(sampledDataOne)
        remainedData.append(remainedDataOne)
        sampledIndex.append(sampledIndexOne)
        remainedIndex.append(remainedIndexOne)

    return sampledData,remainedData,sampledIndex,remainedIndex

def rsnn(sampledData,remainedData,sampledIndex,remainedIndex,singleName):
    predicted_labelAll = []
    for i in range(len(sampledData)):
        # clusters = random.randint(min_clusters,max_clusters)
        clusters = random.randint(2, 11)
        # clusters = random.randint(2,11)#范围是[2,10]
        if singleName == 'kmeans':
            predicted_label = KMeans(n_clusters=clusters).fit_predict(sampledData[i])
        elif singleName in ('ward','complete','average'):
            predicted_label = AgglomerativeClustering(linkage=singleName, n_clusters=clusters).fit_predict(sampledData[i])

        predicted_labelAll.append(predicted_label.tolist())##对采样出来的数据集的预测标签集合



    assinALLNnLabels = []#全部的通过近邻分配的标签

    #remainedData和sampleedData拥有的数据的行数是一致的，所以j的值无论从len(remainedData)还是从len(sampledData)取都可以
    for j in range(len(remainedData)):
        assinNnLabels = []  # 通过近邻分配的标签
        for m in range(len(remainedData[j])):
            minDist = inf;
            minindex = -1
            for k in range(len(sampledData[j])):
                distJI = distEclud(remainedData[j][m], sampledData[j][k])
                if distJI < minDist:
                    minDist = distJI
                    minindex = k
            assinNnLabels.append(predicted_labelAll[j][minindex])#对除采样外的数据集的根据近邻关系得到的预测标签集合
        assinALLNnLabels.append(assinNnLabels)

    #对两个预测标签和序列值分别进行组合
    combineIndex = []
    combinedLables = []
    for column in range(len(predicted_labelAll)):
        combineIndexOne = sampledIndex[column] + remainedIndex[column]
        combinedLablesOne = predicted_labelAll[column] + assinALLNnLabels[column]
        combineIndex.append(combineIndexOne)
        combinedLables.append(combinedLablesOne)
    #把打乱的序号按照从小到大排列出来，得到元素升序的序列值
    seqIndexAll = []
    for combineIndex1 in combineIndex:
        seqIndex = []
        for seq in range(len(sampledData[0]) + len(remainedData[0])):
            for elementIndex in range(len(combineIndex1)):
                if combineIndex1[elementIndex] == seq:
                    seqIndex.append(elementIndex)
        seqIndexAll.append(seqIndex)

    #得到真正的sampledData和remainedData组合后的标签值
    finalLabel = []
    for finalIndex in range(len(combinedLables)):
        finallabelone = []
        for index in seqIndexAll[finalIndex]:
            finallabelone.append(combinedLables[finalIndex][index])
        finalLabel.append(finallabelone) #最终聚类结果
    return finalLabel

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB) 计算欧几里得距离

def matrixplus(mat1,mat2):
    result = zeros((len(mat1),len(mat1)))
    # 迭代输出行
    for i in range(len(mat1)):
        # 迭代输出列
        for j in range(len(mat1[0])):
            result[i][j] = mat1[i][j]+mat2[i][j]
    return result

def computeAve(valuearr):
    sum = 0
    for value in valuearr:
        sum +=value
    ave = float(sum)/len(valuearr)
    return  ave

def multirun(datasetName,times):
    # init_population,init_ari,datamat,datalabels = ini_Cluster(kNumber=6) #多种聚类算法产生初始种群
    # datamat, datalabels = loadDataset("../dataset/cmc.data")

    datamat,datalabels = loadDataset(path)
    print 'data ready'
    # datalabels_to_float = list(map(lambda x: float(x), datalabels))
    ari_arr = []
    nmi_arr = []
    for ci in range(times):

        sampledData, remainedData, sampledIndex, remainedIndex = data_sample(datamat, 1, 40)
        print 'sampledData ready'
        #
        pop_kmeans = rsnn(sampledData, remainedData, sampledIndex, remainedIndex, 'kmeans')
        print 'kmeans end'
        # pop_ward = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'ward')
        # print 'ward end'
        # pop_complete = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'complete')
        # print 'complete end'
        # pop_average = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'average')
        # print 'average end'
        pop = []
        pop.extend(pop_kmeans)
        # pop.extend(pop_ward)
        # pop.extend(pop_complete)
        # pop.extend(pop_average)


        pop = np.array(pop)
        similaritymat = zeros((len(datamat), len(datamat)))
        avesimilaritymat = zeros((len(datamat), len(datamat)))
        popmat = zeros((len(pop), len(datamat), len(datamat)))  # n*n　matrix size
        clusNum = zeros(len(pop))
        for index in range(len(pop)):
            clusNum[index] = len(set(pop[index]))
            for i in range(len(datamat)):
                for j in range(len(datamat)):
                    if pop[index][i] == pop[index][j]:
                        popmat[index][i][j] = 1
                    else:
                        popmat[index][i][j] = 0

        for n in range(len(popmat)):
            similaritymat = matrixplus(similaritymat, popmat[n])

        a = 0.025
        avesimilaritymat = a * similaritymat  # 计算每个矩阵的平均相似度
        avesimilaritymat = np.array(avesimilaritymat)
        consensus_clustering_labels = SpectralClustering(n_clusters=10, affinity="precomputed").fit_predict(avesimilaritymat)
        nmi = normalized_mutual_info_score(datalabels, consensus_clustering_labels)
        ari = adjusted_rand_score(datalabels, consensus_clustering_labels)
        print 'nmi值:'
        print nmi
        print 'ari值:'
        print ari
        nmi_arr.append(nmi)
        ari_arr.append(ari)
        print '第%s次运行结束'%(ci+1)
    ave_nmi = computeAve(nmi_arr)
    ave_ari = computeAve(ari_arr)
    print 'nmi数次运行结果为：%s'%nmi_arr
    print 'nmi%s次的运行平均值为%s'%(times,ave_nmi)
    print 'ari数次运行结果为：%s'%ari_arr
    print 'ari%s次的运行平均值为%s'%(times,ave_ari)
#####################################################################################################
def main():
    multirun('10cardiotocography.data', 10)



if __name__ == "__main__":
    main()