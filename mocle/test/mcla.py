#coding:utf-8
import tables
import numpy as np
from numpy import *
from Cluster_Ensembles.Cluster_Ensembles import build_hypergraph_adjacency, store_hypergraph_adjacency
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
import Cluster_Ensembles as CE
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import AgglomerativeClustering
#######数据集########
def loadDataset(filename):
    #     with open(filename,'r') as fr:
    fr = open(filename)
    #     f = fr.read()
    arraylines = fr.readlines()
    numOfLines = len(arraylines)
    returnMat = np.zeros((numOfLines, 56))
    classlabelVector = []
    index = 0
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(',')  # 用split会返回一个list
        returnMat[index, :] = linestrlist[1:57]
        classlabelVector.append(linestrlist[0])
        index += 1
    # print returnMat
    # print classlabelVector
    return returnMat, classlabelVector

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
            sampleI = np.random.randint(0,length)
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
        clusters = np.random.randint(2, 11)
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
#用作最后实验对比MCLA
def main():
    datamat,datalabels = loadDataset("../dataset/lung-cancer.data")
    print 'data ready'

    sampledData, remainedData, sampledIndex, remainedIndex = data_sample(datamat,1,10)
    print 'sampledData ready'
    pop_kmeans = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'kmeans')
    print 'kmeans end'
    pop_ward = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'ward')
    print 'ward end'
    pop_complete = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'complete')
    print 'complete end'
    pop_average = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'average')
    print 'average end'
    pop = []
    pop.extend(pop_kmeans)
    pop.extend(pop_ward)
    pop.extend(pop_complete)
    pop.extend(pop_average)
    hdf5_file_name = './Cluster_Ensembles.h5'
    fileh = tables.open_file(hdf5_file_name, 'w')
    fileh.create_group(fileh.root, 'consensus_group')
    fileh.close()

    pop = np.array(pop)
    hypergraph_adjacency = build_hypergraph_adjacency(pop)
    store_hypergraph_adjacency(hypergraph_adjacency, hdf5_file_name)
    consensus_clustering_labels = CE.MCLA(hdf5_file_name, pop, verbose=True, N_clusters_max=10)
    nmi = normalized_mutual_info_score(datalabels, consensus_clustering_labels)
    ari = adjusted_rand_score(datalabels, consensus_clustering_labels)
    print('nmi值：　')
    print(nmi)
    print('ari值：　')
    print(ari)

if __name__ == "__main__":
    main()