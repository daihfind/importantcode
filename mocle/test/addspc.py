# coding:utf-8
'''
Created on 2018年3月23日

@author: David
'''
from numpy import *
import numpy as np
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import AgglomerativeClustering
import Cluster_Ensembles as CE
import tables
from Cluster_Ensembles.Cluster_Ensembles import build_hypergraph_adjacency, store_hypergraph_adjacency
from sklearn import preprocessing
from mocle.index_compute import mocle_index,corrected_rand
from deap import base
from deap import creator
from deap import tools
import array
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
from mocle.index_compute import sum_Euc_dist,getCentroids
from sklearn.cluster import SpectralClustering

NDIM = 9
generation = 50 #多目标优化时用到的迭代次数

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) #weights等于-1说明是最小化问题
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("evaluate", mocle_index)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit,  indpb=1.0 / NDIM)
toolbox.register("nondominated", tools.sortNondominated)
#######数据集########

def loadDataset(filename):
    #     with open(filename,'r') as fr:
    fr = open(filename)
    #     f = fr.read()
    arraylines = fr.readlines()
    numOfLines = len(arraylines)
    returnMat = zeros((numOfLines, 16))
    classlabelVector = []
    index = 0
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(',')  # 用split会返回一个list
        returnMat[index, :] = linestrlist[1:17]
        classlabelVector.append(linestrlist[17])
        index += 1
    # print returnMat
    # print classlabelVector
    return returnMat, classlabelVector

#random sampling with replacement
#dateset为重置采样(可放回)的数据集,rate为采样的概率,ensembleSize为针对一个数据集要采样的次数
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



def data_samplefs(dataset,rate,ensembleSize):
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
    featureIndex = []
    for i in range(len(sampledData)):
        featureIndexOne = []
        for j in range(len(sampledData[i][0])):
            featureI = random.randint(0,len(sampledData[i][0]))
            featureIndexOne.append(featureI)
        featureIndexOne = list(set(featureIndexOne))
        featureIndex.append(featureIndexOne)
    sampledDataFs = []
    for i in range(len(featureIndex)):#特征的向量行数等于要进行特征选择的已采样数据集数量
        sampledDataFsOne = []
        for oneSampleRow in sampledData[i]:
            oneSampleFsOne = []
            for fIndex in featureIndex[i]:
                oneSampleFsOne.append(oneSampleRow[fIndex])
            sampledDataFsOne.append(oneSampleFsOne)
        sampledDataFs.append(sampledDataFsOne)

    return sampledData,remainedData,sampledIndex,remainedIndex,sampledDataFs

def fsrsnn(sampledData,remainedData,sampledIndex,remainedIndex,sampledDataFs,singleName):
    predicted_labelAll = []
    for i in range(len(sampledDataFs)):
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
                distJI = distEclud(remainedData[j][m], sampledData[j][k])  # 计算质心和数据点之间的距离
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

def matrixcompute(datalength,population):
    clusters = random.randint(2, 11)
    similaritymat = zeros((datalength, datalength))
    avesimilaritymat = zeros((datalength, datalength))
    popmat = zeros((len(population), datalength, datalength))  # n*n　matrix size
    clusNum = zeros(len(population))
    for index in range(len(population)):
        clusNum[index] = len(set(population[index]))
        for i in range(datalength):
            for j in range(datalength):
                if population[index][i] == population[index][j]:
                    popmat[index][i][j] = 1
                else:
                    popmat[index][i][j] = 0

    for n in range(len(popmat)):
        similaritymat = matrixplus(similaritymat, popmat[n])

    a = 0.5
    avesimilaritymat = a * similaritymat  # 计算每个矩阵的平均相似度
    avesimilaritymat = np.array(avesimilaritymat)
    consensus_clustering_labels = SpectralClustering(n_clusters=7, affinity="precomputed").fit_predict(avesimilaritymat)
    return consensus_clustering_labels


def multirun(datasetName):
    # init_population,init_ari,datamat,datalabels = ini_Cluster(kNumber=6) #多种聚类算法产生初始种群
    # datamat, datalabels = loadDataset("../dataset/lung-cancer.data")
    path = '../dataset/'+datasetName
    datamat,datalabels = loadDataset(path)
    # print 'data ready'
    # datalabels_to_float = list(map(lambda x: float(x), datalabels))

    sampledData, remainedData, sampledIndex, remainedIndex, sampledDataFs = data_samplefs(datamat, 1, 10)
    # print 'sampledData ready'

    pop_kmeans = fsrsnn(sampledData, remainedData, sampledIndex, remainedIndex, sampledDataFs, 'kmeans')
    # print 'kmeans end'
    pop_ward = fsrsnn(sampledData, remainedData, sampledIndex, remainedIndex, sampledDataFs, 'ward')
    # print 'ward end'
    pop_complete = fsrsnn(sampledData, remainedData, sampledIndex, remainedIndex, sampledDataFs, 'complete')
    # print 'complete end'
    pop_average = fsrsnn(sampledData, remainedData, sampledIndex, remainedIndex, sampledDataFs, 'average')
    print 'average end'
    pop = []
    pop.extend(pop_kmeans)
    pop.extend(pop_ward)
    pop.extend(pop_complete)
    pop.extend(pop_average)

    init_population = []
    for indiv1 in pop:
        ind1 = creator.Individual(indiv1)
        init_population.append(ind1)

    filter_pop = filter(lambda x: len(x) > 0, init_population)  ##去除初始化聚类失败的结果
    population = filter_pop  # population是总的种群，后续的交叉算法的结果也要添加进来

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, tile(datamat, (len(invalid_ind), 1, 1)),
                            tile(population, (len(invalid_ind), 1, 1)), invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population = toolbox.select(population, len(population))

    for i in range(generation):
        print '第%s代' % i
        popElite = toolbox.select(population, int(round(len(population) * 0.5)))  # top half from population

        # Vary the population
        parentSpring = tools.selTournamentDCD(population, len(population))
        parentSpring = [toolbox.clone(ind) for ind in parentSpring]
        newoffspring = []
        # applying crossover
        for indiv1, indiv2 in zip(parentSpring[::2], parentSpring[1::2]):
            popCrossover = [indiv1, indiv2]
            ind_spc = matrixcompute(len(datamat), popCrossover)
            ind_spc = creator.Individual(ind_spc)

            newoffspring.append(ind_spc)
        # evaluating fitness of individuals with invalid fitnesses
        invalid_ind = [ind for ind in newoffspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, tile(datamat, (len(invalid_ind), 1, 1)),
                                tile(newoffspring, (len(invalid_ind), 1, 1)), invalid_ind)  # 这里只用了未经处理的数据,没有用到真实类别
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Chossing a population for the next generation
        population = toolbox.select(popElite + newoffspring, len(population))
    result1 = toolbox.nondominated(population, len(population))
    ari_arr = []
    max_ari = -inf
    for ind in result1[0]:
        ari = adjusted_rand_score(datalabels, ind)
        ari_arr.append(ari)
        if ari > max_ari:
            max_ari = ari
    nmi_arr = []
    max_nmi = -inf
    for ind in result1[0]:
        nmi = normalized_mutual_info_score(datalabels, ind)
        nmi_arr.append(nmi)
        if nmi > max_nmi:
            max_nmi = nmi
    print '最大nmi值为:%s'%max_nmi
    return max_nmi

def computeAve(valuearr):
    sum = 0
    for value in valuearr:
        sum +=value
    ave = float(sum)/len(valuearr)
    return  ave
#####################################################################################################

def main():
    nmi_arr = []
    times = 10
    for i in range(times):
        nmi = multirun('zoo.data')
        nmi_arr.append(nmi)
        print '第%s次运行结束' % i
    ave = computeAve(nmi_arr)
    print 'nmi数次运行结果为：%s'%nmi_arr
    print '%s次的运行平均值为%s'%(times,ave)

if __name__ == "__main__":
    main()