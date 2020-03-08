# coding:utf-8
'''
Created on 2018年3月23日

@author: David
'''
import random as rd
from numpy import *
import numpy as np
# from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import  KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
from deap import base
from deap import creator
from deap import tools
from mocle.index_compute import mocle_index,Euclidean_dist
from dsce import *
import array
import tables
import Cluster_Ensembles as CE
from Cluster_Ensembles.Cluster_Ensembles import build_hypergraph_adjacency, store_hypergraph_adjacency
from sklearn.metrics import pairwise_distances

generation = 50 #多目标优化时用到的迭代次数

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) #weights等于-1说明是最小化问题
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("evaluate", mocle_index)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("nondominated", tools.sortNondominated)
#######数据集########
def loadDataset(filename):
    fr = open(filename)
    arraylines = fr.readlines()
    numOfLines = len(arraylines)
    returnMat = zeros((numOfLines, 9))
    classlabelVector = []
    index = 0
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(',')  # 用split会返回一个list
        returnMat[index, :] = linestrlist[0:9]
        classlabelVector.append(linestrlist[9])
        index += 1
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
            sampleI = random.randint(0,length)
            sampledIndexOne.append(sampleI)
        sampledIndexOne = list(set(sampledIndexOne))#采样出来的数据的索引值,去重后
        remainedIndexOne = (list(set(allIndex).difference(set(sampledIndexOne))))
        for j in range(len(sampledIndexOne)):
            sampledDataOne.append(dataset[sampledIndexOne[j]])###############test from append to extend
        for j in range(len(remainedIndexOne)):
            remainedDataOne.append(dataset[remainedIndexOne[j]])###########test from append to extend
        sampledData.append(sampledDataOne)
        remainedData.append(remainedDataOne)
        sampledIndex.append(sampledIndexOne)
        remainedIndex.append(remainedIndexOne)

    return sampledData,remainedData,sampledIndex,remainedIndex

def rsnn(sampledData,remainedData,sampledIndex,remainedIndex,singleName):
    predicted_labelAll = []
    for i in range(len(sampledData)):
        clusters = random.randint(2,11)#范围是[2,10]
        if singleName == 'kmeans':
            data1 = sampledData[i]
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
def ini_population(data,singleName,times):
    predicted_labelAll = []
    for i in range(times):
        clusters = random.randint(2,11)#范围是[2,10]
        if singleName == 'kmeans':
            predicted_label = KMeans(n_clusters=clusters).fit_predict(data)
        elif singleName in ('ward','complete','average'):
            predicted_label = AgglomerativeClustering(linkage=singleName, n_clusters=clusters).fit_predict(data)
        predicted_labelAll.append(predicted_label.tolist())  ##对采样出来的数据集的预测标签集合
    return predicted_labelAll

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

#取得一个种群的子集
def getSubPop(pop):
    subpopArr = []
    popIndexArr = range(len(pop)) #从0开始增大的序号列表
    sublength = len(pop)/4

    for i in range(3):
        subpop = []
        subPopIndex = rd.sample(popIndexArr,sublength)
        popIndexArr = list(set(popIndexArr).difference(set(subPopIndex)))
        for element in subPopIndex:
            subpop.append(pop[element])
        subpopArr.append(subpop)
    subpop = []
    for element in popIndexArr:
        subpop.append(pop[element])
    subpopArr.append(subpop)
    return subpopArr

def computeAve(valuearr):
    sum = 0
    for value in valuearr:
        sum +=value
    ave = float(sum)/len(valuearr)
    return  ave
# def matrixmulti(value,mat):
def dsmoc(datamat):
    # datamat,datalabels = loadDataset("../dataset/glass.data")
    print 'data ready'
    # sampledData, remainedData, sampledIndex, remainedIndex= data_sample(datamat,1,10)
    # print 'sampledData ready'
    #
    # pop_kmeans = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'kmeans')
    # print 'kmeans end'
    # pop_ward = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'ward')
    # print 'ward end'
    # pop_complete = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'complete')
    # print 'complete end'
    # pop_average = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'average')
    # print 'average end'
    pop_kmeans = ini_population(datamat,'kmeans',5)
    print 'kmeans end'
    # pop_ward = ini_population(datamat, 'ward', 10)
    # print 'ward end'
    # pop_complete = ini_population(datamat, 'complete', 10)
    # print 'complete end'
    # pop_average = ini_population(datamat, 'average', 10)
    # print 'average end'
    pop = []
    pop.extend(pop_kmeans)
    # pop.extend(pop_ward)
    # pop.extend(pop_complete)
    # pop.extend(pop_average)

    init_population = []
    for indiv1 in pop:
        ind1 = creator.Individual(indiv1)
        init_population.append(ind1)

    filter_pop = filter(lambda x:len(x)>0,init_population) ##去除初始化聚类失败的结果
    population = filter_pop #population是总的种群，后续的交叉算法的结果也要添加进来

    #为里第二个目标函数所用的矩阵，每个数据点的距离矩阵，计算一半
    dataLen = len(datamat)
    distances_matrix = zeros((dataLen, dataLen))
    for datai in range(dataLen):
        for dataj in range(datai+1,dataLen):
            distances_matrix[datai][dataj] = Euclidean_dist(datamat[datai],datamat[dataj])
    # distances_matrix = pairwise_distances(datamat, metric='euclidean')  # 数据集中数据点两两之间的距离
    print "数据点距离矩阵计算完毕"
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, tile(datamat,(len(invalid_ind),1,1)),tile(distances_matrix,(len(invalid_ind),1,1)),invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # population = toolbox.select(population, len(population))
    popeliteLen = len(population)
    for i in range(generation):
        print '第%s代'%i
        popElite = toolbox.select(population, popeliteLen)
        # Vary the population
        # parentSpring = tools.selTournamentDCD(popElite, popeliteLen)
        # parentSpring = [toolbox.clone(ind) for ind in parentSpring]
        newoffspring = []
        # applying crossover

        subpopArr = getSubPop(popElite)
        count = 0  # 计数增加几个新个体用
        for subpop in subpopArr:
            #dsce做交叉算子
            a1=0.6
            a2=0.5
            transMatrix, popClusterArr_3, popClusterArr_2, clusterNumArr = transformation(datamat, subpop)
            similiarMatrix = measureSimilarity(transMatrix, popClusterArr_3, popClusterArr_2,
                                                              clusterNumArr, datamat, a1=a1)
            dictCownP = assign(similiarMatrix, a2)
            resultList = resultTransform(dictCownP, datamat)
            #其他聚类集成算子
            # hdf5_file_name = './Cluster_Ensembles.h5'
            # fileh = tables.open_file(hdf5_file_name, 'w')
            # fileh.create_group(fileh.root, 'consensus_group')
            # fileh.close()
            # subpop = np.array(subpop)
            # hypergraph_adjacency =     build_hypergraph_adjacency(subpop)
            # store_hypergraph_adjacency(hypergraph_adjacency, hdf5_file_name)
            # resultList = CE.CSPA(hdf5_file_name, subpop, verbose=True, N_clusters_max=10)
            clu = list(set(resultList))
            clulen = len(clu)
            actual_resultList = []

            if clulen > 1:
                ind_ensemble = creator.Individual(resultList)
                newoffspring.append(ind_ensemble)
                actual_resultList = resultList #只有簇的数量不是1才会有子个体
                count += 1
            if actual_resultList:
                predicted_clusternum = len(set(actual_resultList))
                ind_new = KMeans(n_clusters=predicted_clusternum).fit_predict(datamat)
                ind_new_tran = creator.Individual(ind_new)
                newoffspring.append(ind_new_tran)
                count += 1
        print "这一代增加里%s个个体"%count
        # evaluating fitness of individuals with invalid fitnesses
        invalid_ind = [ind for ind in newoffspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, tile(datamat,(len(invalid_ind),1,1)),tile(distances_matrix,(len(invalid_ind),1,1)),invalid_ind)#这里只用了未经处理的数据,没有用到真实类别
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Chossing a population for the next generation
        # population = toolbox.select(popElite + newoffspring, popeliteLen)
        population = popElite + newoffspring
    result1 = toolbox.nondominated(population,len(population))
    final_result = result1[0]
    return final_result
#####################################################################################################
