# coding:utf-8
'''
Created on 2018年3月23日

@author: David
'''
import random as rd
from numpy import *
import numpy as np
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
from deap import base
from deap import creator
from deap import tools
from mocle.index_compute import mocle_index
from dsce import *
import array
import tables
import Cluster_Ensembles as CE
from Cluster_Ensembles.Cluster_Ensembles import build_hypergraph_adjacency, store_hypergraph_adjacency
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
    #     with open(filename,'r') as fr:
    fr = open(filename)
    #     f = fr.read()
    arraylines = fr.readlines()
    numOfLines = len(arraylines)
    returnMat = zeros((numOfLines, 9))
    classlabelVector = []
    index = 0
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(',')  # 用split会返回一个list
        returnMat[index, :] = linestrlist[1:10]
        classlabelVector.append(linestrlist[10])
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
        clusters = random.randint(2,11)#范围是[2,10]
        # clusters = 3
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

#取得一个种群的子集
def getSubPop(pop):
    subpopArr = []
    popIndexArr = range(len(pop)) #从0开始增大的序号列表
    sublength = len(pop)/2

    for i in range(1):
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
def multirun(datasetName):
    # init_population,init_ari,datamat,datalabels = ini_Cluster(kNumber=6) #多种聚类算法产生初始种群
    # datamat,datalabels = loadDataset("../dataset/glass.data")
    path = '../dataset/'+datasetName
    datamat,datalabels = loadDataset(path)
    print 'data ready'
    # datalabels_to_float = list(map(lambda x: float(x), datalabels))

    sampledData, remainedData, sampledIndex, remainedIndex= data_sample(datamat,1,5)
    print 'sampledData ready'
    #
    pop_kmeans = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'kmeans')
    print 'kmeans end'
    max_nmi1 = -inf
    for ind1 in pop_kmeans:
        nmi1 = normalized_mutual_info_score(datalabels, ind1)
        if nmi1 > max_nmi1:
            max_nmi1 = nmi1
    print '初始kmeans最大nmi为%s'%max_nmi1
    pop_ward = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'ward')
    print 'ward end'
    max_nmi2 = -inf
    for ind2 in pop_ward:
        nmi2 = normalized_mutual_info_score(datalabels, ind2)
        if nmi2 > max_nmi2:
            max_nmi2 = nmi2
    print '初始ward最大nmi为%s'%max_nmi2
    pop_complete = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'complete')
    print 'complete end'
    max_nmi3 = -inf
    for ind3 in pop_complete:
        nmi3 = normalized_mutual_info_score(datalabels, ind3)
        if nmi3 > max_nmi3:
            max_nmi3 = nmi3
    print '初始complete最大nmi为%s'%max_nmi3
    pop_average = rsnn(sampledData, remainedData, sampledIndex, remainedIndex,'average')
    print 'average end'
    max_nmi4 = -inf
    for ind4 in pop_average:
        nmi4 = normalized_mutual_info_score(datalabels, ind4)
        if nmi4 > max_nmi4:
            max_nmi4 = nmi4
    print '初始average最大nmi为%s'%max_nmi4
    pop = []
    pop.extend(pop_kmeans)
    pop.extend(pop_ward)
    pop.extend(pop_complete)
    pop.extend(pop_average)


    init_population = []
    for indiv1 in pop:
        ind1 = creator.Individual(indiv1)
        init_population.append(ind1)

    filter_pop = filter(lambda x:len(x)>0,init_population) ##去除初始化聚类失败的结果
    population = filter_pop #population是总的种群，后续的交叉算法的结果也要添加进来


    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, tile(datamat,(len(invalid_ind),1,1)),invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # population = toolbox.select(population, len(population))
    popeliteLen = len(population)
    for i in range(generation):
        print '第%s代'%i
        popElite = toolbox.select(population, popeliteLen) #top half from population
        # Vary the population
        # parentSpring = tools.selTournamentDCD(population, len(population))
        # parentSpring = [toolbox.clone(ind) for ind in parentSpring]
        newoffspring = []
        # applying crossover

        subpopArr = getSubPop(popElite)
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
            # hypergraph_adjacency = build_hypergraph_adjacency(subpop)
            # store_hypergraph_adjacency(hypergraph_adjacency, hdf5_file_name)
            # resultList = CE.CSPA(hdf5_file_name, subpop, verbose=True, N_clusters_max=10)

            ind_ensemble = creator.Individual(resultList)
            newoffspring.append(ind_ensemble)
            predicted_clusternum = len(set(resultList))
            ind_new = KMeans(n_clusters=predicted_clusternum).fit_predict(datamat)
            # print(len(set(resultList)))
            ind_new_tran = creator.Individual(ind_new)
            newoffspring.append(ind_new_tran)
        # evaluating fitness of individuals with invalid fitnesses
        invalid_ind = [ind for ind in newoffspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, tile(datamat,(len(invalid_ind),1,1)),invalid_ind)#这里只用了未经处理的数据,没有用到真实类别
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Chossing a population for the next generation
        # population = toolbox.select(popElite + newoffspring, popeliteLen)
        population = popElite + newoffspring
    result1 = toolbox.nondominated(population,len(population))
    ari_arr = []
    max_ari = -inf
    for ind in result1[0]:
        ari = adjusted_rand_score(datalabels, ind)
        ari_arr.append(ari)
        if ari > max_ari:
            max_ari = ari
    nmi_arr = []
    max_nmi = -inf
    print 'nmi值'
    for ind in result1[0]:
        nmi = normalized_mutual_info_score(datalabels, ind)
        nmi_arr.append(nmi)
        if nmi > max_nmi:
            max_nmi = nmi
    print '最大nmi值为:%s' % max_nmi
    print nmi_arr
    return max_nmi,max_ari
#####################################################################################################
def main():
    nmi_arr = []
    ari_arr = []
    times = 10
    for i in range(times):
        nmi,ari = multirun('glass.data')
        nmi_arr.append(nmi)
        ari_arr.append(ari)
        print '第%s次运行结束' % i
    ave_nmi = computeAve(nmi_arr)
    ave_ari = computeAve(ari_arr)
    print 'nmi数次运行结果为：%s'%nmi_arr
    print 'nmi%s次的运行平均值为%s'%(times,ave_nmi)
    print 'ari数次运行结果为：%s'%ari_arr
    print 'ari%s次的运行平均值为%s'%(times,ave_ari)
if __name__ == "__main__":
    main()