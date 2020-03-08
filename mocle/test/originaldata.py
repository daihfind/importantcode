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
from mocle.newKmeans import *
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
from mocle.index_compute import sum_Euc_dist,getCentroids

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
    returnMat = zeros((numOfLines, 35))
    classlabelVector = []
    index = 0
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(',')  # 用split会返回一个list
        returnMat[index, :] = linestrlist[0:35]
        classlabelVector.append(linestrlist[35])
        index += 1
    # print returnMat
    # print classlabelVector
    return returnMat, classlabelVector

#通过z-score formula处理数据
def z_score_standardization(data):
    scaler = preprocessing.StandardScaler().fit(data)
    standard_data = scaler.transform(data)
    return standard_data

#最小最大值规范化
def minmax_normalization(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    normalize_data = min_max_scaler.fit_transform(data)
    return normalize_data

#####根据真实类标签值的簇数量得到迭代过程的k簇值范围####
def k_range(k):
    min_clusters = 2
    max_clusters = k + 3
    if k - 2 > min_clusters:
        min_clusters = k - 2
    return min_clusters, max_clusters

#####构造皮尔森相关系数距离矩阵#####
def distance_pearson(data):
    distance = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i>j:
                pearson = stats.pearsonr(data[i], data[j])[0]
                distance.append(pearson)
    return distance

# 用kmeans初始化种群时，针对范围内的每一个设定k值分别迭代运行30次，根据ARI指标，ARI越大越好，取得效果最好的一代
def initial_kmeans(k, rand_state, data):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    kmeans_labels_arr = []  # 每一个k簇值的最好划分的集合
    for clusters in range(min_clusters, max_clusters):
        kmeans_labels = KMeans(n_clusters=clusters, random_state=rand_state).fit_predict(data)
        ind_kmeans = creator.Individual(kmeans_labels)

        kmeans_labels_arr.append(ind_kmeans)
    return kmeans_labels_arr




# 用层次聚类al,cl初始化种群时，针对范围内的每一个设定k值分别迭代运行30次，根据ARI指标，ARI越大越好，取得效果最好的一代
def initial_linkage(linkage_name, k, datamat):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    link_labels_arr = []
    for clusters in range(min_clusters, max_clusters):
        y_link = AgglomerativeClustering(linkage=linkage_name, n_clusters=clusters)
        y_link.fit(datamat)
        ind_linkage = creator.Individual(y_link.labels_)
        link_labels_arr.append(ind_linkage)
    return link_labels_arr



#利用二进制锦标赛选择种群中的两个个体用于交叉
def tournament(inputPopulation,inputIndex_arr):
    individual = []
    max = len(inputPopulation)
    selectNum = random.randint(0,max,size=2)#因为这里用到的是numpy模块的randint函数，不包含右边界，所以+1.如果是random模块，则是包含右边界的。
    if(inputIndex_arr[selectNum[0]]>=inputIndex_arr[selectNum[1]]):
        individual.extend(inputPopulation[selectNum[0]])
    else:
        individual.extend(inputPopulation[selectNum[1]])
    return individual

# 用集成聚类利用已有种群进行交叉产生新的 个体
def ensemble_crossover(population,index_arr):
    hdf5_file_name = './Cluster_Ensembles.h5'
    fileh = tables.open_file(hdf5_file_name, 'w')
    fileh.create_group(fileh.root, 'consensus_group')
    fileh.close()
    individuals = [] #用于交叉的父代个体的集合
    clusters_num = []
    # print int(round(len(population)*0.25))
    for i in range(20):
        individuals.append(tournament(population,index_arr)) #二进制锦标赛法选择出父代个体
    individuals = np.array(individuals)
    for j in range(len(individuals)):  #交叉产生的聚类簇的范围
        individual = individuals[j]
        aa =  len(set(individual))
        clusters_num.append(aa)
    sort_clustersNum=sorted(clusters_num)#sort对原list操作，但这里是set不能用sort(),只有用sorted()
    clusters_max = random.randint(sort_clustersNum[0],sort_clustersNum[-1]+1)
    hypergraph_adjacency = build_hypergraph_adjacency(individuals)
    store_hypergraph_adjacency(hypergraph_adjacency, hdf5_file_name)
    consensus_clustering_labels = CE.MCLA(hdf5_file_name, individuals, verbose=True, N_clusters_max=clusters_max)
    ind_ensemble = creator.Individual(consensus_clustering_labels)
    print('交叉后的结果是：%s'%ind_ensemble)
    return ind_ensemble

def all_ensemble(population,k):
    hdf5_file_name = './Cluster_Ensembles.h5'
    fileh = tables.open_file(hdf5_file_name, 'w')
    fileh.create_group(fileh.root, 'consensus_group')
    fileh.close()
    pop = []
    for i in range(len(population)):
        ind = []
        ind.extend(population[i])
        pop.append(ind)
    pop = np.array(pop)
    hypergraph_adjacency = build_hypergraph_adjacency(pop)
    store_hypergraph_adjacency(hypergraph_adjacency, hdf5_file_name)
    consensus_clustering_labels = CE.MCLA(hdf5_file_name, pop, verbose=True, N_clusters_max=k+2)
    return consensus_clustering_labels
########################################################################

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



def rsnn(sampledData,remainedData,sampledIndex,remainedIndex,k,singleName):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    predicted_labelAll = []
    for i in range(len(sampledData)):
        clusters = random.randint(min_clusters,max_clusters)
        # clusters = random.randint(2,11)#范围是[2,10]
        if singleName == 'Kmeans':
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
    for i in range(len(sampledData)):
        # clusters = random.randint(min_clusters,max_clusters)
        clusters = random.randint(2, 11)
        # clusters = random.randint(2,11)#范围是[2,10]
        if singleName == 'kmeans':
            predicted_label = KMeans(n_clusters=clusters).fit_predict(sampledDataFs[i])
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

#每个方法运行１０次，框架的簇的数量的取值范围为２－１０，每次随机选择一个
def multiRunSelect(sampledData, remainedData, sampledIndex, remainedIndex,sampledDataFs,singleName,data,times=10):
    # clusters = random.randint(2, 11)
    pop_all = []#预测标签的总和
    for i in range(times):
        pop = fsrsnn(sampledData, remainedData, sampledIndex, remainedIndex, sampledDataFs,singleName)
        pop_all.append(pop)

    #某一个k值运行１０次后的每个采样数据的最好聚类结果
    min_indexAll = []
    for i in range(len(pop_all[0])):#等同于重采样数据的数
        min_euc = inf
        min_index = 0
        for j in range(times):
            centroids = getCentroids(data,pop_all[j][i])
            eucValue = sum_Euc_dist(data,pop_all[j][i], centroids)
            if eucValue<min_euc:
                min_euc = eucValue #得到针对某一个处理过的数据集的times运行后得到的最小的簇内距离
                min_index = j
        min_indexAll.append(min_index)

    clusterResult = []
    for i in range(len(pop_all[0])):
        clusterResult.append(pop_all[min_indexAll[i]][i])
    return clusterResult

def initialMultiRun(data,times,singleName):
    predicted_labelAll = []
    for i in range(times):
        clusters = random.randint(2, 11)
        if singleName == "kmeans":
            predicted_label = KMeans(n_clusters=clusters).fit_predict(data)
        elif singleName in ('ward','average','complete'):
            predicted_label = AgglomerativeClustering(linkage=singleName, n_clusters=clusters).fit_predict(data)
        predicted_labelAll.append(predicted_label.tolist())
    return predicted_labelAll

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB) 计算欧几里得距离
#####################################################################################################

def main():
    # init_population,init_ari,datamat,datalabels = ini_Cluster(kNumber=6) #多种聚类算法产生初始种群
    datamat,datalabels = loadDataset("../dataset/soybean-small.data")
    print 'data ready'

    pop_kmeans = initialMultiRun(datamat,10,'kmeans')
    print 'kmeans end'
    pop_ward = initialMultiRun(datamat,10,'ward')
    print 'ward end'
    pop_complete = initialMultiRun(datamat,10,'complete')
    print 'complete end'
    pop_average = initialMultiRun(datamat,10,'average')
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

    filter_pop = filter(lambda x:len(x)>0,init_population) ##去除初始化聚类失败的结果
    population = filter_pop #population是总的种群，后续的交叉算法的结果也要添加进来


    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, tile(datamat,(len(invalid_ind),1,1)),tile(population,(len(invalid_ind),1,1)),invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population = toolbox.select(population, len(population))

    for i in range(generation):
        print '第%s代'%i
        popElite = toolbox.select(population, int(round(len(population)*0.5))) #top half from population

        # Vary the population
        parentSpring = tools.selTournamentDCD(population, len(population))
        parentSpring = [toolbox.clone(ind) for ind in parentSpring]
        newoffspring = []
        # applying crossover
        for indiv1, indiv2 in zip(parentSpring[::2], parentSpring[1::2]):
            randNum = random.random()  # generate a random number from 0 to 1
            if randNum<0.8:
                toolbox.mate(indiv1, indiv2)
                toolbox.mutate(indiv1)
                toolbox.mutate(indiv2)
                del indiv1.fitness.values, indiv2.fitness.values
                newoffspring.append(indiv1)
                newoffspring.append(indiv2)
            else:
                hdf5_file_name = './Cluster_Ensembles.h5'
                fileh = tables.open_file(hdf5_file_name, 'w')
                fileh.create_group(fileh.root, 'consensus_group')
                fileh.close()
                individuals = []
                individuals.append(indiv1)
                individuals.append(indiv2)
                individuals = np.array(individuals)
                hypergraph_adjacency = build_hypergraph_adjacency(individuals)
                store_hypergraph_adjacency(hypergraph_adjacency, hdf5_file_name)
                consensus_clustering_labels = CE.MCLA(hdf5_file_name, individuals, verbose=True,N_clusters_max=10)
                ind_ensemble = creator.Individual(consensus_clustering_labels)
                newoffspring.append(ind_ensemble)

        # evaluating fitness of individuals with invalid fitnesses
        invalid_ind = [ind for ind in newoffspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, tile(datamat,(len(invalid_ind),1,1)),tile(newoffspring,(len(invalid_ind),1,1)),invalid_ind)#这里只用了未经处理的数据,没有用到真实类别
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Chossing a population for the next generation
        population = toolbox.select(popElite + newoffspring, len(population))
    result1 = toolbox.nondominated(population,len(population))
    print len(result1)
    print result1
    print len(result1[0])
    print result1[0]
    print 'ari值'
    ari_arr = []
    max_ari = -inf
    for ind in result1[0]:
        ari = adjusted_rand_score(datalabels, ind)
        ari_arr.append(ari)
        if ari > max_ari:
            max_ari = ari
    print ari_arr
    print max_ari
    nmi_arr = []
    max_nmi = -inf
    print 'nmi值'
    for ind in result1[0]:
        nmi = normalized_mutual_info_score(datalabels, ind)
        nmi_arr.append(nmi)
        if nmi > max_nmi:
            max_nmi = nmi
    print nmi_arr
    print max_nmi
if __name__ == "__main__":
    main()