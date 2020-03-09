# coding:utf-8
'''
Created on 2018年3月23日

@author: David
'''
import random as rd
import Cluster_Ensembles as CE

from Cluster_Ensembles.Cluster_Ensembles import build_hypergraph_adjacency, store_hypergraph_adjacency
from numpy import *
import numpy as np
# from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
from deap import base
from deap import creator
from deap import tools
from mocle.index_compute import *
from dsce import *
import array
import tables
# import Cluster_Ensembles as CE
# from Cluster_Ensembles.Cluster_Ensembles import build_hypergraph_adjacency, store_hypergraph_adjacency
from sklearn.metrics import pairwise_distances

generation = 20 #多目标优化时用到的迭代次数

# creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0,-1.0)) #weights等于-1说明是最小化问题
# creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

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
def ini_population(data,singleName,times):
    predicted_labelAll = []
    for i in range(times):
        clusters = random.randint(3,6)#范围是[2,10]
        # clusters = 4
        if singleName == 'kmeans':
            predicted_label = KMeans(n_clusters=clusters).fit_predict(data)
        elif singleName in ('ward','complete','average'):
            predicted_label = AgglomerativeClustering(linkage=singleName, n_clusters=clusters).fit_predict(data)
        elif singleName == "spc":
            predicted_label = SpectralClustering(n_clusters=clusters).fit_predict(data)
        predicted_labelAll.append(predicted_label.tolist())  ##对采样出来的数据集的预测标签集合
    return predicted_labelAll

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB) 计算欧几里得距离




def computePBM(datamat,finalresult):
    maxValue = -inf
    index = 0
    resultIndex = 0
    for element in finalresult:
        record = list(set(element))
        c = len(record)
        ec = sum_Euc_distForSegmentation(datamat,element)
        e1 = computeE1(datamat)
        centroids = getCentroids(datamat,element)
        max_sep = getmax_sep(centroids)
        value = float(float(1.0/float(c))*float(float(e1)/float(ec))*float(max_sep))
        # value = float(float(e1/float(ec))*float(max_sep))

        if(value>maxValue):
            maxValue = value
            resultIndex = index
        index += 1
    return finalresult[resultIndex],maxValue





# def matrixmulti(value,mat):
def moclenew(datamat):
    # datamat,datalabels = loadDataset("../dataset/glass.data")
    print 'data ready'
    pop_kmeans = ini_population(datamat,'kmeans',10)
    print 'kmeans end'
    pop_ward = ini_population(datamat, 'ward', 10)
    print 'ward end'
    pop_complete = ini_population(datamat, 'complete', 10)
    print 'complete end'
    pop_average = ini_population(datamat, 'average', 10)
    print 'average end'
    # pop_spc = ini_population(datamat, 'spc', 1)
    # print 'spc end'
    pop = []
    pop.extend(pop_kmeans)
    pop.extend(pop_complete)
    pop.extend(pop_average)
    # pop.extend(pop_spc)
    init_population = []
    for indiv1 in pop:
        ind1 = creator.Individual(indiv1)
        init_population.append(ind1)

    filter_pop = filter(lambda x:len(x)>0,init_population) ##去除初始化聚类失败的结果
    population = filter_pop #population是总的种群，后续的交叉算法的结果也要添加进来

    #为里第二个目标函数所用的矩阵，每个数据点的距离矩阵，计算一半
    # dataLen = len(datamat)
    # distances_matrix = zeros((dataLen, dataLen))
    # for datai in range(dataLen):
    #     for dataj in range(datai+1,dataLen):
    #         distances_matrix[datai][dataj] = Euclidean_dist(datamat[datai],datamat[dataj])
    distances_matrix = pairwise_distances(datamat, metric='euclidean')  # 数据集中数据点两两之间的距离
    print "数据点距离矩阵计算完毕"
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid_ind:
        euDistance,eu_connect = mocle_index(datamat,distances_matrix,ind)
        fitnesses = (euDistance,eu_connect)
        ind.fitness.values = fitnesses
    # fitnesses = toolbox.map(toolbox.evaluate, tile(datamat,(len(invalid_ind),1,1)),tile(distances_matrix,(len(invalid_ind),1,1)),invalid_ind)
    #
    # for ind, fit in zip(invalid_ind, fitnesses):
    #     ind.fitness.values = fit

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
        popcrossover = toolbox.select(population, 2)

        k1 = len(list(set(popcrossover[0])))
        k2 = len(list(set(popcrossover[1])))
        if k1<=k2:
            k = random.randint(k1,k2+1)
        else:
            k = random.randint(k2,k1+1)
        # 其他聚类集成算子
        hdf5_file_name = './Cluster_Ensembles.h5'
        fileh = tables.open_file(hdf5_file_name, 'w')
        fileh.create_group(fileh.root, 'consensus_group')
        fileh.close()
        popcrossover = np.array(popcrossover)
        hypergraph_adjacency = build_hypergraph_adjacency(popcrossover)
        store_hypergraph_adjacency(hypergraph_adjacency, hdf5_file_name)
        resultList = CE.CSPA(hdf5_file_name, popcrossover, verbose=True, N_clusters_max=k)
        ind_ensemble = creator.Individual(resultList)
        newoffspring.append(ind_ensemble)

        # evaluating fitness of individuals with invalid fitnesses
        invalid_ind = [ind for ind in newoffspring if not ind.fitness.valid]
        for ind1 in invalid_ind:
            euDistance1, eu_connect1 = mocle_index(datamat, distances_matrix, ind1)
            fitnesses1 = (euDistance1, eu_connect1)
            ind1.fitness.values = fitnesses1


        # fitnesses = toolbox.map(toolbox.evaluate, tile(datamat,(len(invalid_ind),1,1)),tile(distances_matrix,(len(invalid_ind),1,1)),invalid_ind)#这里只用了未经处理的数据,没有用到真实类别
        #
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit

        # Chossing a population for the next generation
        # population = toolbox.select(popElite + newoffspring, popeliteLen)
        population = popElite + newoffspring
    result1 = toolbox.nondominated(population,len(population))
    nondominated_result = result1[0]
    final_result,pbmValue = computePBM(datamat,nondominated_result)
    return final_result,pbmValue
    # return nondominated_result
#####################################################################################################
