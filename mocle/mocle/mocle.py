# coding:utf-8
'''
Created on 2018年3月23日

@author: David
'''
from numpy import *
import numpy as np
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn import manifold
# from scipy.cluster._hierarchy import linkage
import Cluster_Ensembles as CE
import tables
from Cluster_Ensembles.Cluster_Ensembles import build_hypergraph_adjacency, store_hypergraph_adjacency
from sklearn import preprocessing
from index_compute import mocle_index,corrected_rand
from deap import base
from deap import creator
from deap import tools
import array
from scipy import stats
import scipy.spatial.distance as dist
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score


ini_generation = 30 #初始化种群时用到的迭代次数
generation = 50 #多目标优化时用到的迭代次数

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0,-1.0,-1.0)) #weights等于-1说明是最小化问题
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("evaluate", mocle_index)
toolbox.register("select", tools.selNSGA2)
#######数据集########

def loadDataset(filename):
    #     with open(filename,'r') as fr:
    fr = open(filename)
    #     f = fr.read()
    arraylines = fr.readlines()
    numOfLines = len(arraylines)
    returnMat = zeros((numOfLines, 12))
    classlabelVector = []
    index = 0
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(',')  # 用split会返回一个list
        returnMat[index, :] = linestrlist[1:13]
        classlabelVector.append(linestrlist[0])
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
#####用皮尔森相关系数版本的进行聚类初始化#####
def initial_kmeansPear(k, data, reallabels):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    bestAri_arr = []  # 每一个k簇值ari最好值的集合
    # bestCr_arr = [] #每一个k簇值ＣＲ最好值的集合
    kmeans_labels = []  # 某一k簇值得到的最好的划分
    kmeans_labels_arr = []  # 每一个k簇值的最好划分的集合
    for clusters in range(min_clusters, max_clusters):
        bestAri = 0  # 某一k簇值中的ari最好值
        # bestCr = -1 #某一k簇值中的ＣＲ最好值
        for i in range(ini_generation):
            y_kmeans = kMeans_Pearson(data,clusters)
            # y_kmeans = KMeans(n_clusters=clusters, random_state=rand_state).fit_predict(data)
            kmeans_ari = adjusted_rand_score(reallabels, y_kmeans)
            # kmeans_cr = corrected_rand(reallabels, y_kmeans)
            if kmeans_ari > bestAri:
                bestAri = kmeans_ari
                kmeans_labels = y_kmeans
            # if kmeans_cr > bestCr:
            #     bestCr = kmeans_cr
        bestAri_arr.append(bestAri)
        # bestCr_arr.append(bestCr)
        ind_kmeans = creator.Individual(kmeans_labels)
        kmeans_labels_arr.append(ind_kmeans)
    # print ('皮尔森版本的kmeans的最好ＣＲ值为:%s'%bestCr_arr)
    return kmeans_labels_arr, bestAri_arr
###外部传入距离矩阵进行层次聚类######
###该实验传入皮尔森相关系数的距离矩阵进行聚类#####
def precomputed_linkage(linkage_name, k, dist_matrix, datalabels):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    linkAri_arr = []
    # linkCr_arr = []
    link_labels_arr = []
    for clusters in range(min_clusters, max_clusters):
        y_link = AgglomerativeClustering(linkage=linkage_name,affinity="precomputed", n_clusters=clusters)
        labels = y_link.fit_predict(dist_matrix)
        link_ari = adjusted_rand_score(datalabels, labels)
        # link_cr = corrected_rand(datalabels, labels)
        # link_ari = corrected_rand(datalabels, labels)
        linkAri_arr.append(link_ari)
        # linkCr_arr.append(link_cr)
        ind_linkage = creator.Individual(labels)
        link_labels_arr.append(ind_linkage)
    # print ('皮尔森版本的层次的%s的ＣＲ值为%s:' % (linkage_name, linkCr_arr))
    return link_labels_arr, linkAri_arr

####外部传入距离矩阵进行谱聚类####
####该实验传入皮尔森相关系数的距离矩阵进行聚类####
def precomputed_spc(k, dist_matrix, datalabels):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    bestAri_arr = []  # 每一个k簇值ari最好值的集合
    # bestCr_arr = [] #每一个k簇值ＣＲ最好值的集合
    spc_labels = []  # 某一k簇值得到的最好的划分
    spc_labels_arr = []  # 每一个k簇值的最好划分的集合
    for clusters in range(min_clusters, max_clusters):
        bestAri = 0  # 某一k簇值中的ari最好值
        # bestCr = -1 #某一k簇值中的ＣＲ最好值
        for i in range(ini_generation):
            y_spc = SpectralClustering(n_clusters=clusters,affinity="precomputed").fit_predict(dist_matrix)
            spc_ari = adjusted_rand_score(datalabels, y_spc)
            # spc_cr = corrected_rand(datalabels, y_spc)
            if spc_ari > bestAri:
                bestAri = spc_ari
                spc_labels = y_spc
        #     if spc_cr > bestCr:
        #         bestCr = spc_cr
        # bestCr_arr.append(bestCr)
        bestAri_arr.append(bestAri)
        ind_spc = creator.Individual(spc_labels)
        spc_labels_arr.append(ind_spc)
    # print ("皮尔森版本的谱聚类的最好ＣＲ值为%s"%bestCr_arr)
    return spc_labels_arr, bestAri_arr
# 用kmeans初始化种群时，针对范围内的每一个设定k值分别迭代运行30次，根据ARI指标，ARI越大越好，取得效果最好的一代
def initial_kmeans(k, rand_state, data, reallabels):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    bestAri_arr = []  # 每一个k簇值ari最好值的集合
    # bestCr_arr = [] #每一个k簇值ＣＲ最好值的集合
    kmeans_labels = []  # 某一k簇值得到的最好的划分
    kmeans_labels_arr = []  # 每一个k簇值的最好划分的集合
    for clusters in range(min_clusters, max_clusters):
        bestAri = 0  # 某一k簇值中的ari最好值
        # bestCr = -1
        for i in range(ini_generation):
            y_kmeans = KMeans(n_clusters=clusters, random_state=rand_state).fit_predict(data)
            kmeans_ari = adjusted_rand_score(reallabels, y_kmeans)
            # kmeans_cr = corrected_rand(reallabels, y_kmeans)
            if kmeans_ari > bestAri:
                bestAri = kmeans_ari
                kmeans_labels = y_kmeans
            # if kmeans_cr > bestCr:
            #     bestCr = kmeans_cr
        # bestCr_arr.append(bestCr)
        bestAri_arr.append(bestAri)
        ind_kmeans = creator.Individual(kmeans_labels)
        kmeans_labels_arr.append(ind_kmeans)
    # print ('kmeans的最好ＣＲ值为:%s'%bestCr_arr)
    return kmeans_labels_arr, bestAri_arr


# 用spc初始化种群时，针对范围内的每一个设定k值分别迭代运行30次，根据ARI指标，ARI越大越好，取得效果最好的一代
def initial_spc(k, datamat, datalabels):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    bestAri_arr = []  # 每一个k簇值ari最好值的集合
    # bestCr_arr = [] #每一个k簇值ari最好值的集合
    spc_labels = []  # 某一k簇值得到的最好的划分
    spc_labels_arr = []  # 每一个k簇值的最好划分的集合
    for clusters in range(min_clusters, max_clusters):
        # print clusters
        bestAri = 0  # 某一k簇值中的ari最好值
        # bestCr = -1 #某一k簇值中的ＣＲ最好值
        for i in range(ini_generation):
            y_spc = SpectralClustering(n_clusters=clusters).fit_predict(datamat)
            spc_ari = adjusted_rand_score(datalabels, y_spc)
            # spc_cr = corrected_rand(datalabels, y_spc)
            if spc_ari > bestAri:
                bestAri = spc_ari
                spc_labels = y_spc
            # if spc_cr > bestCr:
            #     bestCr = spc_cr
        # bestCr_arr.append(bestCr)
        bestAri_arr.append(bestAri)
        ind_spc = creator.Individual(spc_labels)
        spc_labels_arr.append(ind_spc)
    # print ('谱聚类的最好ＣＲ值为%s'%bestCr_arr)
    return spc_labels_arr, bestAri_arr


# 用层次聚类al,cl初始化种群时，针对范围内的每一个设定k值分别迭代运行30次，根据ARI指标，ARI越大越好，取得效果最好的一代
def initial_linkage(linkage_name, k, datamat, datalabels):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    linkAri_arr = []
    # linkCr_arr = []
    link_labels_arr = []
    for clusters in range(min_clusters, max_clusters):
        y_link = AgglomerativeClustering(linkage=linkage_name, n_clusters=clusters)
        y_link.fit(datamat)
        link_ari = adjusted_rand_score(datalabels, y_link.labels_)
        # link_cr = corrected_rand(datalabels, y_link.labels_)
        # linkCr_arr.append(link_cr)
        linkAri_arr.append(link_ari)
        ind_linkage = creator.Individual(y_link.labels_)
        link_labels_arr.append(ind_linkage)
    # print('层次聚类%s的ＣＲ值为%s'%(linkage_name,linkCr_arr))
    return link_labels_arr, linkAri_arr


# 用kmeans,average linkage,complete linkage，SPC谱聚类四种方法初始化种群
def ini_Cluster(kNumber): #kNumber为真实类别簇的数量
    #数据集
    dataMat, dataLabels = loadDataset("../dataset/wine.data")
    #数据预处理
    z_score_data = z_score_standardization(dataMat)
    minmax_data = minmax_normalization(dataMat)


    # dataMat = manifold.SpectralEmbedding(n_components=6).fit_transform(dataMat)#降维,每一次降维的结果都会不同
    #     graph = image.img_to_graph(dataMat_trans)
    ######初始化聚类方法的皮尔森相关系数版本#######
    dist_list = distance_pearson(dataMat)
    dist_matrix = dist.squareform(dist_list)  # dist_list 为n*（n-1）/2大小的list
    y_kmeansPear, kmeans_ariPear = initial_kmeansPear(k=kNumber, data=dataMat, reallabels=dataLabels)
    y_al_pear, al_ari_pear = precomputed_linkage(linkage_name='average', k=kNumber, dist_matrix=dist_matrix, datalabels=dataLabels)
    y_cl_pear, cl_ari_pear = precomputed_linkage(linkage_name='complete', k=kNumber, dist_matrix=dist_matrix, datalabels=dataLabels)
    y_spc_pear, spc_ari_pear = precomputed_spc(k=kNumber, dist_matrix=dist_matrix, datalabels=dataLabels)  # 谱聚类初始化种群时，运行30次选择结果最好的


    ######初始化聚类方法的欧氏距离版本
    #未经处理过的数据
    y_kmeans, kmeans_ari = initial_kmeans(k=kNumber, rand_state=None, data=dataMat, reallabels=dataLabels)
    y_al, al_ari = initial_linkage(linkage_name='average', k=kNumber, datamat=dataMat, datalabels=dataLabels)
    y_cl, cl_ari = initial_linkage(linkage_name='complete', k=kNumber, datamat=dataMat, datalabels=dataLabels)
    y_spc, spc_ari = initial_spc(k=kNumber, datamat=dataMat, datalabels=dataLabels)  # 谱聚类初始化种群时，运行30次选择结果最好的
    #经过z-score-formula处理的数据
    y_kmeans1, kmeans_ari1 = initial_kmeans(k=kNumber, rand_state=None, data=z_score_data, reallabels=dataLabels)
    y_al1, al_ari1 = initial_linkage(linkage_name='average', k=kNumber, datamat=z_score_data, datalabels=dataLabels)
    y_cl1, cl_ari1 = initial_linkage(linkage_name='complete', k=kNumber, datamat=z_score_data, datalabels=dataLabels)
    y_spc1, spc_ari1 = initial_spc(k=kNumber, datamat=z_score_data, datalabels=dataLabels)  # 谱聚类初始化种群时，运行30次选择结果最好的
    #经过最小最大规范化，每个特征缩放数据范围到[0,1]
    y_kmeans2, kmeans_ari2 = initial_kmeans(k=kNumber, rand_state=None, data=minmax_data, reallabels=dataLabels)
    y_al2, al_ari2 = initial_linkage(linkage_name='average', k=kNumber, datamat=minmax_data, datalabels=dataLabels)
    y_cl2, cl_ari2 = initial_linkage(linkage_name='complete', k=kNumber, datamat=minmax_data, datalabels=dataLabels)
    y_spc2, spc_ari2 = initial_spc(k=kNumber, datamat=minmax_data, datalabels=dataLabels)  # 谱聚类初始化种群时，运行30次选择结果最好的
    #四种聚类的初始化种群,分别对应3种数据处理方式
    initial_population = []
    initial_population.extend(y_kmeansPear)
    initial_population.extend(y_al_pear)
    initial_population.extend(y_cl_pear)
    initial_population.extend(y_spc_pear)
    initial_population.extend(y_kmeans)
    initial_population.extend(y_al)
    initial_population.extend(y_cl)
    initial_population.extend(y_spc)
    initial_population.extend(y_kmeans1)
    initial_population.extend(y_al1)
    initial_population.extend(y_cl1)
    initial_population.extend(y_spc1)
    initial_population.extend(y_kmeans2)
    initial_population.extend(y_al2)
    initial_population.extend(y_cl2)
    initial_population.extend(y_spc2)

    #初始化种群对应的ari指标值，分别对应三种数据处理方式
    initial_ari = []
    initial_ari.extend(kmeans_ariPear)
    initial_ari.extend(al_ari_pear)
    initial_ari.extend(cl_ari_pear)
    initial_ari.extend(spc_ari_pear)
    initial_ari.extend(kmeans_ari)
    initial_ari.extend(al_ari)
    initial_ari.extend(cl_ari)
    initial_ari.extend(spc_ari)
    initial_ari.extend(kmeans_ari1)
    initial_ari.extend(al_ari1)
    initial_ari.extend(cl_ari1)
    initial_ari.extend(spc_ari1)
    initial_ari.extend(kmeans_ari2)
    initial_ari.extend(al_ari2)
    initial_ari.extend(cl_ari2)
    initial_ari.extend(spc_ari2)
    print ('皮尔森相关系数版本的kmeans的ARI指标为：%s' % (kmeans_ariPear))
    print ('kmeans的ARI指标为：%s' % (kmeans_ari))
    print ('kmeans经过z-score formula处理后的数据ARI指标:%s'%(kmeans_ari1))
    print ('kmeans经过最小最大规范化处理后的数据ARI指标:%s'%(kmeans_ari2))
    print ('皮尔森相关系数版本的al的ARI指标为：%s' % (al_ari_pear))
    print ('al的ARI指标为：%s' % (al_ari))
    print ('al经过z-score formula处理后的数据ARI指标:%s'%(al_ari1))
    print ('al经过最小最大规范化处理后的数据ARI指标:%s'%(al_ari2))
    print ('皮尔森相关系数版本的cl的ARI指标为：%s' % (cl_ari_pear))
    print ('cl的ARI指标为：%s' % (cl_ari))
    print ('cl经过z-score formula处理后的数据ARI指标:%s'%(cl_ari1))
    print ('cl经过最小最大规范化处理后的数据ARI指标:%s'%(cl_ari2))
    print ('皮尔森相关系数版本的SPC的ARI指标为：%s' % (spc_ari_pear))
    print ('SPC的ARI指标为：%s' % (spc_ari))
    print ('SPC经过z-score formula处理后的数据ARI指标:%s'%(spc_ari1))
    print ('SPC经过最小最大规范化处理后的数据ARI指标:%s'%(spc_ari2))
    print ('********************************')
    print ('第一代初始化种群为：%s'%(initial_population))
    print ('第一代初始化种群对应的ari值为：%s'%initial_ari)
    return initial_population,initial_ari,dataMat,dataLabels

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

# 用集成聚类利用已有种群进行交叉产生新的个体
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



def rsnn(sampledData,remainedData,sampledIndex,remainedIndex,k):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    predicted_labelAll = []
    for i in range(len(sampledData)):
        clusters = random.randint(min_clusters,max_clusters)
        # clusters = random.randint(2,11)#范围是[2,10]
        predicted_label = KMeans(n_clusters=clusters).fit_predict(sampledData[i])

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

def fsrsnn(sampledData,remainedData,sampledIndex,remainedIndex,sampledDataFs,k):
    min_clusters, max_clusters = k_range(k)  # 根据真实类标签数得到实验所用的簇数量范围
    predicted_labelAll = []
    for i in range(len(sampledData)):
        clusters = random.randint(min_clusters,max_clusters)
        # clusters = random.randint(2,11)#范围是[2,10]
        predicted_label = KMeans(n_clusters=clusters).fit_predict(sampledDataFs[i])

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
#####################################################################################################

def main():
    init_population,init_ari,datamat,datalabels = ini_Cluster(kNumber=3) #多种聚类算法产生初始种群

    # datamat, datalabels = loadDataset("../dataset/glass.data")
    #
    # sampledData, remainedData, sampledIndex, remainedIndex ,sampledDataFs= data_samplefs(datamat,1,48)
    # pop = fsrsnn(sampledData, remainedData, sampledIndex, remainedIndex,sampledDataFs,6)

    # 数据预处理
    # z_score_data = z_score_standardization(datamat)
    # minmax_data = minmax_normalization(datamat)
    # sampledData1, remainedData1, sampledIndex1, remainedIndex1,sampledDataFs1 = data_samplefs(z_score_data,1,48)
    # pop1 = fsrsnn(sampledData1, remainedData1, sampledIndex1, remainedIndex1,sampledDataFs1,3)
    # sampledData2, remainedData2, sampledIndex2, remainedIndex2,sampledDataFs2 = data_samplefs(minmax_data,1,48)
    # pop2 = fsrsnn(sampledData2, remainedData2, sampledIndex2, remainedIndex2,sampledDataFs2,3)
    #
    # pop.extend(pop1)
    # pop.extend(pop2)

    # initpop_ari = []  # 重新为选择过后的种群进行评估
    # for i in range(len(pop)):
    #     initpop_ari.append(adjusted_rand_score(datalabels, pop[i]))
    # print ('初始种群的ari值为：%s'%initpop_ari)
    # init_population = []
    # init_ari = []
    # for indiv1 in pop:
    #     ari1 = adjusted_rand_score(datalabels, indiv1)
    #     init_ari.append(ari1)
    #     ind1 = creator.Individual(indiv1)
    #     init_population.append(ind1)

    filter_pop = filter(lambda x:len(x)>0,init_population) ##去除初始化聚类失败的结果
    population = filter_pop #population是总的种群，后续的交叉算法的结果也要添加进来

    # ################所有种群进行聚类集成################
    # all_ensembleResult = all_ensemble(population,k=6)
    # allensemble_ari = adjusted_rand_score(datalabels,all_ensembleResult)
    # print ('全部初始种群进行聚类集成结果为:%s'%allensemble_ari)
    # #################################################





    for i in range(generation):
        cross_result = ensemble_crossover(population, init_ari)  #应用聚类集成进行交叉操作
        population.append(cross_result)
        init_ari.append(adjusted_rand_score(datalabels,cross_result))
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, tile(datamat,(len(invalid_ind),1,1)),tile(population,(len(invalid_ind),1,1)),invalid_ind)#这里只用了未经处理的数据,没有用到真实类别
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population, 40)
        init_ari = []  #重新为选择过后的种群进行评估
        for i in range(len(population)):
            init_ari.append(adjusted_rand_score(datalabels,population[i]))
    init_nmi = []
    for i in range(len(population)):
        init_nmi.append(normalized_mutual_info_score(datalabels,population[i]))
    print ('最后种群结果为：%s'%population)
    # correct_rand = []
    # for correct in range(len(population)):
    #     correct_rand.append(corrected_rand(population[correct],datalabels))
    max_ari = -inf
    for i in range(len(population)):
        ari_one = adjusted_rand_score(datalabels, population[i])
        if ari_one>max_ari:
            max_ari=ari_one
    print ('最后种群对应的ARI值为：%s'%init_ari)
    print ('最后种群对应的最大ARI值为：%s'%max_ari)
    # print ('结果corrected rand值为：%s'%correct_rand)
    print ('最后种群对应的NMI值为：%s'%init_nmi)


if __name__ == "__main__":
    main()