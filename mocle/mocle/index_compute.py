#coding:utf-8
from numpy import *
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import  cosine_distances,paired_cosine_distances
from scipy import stats
# import array
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
from scipy.spatial.distance import pdist, euclidean
#根据数据集和聚类结果计算每个簇的簇心
def getCentroids(data,labels):
    k = len(set(labels))
    centroids = zeros((k,data.shape[1]))
    #计算簇心
    column = 0 #簇心列表的行数
    for cent in set(labels):
        match_index = []
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i)
        dataInClust = data[match_index]
        centroids[column,:] = mean(dataInClust, axis=0)
        column += 1
    return centroids

#计算两个向量间的欧式距离
def Euclidean_dist(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#计算整个数据集的数据点中心
def getDataCenter(data):
    dataCenter = zeros(2)
    dataCenter = mean(data,axis=0)
    return dataCenter

#ABGSS的计算
def abgss(data,labels):
    dataCenter = getDataCenter(data)
    sumDis = 0 #ABGSS的分子
    for cent in (set(labels)):
        match_index = []
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i) #针对某个簇心，数据集中对应的数据索引值
        dataset_cent = data[match_index]
        dist = 0
        for j in range(len(dataset_cent)):
            dist += Euclidean_dist(dataset_cent[j],dataCenter)
        sumDis += dist*len(dataset_cent)
    result = sumDis/len(labels)
    return result



def computeE1(data):
    dataCenter = getDataCenter(data)
    E1 = 0
    for i in range(len(data)):
        E1 += Euclidean_dist(data[i],dataCenter)
    return E1

#计算数据集与所属聚类结果簇心的欧式距离总和,
#该版本需要外界传入簇心,如果数据运算过程和外界的簇心顺序不匹配会出错
def sum_Euc_dist(data,labels,centroids):
    distanceToCent = []
    column = 0
    for cent in (set(labels)):
        match_index = []
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i) #针对某个簇心，数据集中对应的数据索引值
        dataset_cent = data[match_index]
        dist = 0
        for j in range(len(dataset_cent)):
            dist += Euclidean_dist(dataset_cent[j],centroids[column])
        distanceToCent.append(dist) #每个簇心对应的簇内距离和
        column += 1
    distance_all = sum(distanceToCent)#整个集群的距离和
    return distance_all


#计算数据集与所属聚类结果簇心的欧式距离总和
def sum_Euc_dist2(data,labels):
    k = len(set(labels))#簇的数量
    centroids = zeros((k,data.shape[1])) #簇心
    distanceToCent = [] #每个簇内数据点到簇心的距离之和
    column = 0
    for cent in set(labels):
        match_index = []#同属于一个簇的数据点的序列号集合
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i) #针对某个簇心，数据集中对应的数据索引值
        dataInClust = data[match_index] #得到序号对应的数据
        centroids[column, :] = mean(dataInClust, axis=0) #得到对应簇的簇心
        dist = 0
        for j in range(len(dataInClust)):
            dist += Euclidean_dist(dataInClust[j],centroids[column])
        distanceToCent.append(dist) #每个簇心对应的簇内距离和
        column += 1
    distance_all = sum(distanceToCent)  # 整个集群的距离和
    return distance_all
#计算数据集与所属聚类结果簇心的欧式距离总和
def sum_Euc_distForSegmentation(data,labels):
    k = len(set(labels))#簇的数量
    centroids = zeros((k,data.shape[1])) #簇心
    distanceToCent = [] #每个簇内数据点到簇心的距离之和
    column = 0
    for cent in set(labels):
        match_index = []#同属于一个簇的数据点的序列号集合
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i) #针对某个簇心，数据集中对应的数据索引值
        dataInClust = data[match_index] #得到序号对应的数据
        centroids[column, :] = mean(dataInClust, axis=0) #得到对应簇的簇心
        dist = 0
        for j in range(len(dataInClust)):
            dist += Euclidean_dist(dataInClust[j],centroids[column])
        distanceToCent.append(dist) #每个簇心对应的簇内距离和
        column += 1
    distance_all = sum(distanceToCent)  # 整个集群的距离和
    return distance_all

#计算连通性函数值,欧氏距离版本
def connectivity_eu(distances_matrix,labels,num,**kwds):
    # distances = pairwise_distances(data, metric='euclidean', **kwds)#数据集中数据点两两之间的距离
    dist_seq = [] #对距离矩阵分别进行每一行的排序（从小到大），排序序列值的组合数组
    connect_sum = 0
    for i in range(len(distances_matrix)):
        seq = distances_matrix[i].argsort() #每一行的排序
        seq_except = ndarray.tolist(seq)
        seq_except.remove(i)
        # a = list(reversed(seq_except))
        dist_seq.append(seq_except)#要除掉数据点本身，距离为0
    for j in range(len(dist_seq)):
        for k in range(num):
            if labels[dist_seq[j][k]] != labels[j]:
                connect_sum += 1/(k+1) #k要加1是是因为数组从0开始
    return connect_sum


#计算连通性函数值，皮尔森相关系数版本
#皮尔森相关系数的值介于-1和１之间,绝对值越大表示相关度越大
def connectivity_pears(data,labels,num):
    distances = [] #两两之间的皮尔森相关系数
    dist_seq = []
    connect_sum = 0
    for i in range(len(data)):
        pearsonRow = []
        for j in range(len(data)):
            pearson = stats.pearsonr(data[i], data[j])[0]
            pearsonRow.append(abs(pearson))#这里要取绝对值
        distances.append(pearsonRow)
    distances = np.array(distances) #list转换为ndarray
    for k in range(len(distances)):
        seq = distances[k].argsort() #每一行的排序
        seq_except = ndarray.tolist(seq)
        seq_except.remove(k) #除掉自身
        reverse_seq = list(reversed(seq_except)) #反转list,相关度从大到小排列
        dist_seq.append(reverse_seq)#要除掉数据点本身
    for m in range(len(dist_seq)): #reverse_seq是相关度绝对值从大到小排列的数据点索引值
        for n in range(num):
            if labels[dist_seq[m][n]] != labels[m]:
                connect_sum += 1/(n+1) #k要加1是是因为数组从0开始
    return connect_sum


#为保持种群的多样性
def ind_similarity(pop_predicted,label,**kwds):
    pop_predicted = pop_predicted.tolist()
    nmi = 0
    popLen = len(pop_predicted)
    # label  = label.tolist()
    for i in range(popLen):
        normalized_mutual_info_score(label,pop_predicted[i])
    nmiAve = nmi/(popLen*popLen)
    return nmiAve



#计算每个簇内数据点到簇心的距离之和,返回距离之和以及隶属于每一个簇的数据点个数集合
def intra_cluster_distances(data,labels):
    k = len(set(labels))#簇的数量
    centroids = zeros((k,data.shape[1])) #簇心
    distanceToCent = [] #每个簇内数据点到簇心的距离之和
    centNumCount = [] #隶属于每一个簇的数据点个数集合
    column = 0
    for cent in set(labels):
        match_index = []#同属于一个簇的数据点的序列号集合
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i) #针对某个簇心，数据集中对应的数据索引值
        dataInClust = data[match_index] #得到序号对应的数据
        centNumCount.append(len(match_index)) #对应簇中的数据点数量
        centroids[column, :] = mean(dataInClust, axis=0) #得到对应簇的簇心
        dist = 0
        for j in range(len(dataInClust)):
            dist += Euclidean_dist(dataInClust[j],centroids[column])
        distanceToCent.append(dist) #每个簇心对应的簇内距离和
        column += 1
    return distanceToCent,centNumCount,centroids

def daviesbouldin(data,labels):
    distanceToCentAve = [] #每个簇内数据点到簇心平均距离
    distanceToCent,centNumCount,centroids = intra_cluster_distances(data,labels)
    for i in range(len(centroids)):
        if centNumCount[i] == 0:
            print '000000000000'
        distanceToCentAve.append(distanceToCent[i]/centNumCount[i])
    distances = pairwise_distances(centroids, metric='euclidean')
    dbMaxAll = 0
    for i in range(len(centroids)):
        dbMax = 0
        # for j in range(i+1,len(centroids)):
        for j in range(len(centroids)):
            if i != j:
                db = distanceToCentAve[i] + distanceToCentAve[j]/distances[i][j]
                if db > dbMax:
                    dbMax = db
        dbMaxAll += dbMax
    dbResult = dbMax/len(centroids)
    return dbResult




#簇内个体样本间距离
def intra_sample(data,labels):
    dist_arr = []
    centNumCount = []  # 隶属于每一个簇的数据点个数集合
    for cent in set(labels):
        match_index = []  # 同属于一个簇的数据点的序列号集合
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i)  # 针对某个簇，数据集中对应的数据索引值
        dataInClust = data[match_index]  # 得到序号对应的数据
        centNumCount.append(len(match_index))  # 对应簇中的数据点数量
        dist = 0
        for m in range(len(dataInClust)):
            for n in range(len(dataInClust)):
                dist += Euclidean_dist(dataInClust[m],dataInClust[n])
        dist_arr.append(dist)#得到各个簇内的样本距离之和
    compact = 0
    for i in range(len(dist_arr)):
        compact += ((dist_arr[i]/(centNumCount[i]*(centNumCount[i]-1)))/sqrt(centNumCount[i]))
    compact = compact/len(set(labels))
    return compact

def getmin_sep(centroids):
    distances = pairwise_distances(centroids, metric='euclidean')#数据集中数据点两两之间的距离
    dist_seq = [] #对距离矩阵分别进行每一行的排序（从小到大），排序序列值的组合数组
    for i in range(len(distances)):
        seq = distances[i].argsort() #每一行的排序
        seq_except = ndarray.tolist(seq)
        seq_except.remove(i)
        # a = list(reversed(seq_except))
        dist_seq.append(seq_except)#要除掉数据点本身，距离为0
    min_sep = inf
    for j in range(len(distances)):
        if distances[j][dist_seq[j][0]] < min_sep:
            min_sep = distances[j][dist_seq[j][0]]
    return min_sep

def getmax_sep(centroids):
    distances = pairwise_distances(centroids, metric='euclidean')#数据集中数据点两两之间的距离
    dist_seq = [] #对距离矩阵分别进行每一行的排序（从小到大），排序序列值的组合数组
    for i in range(len(distances)):
        seq = distances[i].argsort() #每一行的排序
        seq_except = ndarray.tolist(seq)
        seq_except.remove(i)
        # a = list(reversed(seq_except))
        dist_seq.append(seq_except)#要除掉数据点本身，距离为0
    max_sep = -inf
    length = len(centroids)
    for j in range(len(distances)):
        if distances[j][dist_seq[j][length-2]] > max_sep:
            max_sep = distances[j][dist_seq[j][length-2]]
    return max_sep

#计算分离性函数
def getSepration(centroids):
    distances = pairwise_distances(centroids, metric='euclidean')  # 数据集中数据点两两之间的距离
    sumDis = 0
    for i in range(len(distances)):
        for j in range(len(distances[0])):
            if i != j:
                sumDis += distances[i][j]
    sepResult = 2*sumDis/(len(centroids)*(len(centroids)-1))
    return sepResult

#目标函数创新１，结合的是：簇内样本间平均距离和最小簇间簇心距离之间的比值
def formula_one(data,labels):
    k = len(set(labels))#簇的数量
    centroids = zeros((k,data.shape[1])) #簇心
    column = 0
    dist_arr = []
    centNumCount = []  # 隶属于每一个簇的数据点个数集合
    for cent in set(labels):
        match_index = []  # 同属于一个簇的数据点的序列号集合
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i)  # 针对某个簇，数据集中对应的数据索引值
        dataInClust = data[match_index]  # 得到序号对应的数据
        centNumCount.append(len(match_index))  # 对应簇中的数据点数量
        centroids[column, :] = mean(dataInClust, axis=0)  # 得到对应簇的簇心
        dist = 0
        for m in range(len(dataInClust)):
            for n in range(len(dataInClust)):
                dist += Euclidean_dist(dataInClust[m],dataInClust[n])
        dist_arr.append(dist)#得到各个簇内的样本距离之和
    compact = 0
    for i in range(len(dist_arr)):
        compact += ((dist_arr[i]/(centNumCount[i]*(centNumCount[i]-1)))/sqrt(centNumCount[i]))
    compact = compact/len(set(labels))
    min_sep = getmin_sep(centroids)
    result = compact/(min_sep*len(centroids))
    return result

#目标函数创新2,结合的是：簇内样本间平均距离和ABGSS的比值
def formula_two(data,labels):
    k = len(set(labels))#簇的数量
    dist_arr = []
    centNumCount = []  # 隶属于每一个簇的数据点个数集合
    for cent in set(labels):
        match_index = []  # 同属于一个簇的数据点的序列号集合
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i)  # 针对某个簇，数据集中对应的数据索引值
        dataInClust = data[match_index]  # 得到序号对应的数据
        centNumCount.append(len(match_index))  # 对应簇中的数据点数量
        dist = 0
        for m in range(len(dataInClust)):
            for n in range(len(dataInClust)):
                dist += Euclidean_dist(dataInClust[m],dataInClust[n])
        dist_arr.append(dist)#得到各个簇内的样本距离之和
    compact = 0
    for i in range(len(dist_arr)):
        compact += ((dist_arr[i]/(centNumCount[i]*(centNumCount[i]-1)))/sqrt(centNumCount[i]))
    compact = compact/len(labels)
    abgssresult = abgss(data,labels)
    result = compact/abgssresult
    return result

#目标函数创新2,结合的是：簇内样本间平均距离和(ABGSS*sepration)的比值
def formula_three(data,labels):
    k = len(set(labels))#簇的数量
    centroids = zeros((k,data.shape[1])) #簇心
    column = 0
    dist_arr = []
    centNumCount = []  # 隶属于每一个簇的数据点个数集合
    for cent in set(labels):
        match_index = []  # 同属于一个簇的数据点的序列号集合
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i)  # 针对某个簇，数据集中对应的数据索引值
        dataInClust = data[match_index]  # 得到序号对应的数据
        centNumCount.append(len(match_index))  # 对应簇中的数据点数量
        centroids[column, :] = mean(dataInClust, axis=0)  # 得到对应簇的簇心
        dist = 0
        for m in range(len(dataInClust)):
            for n in range(len(dataInClust)):
                dist += Euclidean_dist(dataInClust[m],dataInClust[n])
        dist_arr.append(dist)#得到各个簇内的样本距离之和
    compact = 0
    for i in range(len(dist_arr)):
        compact += ((dist_arr[i]/(centNumCount[i]*(centNumCount[i]-1)))/sqrt(centNumCount[i]))
    compact = compact/len(set(labels))
    abgssresult = abgss(data,labels)
    sepresult = getSepration(centroids)
    result = compact/(abgssresult*sepresult)
    return result

#目标函数创新2,结合的是：簇内样本间平均距离和ABGSS的比值
def formula_four(data,labels,eudataPointMatrix):
    dist_arr = []
    centNumCount = []  # 隶属于每一个簇的数据点个数集合
    for cent in set(labels):
        match_index = []  # 同属于一个簇的数据点的序列号集合
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i)  # 针对某个簇，数据集中对应的数据索引值
        centNumCount.append(len(match_index))  # 对应簇中的数据点数量
        dataInClust = data[match_index]  # 得到序号对应的数据

        dist = 0
        for m in range(len(match_index)):
            for n in range(m+1,len(match_index)):
                dist += eudataPointMatrix[match_index[m]][match_index[n]]
        dist = dist*2
        dist_arr.append(dist)#得到各个簇内的样本距离之和
    compact = 0
    for i in range(len(dist_arr)):
        # if centNumCount[i] == 1:
        #     # print "有簇只有一个个体"
        #     # print "这个簇个体距离之和为%s"%dist_arr[i]
        #     continue
        # else:
        compact += ((dist_arr[i]/(centNumCount[i]*centNumCount[i]))) #被除数可能为0
    compact = compact/len(set(labels))
    abgssresult = abgss(data,labels)
    result = compact/abgssresult
    return result

def formula_five(data,labels):
    k = len(set(labels))#簇的数量
    centroids = zeros((k,data.shape[1])) #簇心
    column = 0
    dist_arr = []
    centNumCount = []  # 隶属于每一个簇的数据点个数集合
    for cent in set(labels):
        match_index = []  # 同属于一个簇的数据点的序列号集合
        for i in range(len(labels)):
            if labels[i] == cent:
                match_index.append(i)  # 针对某个簇，数据集中对应的数据索引值
        dataInClust = data[match_index]  # 得到序号对应的数据
        centNumCount.append(len(match_index))  # 对应簇中的数据点数量
        centroids[column, :] = mean(dataInClust, axis=0)  # 得到对应簇的簇心
        dist = 0
        for m in range(len(dataInClust)):
            for n in range(len(dataInClust)):
                dist += Euclidean_dist(dataInClust[m],dataInClust[n])
        dist_arr.append(dist)#得到各个簇内的样本距离之和
    compact = 0
    for i in range(len(dist_arr)):
        compact += ((dist_arr[i]/(centNumCount[i]*(centNumCount[i]-1)))/(centNumCount[i]))
    compact = compact/len(set(labels))
    abgssresult = abgss(data,labels)
    result = compact/abgssresult
    return result
#汇总评价函数
def mocle_index(dataset,distances_matrix,label,num=3): #num　是相关系数的个数取值
    # 欧氏距离和，紧凑度
    euDistance = sum_Euc_dist2(dataset,label)
    #连通性函数
    eu_connect = connectivity_eu(distances_matrix=distances_matrix,labels=label,num=num)
    # eu_sample = intra_sample(data = dataset,labels=label)
    # centroids = getCentroids(data=dataset, labels=label)
    # dbi = daviesbouldin(data = dataset, labels= label)
    # P_connect = connectivity_pears(data=dataset,labels=label,num=num)
    # cosine = ind_similarity(pop_predicted,label)
    # result = formula_one(data = dataset,labels=label)
    # centroids = getCentroids(data=dataset, labels=label)
    # sep = getSepration(centroids)
    # nor_sep = 1.0/sep
    # sepresult = 1/sep
    # result = formula_two(data=dataset, labels=label)
    # result = formula_three(data=dataset, labels=label)
    # result = formula_four(data=dataset, labels=label,eudataPointMatrix=eudataPointMatrix) #用这个做实验
    # result = formula_five(data=dataset, labels=label)
    return euDistance,eu_connect



# Takes two partitions and returns the correct Rand index of similarity (x <= 1)
# Always returns 1 on equal data
def corrected_rand(part1, part2):
    assert(len(part1) == len(part2))
    n = len(part1)
    k = max(len(set(part1)), len(set(part2)))
    matches = 0
    for i in range(n):
        for j in range(n):
            if i != j and ((part1[i] == part1[j]) == (part2[i] == part2[j])):
                matches += 1
    if k == 1:
        return 1
    return (matches - 1.0*(k*k - 2*k + 2)/(k*k)*n*(n - 1)) \
            / (2.0*(k-1)/(k*k)*n*(n - 1))


#DB index
def daviesbouldin1(X, labels, centroids):
    nbre_of_clusters = len(centroids) #Get the number of clusters
    distances = [[] for e in range(nbre_of_clusters)] #Store intra-cluster distances by cluster
    distances_means = [] #Store the mean of these distances
    DB_indexes = [] #Store Davies_Boulin index of each pair of cluster
    second_cluster_idx = [] #Store index of the second cluster of each pair
    first_cluster_idx = 0 #Set index of first cluster of each pair to 0

    # Step 1: Compute euclidean distances between each point of a cluster to their centroid
    for cluster in range(nbre_of_clusters):
        for point in range(X[labels == cluster].shape[0]):
            distances[cluster].append(euclidean(X[labels == cluster][point], centroids[cluster]))

    # Step 2: Compute the mean of these distances
    for e in distances:
        distances_means.append(np.mean(e))

    # Step 3: Compute euclidean distances between each pair of centroid
    ctrds_distance = pdist(centroids)

    # Tricky step 4: Compute Davies-Bouldin index of each pair of cluster
    for i, e in enumerate(e for start in range(1, nbre_of_clusters) for e in range(start, nbre_of_clusters)):
        second_cluster_idx.append(e)
        if second_cluster_idx[i-1] == nbre_of_clusters - 1:
            first_cluster_idx += 1
        DB_indexes.append((distances_means[first_cluster_idx] + distances_means[e]) / ctrds_distance[i])

    # Step 5: Compute the mean of all DB_indexes
    # print("DAVIES-BOULDIN Index: %.5f" % np.mean(DB_indexes))
    return np.mean(DB_indexes)

#图像分割指标ＰＢＭ
def getPBM(datamat,result):
    record = list(set(result))
    c = len(record)
    ec = sum_Euc_dist2(datamat, result)
    e1 = 10
    centroids = getCentroids(datamat, result)
    max_sep = getmax_sep(centroids)
    value = (1 / c) * (e1 / ec) * max_sep
    return result

if __name__ == "__main__":
    # label =[0,1]
    # dataset = array([[1,2,3,4,5],[5,6,7,8,7]])
    # a = np.array(label)
    # # result = getCentroids(data=dataset,labels=label)
    # # print result
    # # distances,a = sum_Euc_dist(dataset,label,result)
    # # print distances
    # # print a
    # connectivity_sum = connectivity_eu(data=dataset,labels=label,num=1)
    # print connectivity_sum
    #
    # Pconnectivity_sum = connectivity_pears(data=dataset,labels=label,num=1)
    # print Pconnectivity_sum
    print 0.0/(1-1)
    print range(2,2)
    for i in range(2):
        for j in range(1,2):
            print j
    label = [0, 0,1]
    dataset = array([[1, 2, 44, 33, 5], [5, 6, 7, 8, 7],[1,1,1,3,4]])
    result = getCentroids(data=dataset, labels=label)
    a = np.array(label)

    a = set(dataset[0])
    for i in a:
        print i

    dataset1 = array([[1, 2, 44, 33, 5], [5, 6, 7, 8, 7], [1, 1, 1, 3, 4]])
    a1 = set(dataset[0])
    for i in a1:
        print i




