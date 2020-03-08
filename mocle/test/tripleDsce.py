#coding:utf-8

from numpy import *
import math
from sklearn import preprocessing
#计算属于同一个簇的序列值
#参数：初始种群的一个个体，即一个聚类结果
#返回:每个簇的数据点的序号列表，以及聚类结果的簇数量
def clusterIndex(individual):
    seqarr = []
    indSet = set(individual)
    clusterNum = len(indSet)
    for value in indSet:
        seq = []
        for i in range(len(individual)):
            if individual[i] == value:
                seq.append(i)
        seqarr.append(seq)
    return seqarr,clusterNum

#dsce的第一阶段工作
#输入:原始数据集和初始种群
#返回:转换后的相似矩阵，３维聚类簇集合(分不同种群个体)，２维聚类簇集合，每个聚类结果的簇数量
def transformation(datamat,initialPop):
    popClusterArr_3 = [] #３维数组
    sumClusterNum = 0
    popClusterArr_2 = [] #2维数组
    clusterNumArr = []
    count = 0 #计数用
    for ind in initialPop:
        clusterArr,clusterNum = clusterIndex(ind)
        if clusterNum != 1:
            popClusterArr_3.append(clusterArr)#３维
            popClusterArr_2.extend(clusterArr)#2维
            sumClusterNum += clusterNum
            clusterNumArr.append(clusterNum)
        else:
            count = count+1
    if count>0:
        print '此次只有一个簇的数量为%s'%count
    transMatrix = zeros((len(datamat),sumClusterNum)) #初始化θ矩阵
    for i in range(len(datamat)):
        for j in range(sumClusterNum):
            if i in popClusterArr_2[j]:
                transMatrix[i][j] = 1
            else:
                transMatrix[i][j] = 0
    return  transMatrix,popClusterArr_3,popClusterArr_2,clusterNumArr

#dsce的第二阶段工作
#返回:新的相似矩阵θ1，以及对应合并后的候选矩阵
def measureSimilarity(transMatrix,popClusterArr_3,popClusterArr_2,clusterNumArr,datamat,a1):
    rMatrix = getRMatrix(popClusterArr_3,popClusterArr_2,clusterNumArr,datamat)
    dictSimi,similarNum = getSimiliarClu(rMatrix, a1)
    similarMatrix = zeros((len(datamat),similarNum))
    unionClusterArr_2 = []#相似矩阵合并,二位数组
    j = 0  # 用于记录新矩阵的列，用于下面的迭代
    for key in dictSimi:
        unionArr = []
        unionArr.extend(popClusterArr_2[key])
        clusterSimiIndexArr = dictSimi[key]
        #合并相似的矩阵，去除重复的元素
        for index in clusterSimiIndexArr:
            unionArr = list(set(unionArr).union(set(popClusterArr_2[index])))
        unionClusterArr_2.append(unionArr)
        #计算新矩阵的值
        for i in range(len(datamat)):
            similarMatrix[i][j] = transMatrix[i][key]
            for clusterindex in clusterSimiIndexArr:
                similarMatrix[i][j] += transMatrix[i][clusterindex]
        j += 1
    similarMatrix1,unionClusterArr_21=moreUnion(unionClusterArr_2, a1, datamat, similarMatrix)
    return similarMatrix1,unionClusterArr_21,datamat


# 初次合并后的簇再计算合并后的簇的相似度，大于a１的继续进行合并
def moreUnion(unionClusterArr_2, a1,datamat,oldSimiliarMatrix):
    flag = 0 #这是个标志位，初始化为０,如果没有变化说明没有新的需要合并的簇，变为１说明有需要合并的新簇
    dictSimi = {}
    similarNum = 0
    rMatrix = zeros((len(unionClusterArr_2), len(unionClusterArr_2)))  # 初始化矩阵
    for i in range(len(unionClusterArr_2)):
        arr = []
        for j in range(len(unionClusterArr_2)):
            if i != j:
                rMatrix[i][j] = computeR(unionClusterArr_2[i],unionClusterArr_2[j],datamat)#新的合并后的簇的相似度计算
                if rMatrix[i][j] > a1 and i<j:
                    arr.append(j)
        if arr != []:
            dictSimi[i] = arr
            similarNum += 1
            flag = 1

    if flag == 0:
        return oldSimiliarMatrix,unionClusterArr_2
    if flag == 1:
        sortedUnionIndexArr = []
        sortedUnunionIndexArr = []
        simiindexarr = [] #记录相似簇的索引,总的
        newunionClusterArr_2 = []  # 相似矩阵合并,二位数组
        columnNum = similarNum #最后的相似矩阵的列数，初始化
        for key in dictSimi:
            unionArr = []
            unionIndexArr = []
            unionArr.extend(unionClusterArr_2[key])
            clusterSimiIndexArr = dictSimi[key]
            simiindexarr.append(key)
            # 合并相似的矩阵，去除重复的元素
            unionIndexArr.append(key)
            for index in clusterSimiIndexArr:
                unionArr = list(set(unionArr).union(set(unionClusterArr_2[index])))
                simiindexarr.append(index)
                unionIndexArr.append(index)
            newunionClusterArr_2.append(unionArr)
            sortedUnionIndexArr.append(unionIndexArr)
        for oriIndex in range(len(unionClusterArr_2)):
            if oriIndex not in simiindexarr:
                newunionClusterArr_2.append(unionClusterArr_2[oriIndex])
                sortedUnunionIndexArr.append(oriIndex)
                columnNum += 1
        similiarMatrix = zeros((len(datamat), columnNum)) #相似矩阵初始化大小
        for i in range(len(datamat)):
            j = 0
            for unionindex in sortedUnionIndexArr:
                for index1 in unionindex:
                    similiarMatrix[i][j] += oldSimiliarMatrix[i][index1]
                j += 1
            for ununionindex in sortedUnunionIndexArr:
                similiarMatrix[i][j] += oldSimiliarMatrix[i][ununionindex]
                j += 1
        return moreUnion(newunionClusterArr_2,a1,datamat,similiarMatrix)
#dsce的第三阶段工作
def assign(similiarMatrix,a2,datamat):
    X_normalized = nomalise(similiarMatrix)#正则化后的相似矩阵
    dictPtoC = {} #数据点分配到对应簇的字典，key是数据点序列值,value是所属簇索引值
    dictCownP = {} #簇中拥有的数据点,key是簇索引值,value是数据点集合
    candidateClusterArr = [] #候选簇集合
    certainAssignP = [] #先分配的数据点,超过tc,c的数据点
    for i in range(len(X_normalized)): #数据索引
        tmpmaxRc = -inf
        index = -1
        for clusterindex in range(len(X_normalized[i])):#簇类索引
            if X_normalized[i][clusterindex] >= a2:
                certainAssignP.append(i)
                if X_normalized[i][clusterindex] > tmpmaxRc:
                    tmpmaxRc = X_normalized[i][clusterindex]
                    index = clusterindex
        if index != -1:
            dictPtoC[i] = index #将数据点分配到最ＲＣ最大的簇中去，key是数据点序列值,value是所属簇序列值
    certainAssignP = list(set(certainAssignP))
    for key in dictPtoC:
        cluster = dictPtoC[key]
        dictCownP[cluster] = []
    for key in dictPtoC:
        cluster = dictPtoC[key]
        dictCownP[cluster].append(key)
        #候选簇集合
        candidateClusterArr.append(cluster)
        candidateClusterArr = list(set(candidateClusterArr))
    dataIndex = range(len(similiarMatrix))  # len(similiarMatrix)等同于数据集的长度
    uncertainP = list(set(dataIndex).difference(set(certainAssignP)))
    #根据nearest　neighbor approach来分配数据点到簇
    for uncerp in uncertainP:
        minDist = inf;
        minindex = -1
        for cerp in certainAssignP:
            distJI = distEclud(datamat[uncerp], datamat[cerp])
            if distJI < minDist:
                minDist = distJI
                minindex = cerp
        cancluster = dictPtoC[cerp] #得到不确定数据点根据近邻原则应分配到的簇值
        dictCownP[cancluster].append(uncerp) #对应簇增加数据点序号
    if len(dictCownP) == 1:
        print '重新分配一次,再来再来!'
        a2 = a2 - 0.1
        if a2<0.1:
            print '不行了不行了不行列不行了不行了不行了不行列不行了不行了不行了不行列不行了'
        return assign(similiarMatrix,a2,datamat)
    return dictCownP
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB) 计算欧几里得距离


#rＭatrix的计算
def getRMatrix(popClusterArr_3,popClusterArr_2,clusterNumArr,datamat):
    rMatrix = zeros((len(popClusterArr_2),len(popClusterArr_2)))#初始化矩阵
    for i in range(len(popClusterArr_3)):
        for j in range(len(popClusterArr_3)):
            if i!= j:
                for index1 in range(len(popClusterArr_3[i])):
                    for index2 in range(len(popClusterArr_3[j])):
                        coordinate_x = getCoordinate(clusterNumArr,index1,i)
                        coordinate_y = getCoordinate(clusterNumArr,index2,j)
                        rMatrix[coordinate_x][coordinate_y] = computeR(popClusterArr_3[i][index1],popClusterArr_3[j][index2],datamat)
    return rMatrix

#计算rMatrix中的坐标值
def getCoordinate(clusterNumArr,index,preClusterNum):
    coordinate = 0
    for i in range(preClusterNum):
        coordinate += clusterNumArr[i]
    coordinate += index
    return coordinate

#‘set correlation’ (R)的计算
def computeR(cluster1,cluster2,datamat):
    CM = computeCM(cluster1,cluster2)
    if len(datamat)*CM-math.sqrt(len(cluster1)*len(cluster2))!=0 and (len(datamat)-len(cluster1))!= 0 and (len(datamat)-len(cluster2)) != 0:
        R = float(len(datamat)*CM-math.sqrt(len(cluster1)*len(cluster2)))/math.sqrt((len(datamat)-len(cluster1))*(len(datamat)-len(cluster2)))
    else:
        R = 0
    return R
#余弦相似度另一种形式,Ochiai coefficientjisuan
def computeCM(cluster1,cluster2):
    # a = list((set(cluster1).union(set(cluster2))) ^ (set(cluster1) ^ set(cluster2)))
    ret_list = list(set(cluster1).intersection(set(cluster2)))
    deno = math.sqrt(len(cluster1)*len(cluster2))
    CM = float(len(ret_list))/deno
    return CM

#根据α1判断相似度
#输入:rMatrix，阈值α1
#返回:dict嵌套list的结果,达到阈值的行数
def getSimiliarClu(rMatrix,a1):
    dict1 = {}
    similiarNum = 0
    #行列长度相等
    for i in range(len(rMatrix)): #行
        arr = []
        for j in range(len(rMatrix)): #列
            if i<j and rMatrix[i,j] >= a1:
                arr.append(j)
        if arr!=[]:
            dict1[i]=arr
            similiarNum += 1
    return dict1,similiarNum


#第三阶段工作计算original quality
#输入：dictCownP,similiarMatrix
#返回:qualty
def getCQ(dictCownP,similiarMatrix):
    dictCQ = {} #key是簇类索引号,value是original quality
    for key in dictCownP:
        sumQ = 0
        for element in dictCownP[key]:
            sumQ += similiarMatrix[element][key]
        CQ = float(sumQ)/len(dictCownP[key])
        dictCQ[key] = CQ
    return dictCQ

#规范化相似矩阵
def nomalise(similiarityMatrix):
    maxValue = -inf
    minValue = inf
    for i in range(len(similiarityMatrix)):
        for j in range(len(similiarityMatrix[0])):
            if similiarityMatrix[i][j] > maxValue:
                maxValue = similiarityMatrix[i][j]
            if similiarityMatrix[i][j] < minValue:
                minValue = similiarityMatrix[i][j]
    interval = abs(maxValue-minValue)
    for i in range(len(similiarityMatrix)):
        for j in range(len(similiarityMatrix[0])):
            similiarityMatrix[i][j] = float(similiarityMatrix[i][j])/interval
    return similiarityMatrix

#将dict的算法dsce结果转换为直接编码的个体
def resultTransform(dictResult,datamat):
    resultList = range(len(datamat))
    for key in dictResult:
        for j in dictResult[key]:
            resultList[j] = key
    return resultList

def aa():
    datamat = [1.1,2.1,3.1,4.1,5.1,6.1]
    a_2 = [[1,1,2,2,1,3],[1,2,1,2,3,4]]
    transMatrix,popClusterArr_3,popClusterArr_2,clusterNumArr= transformation(datamat,a_2)
    similiarMatrix, unionClusterArr_2 = measureSimilarity(transMatrix, popClusterArr_3, popClusterArr_2, clusterNumArr, datamat, a1=0.5)
    # print similiarMatrix
    # print unionClusterArr_2

    dictCownP = assign(similiarMatrix,0.5)
    # print dictCownP
    X = [[1., 1., 4.],
         [2., 0., 0.],
         [0., 1., 1.]]
    min_max_scaler = preprocessing.MinMaxScaler()
    normalize_data = min_max_scaler.fit_transform(X)
    print normalize_data
def main():
    aa()
if __name__ == "__main__":
    main()