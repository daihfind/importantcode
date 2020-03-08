#coding:utf-8
from sklearn.cluster.k_means_ import KMeans
from numpy import *
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score

def loadDataset(filename):
    fr = open(filename)
    arraylines = fr.readlines()
    numOfLines = len(arraylines)
    returnMat = zeros((numOfLines, 4))
    classlabelVector = []
    index = 0
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(',')  # 用split会返回一个list
        returnMat[index, :] = linestrlist[0:4]
        classlabelVector.append(linestrlist[4])
        index += 1
    return returnMat, classlabelVector



def main():
    predicted_labelAll = []
    datamat,datalabels = loadDataset("../dataset/iris.data")
    print 'data ready'
    nmi_max = -inf
    ari_max = -inf
    for i in range(10):
        clusters = random.randint(2, 11)
        predicted_label = KMeans(n_clusters=clusters).fit_predict(datamat)
        predicted_label = predicted_label.tolist()
        nmi = normalized_mutual_info_score(datalabels, predicted_label)
        ari = adjusted_rand_score(datalabels, predicted_label)
        if nmi > nmi_max:
            nmi_max = nmi
        if ari > ari_max:
            ari_max = ari
    print ('nmi值为：')
    print (nmi_max)
    print ('ari值为：')
    print (ari_max)

if __name__ == "__main__":
    main()