#coding:utf-8

from dsce import *
from numpy import *

def main():
    datamat = range(10)
    a = [[2,2,3,3,3,1,1,1,1,2],[3,3,1,1,1,3,2,2,3,2],[1,1,1,3,3,2,2,2,2,1]]
    transMatrix, popClusterArr_3, popClusterArr_2, clusterNumArr = transformation(datamat,a)
    similiarMatrix, unionClusterArr_2 = measureSimilarity(transMatrix,popClusterArr_3,popClusterArr_2,clusterNumArr,datamat,0.8)
    print similiarMatrix
if __name__ == "__main__":
        main()