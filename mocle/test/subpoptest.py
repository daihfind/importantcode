#coding:utf-8

import random as rd
from numpy import *

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
def main():
    pop = [[11],[22],[33],[44],[55],[66],[77],[88],[99],[100]]
    subpopArr = getSubPop(pop)
    print subpopArr


if __name__ == "__main__":
        main()