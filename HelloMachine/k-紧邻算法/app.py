# coding=utf-8
__author__ = 'Administrator'
import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

# group,labels=kNN.createDataSet()
# print(group)
# print(labels)
# point=kNN.classify0([0,0],group,labels,3)
# print(point)

reload(kNN)
# datingDataMat,datingLables=kNN.file2matrix('datingTestSet.txt')
# print(datingDataMat)
# print(datingLables[0:20])
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLables),15.0*array(datingLables))
# # plt.show()
# normMat,ranges,minVals=kNN.autoNorm(datingDataMat)
# print(normMat)
# print(ranges)
# print (minVals)
# kNN.datingClassTest()

testVect = kNN.img2Vector('digits/testDigits/0_3.txt')
print (testVect[0, 0:31])
