#coding:utf-8
from slic_segment.slic import SLICProcessor
#
pro = SLICProcessor('../slic_segment/1000615.jpg', 1000, 10)
superPixelMatrix, clusters = pro.getDataForDSMOCMulti()
doc = open('../mockimagedata/mockdata1000615.txt','w')
doc2 = open('../mockimagedata/clusters1000615.txt','w')
for m in range(len(superPixelMatrix)):
    for n in range(len(superPixelMatrix[0])):
        # if (n != 0):
        #     doc.write(',')
        doc.write(str(superPixelMatrix[m][n]))
        doc.write(' ')
    doc.write('1')
    doc.write("\n")
doc.close()
for i in range(len(clusters)):
    for element in (clusters[i].pixels):
        doc2.write(str(element))
        doc2.write('s')
    doc2.write('\n')
doc2.close()
print "mock数据写入完成"