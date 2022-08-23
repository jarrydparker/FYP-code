import numpy 

x = numpy.array([[1,1], [2,2], [3,3], [4,4]])
y = numpy.array([[4,4], [3,3], [2,2], [1,1]])

C = numpy.corrcoef(x)
print(C)

