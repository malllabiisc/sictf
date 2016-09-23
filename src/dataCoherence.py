import argparse
import pickle
import os
import numpy as np
from numpy import zeros, ones, array, arange, copy, ravel_multi_index, unravel_index
from numpy import setdiff1d, hstack, hsplit, vsplit, sort, prod, lexsort, unique, bincount
from numpy import dot, zeros, array, eye, kron, prod
from scipy.sparse import lil_matrix, coo_matrix
from scipy.linalg import norm as spnorm
import pdb
import logging
import timeit

cwd = os.getcwd()
# get user inputs
parser = argparse.ArgumentParser()
parser.add_argument("inputFolderName", help = "Enter the name of the folder that contains Tensor and Side-Info :", type = str)
args = parser.parse_args()
inputFolderName = args.inputFolderName

# load data 
fname = 'listOfSlices' #input('Enter name of the file that is the source of Tensor :')
# TensorArray = np.load(os.getcwd()+'/'+inputFolder+fname+'.npy')
# X = [lil_matrix(TensorArray[k, :, :]) for k in range(TensorArray.shape[0])]
fileHandle = open(os.path.join(cwd,inputFolderName,fname+'.pkl'),'rb')
XList = pickle.load(fileHandle)
fileHandle.close()
# Tensor = fromarray(TensorArray)

sideInfoMatrix = 'sideInfoMatrix'
fileHandle = open(os.path.join(os.getcwd(),inputFolderName,sideInfoMatrix+'.pkl'),'rb')
D = pickle.load(fileHandle)
fileHandle.close()


print("Type of Side info : %s"%(type(D)))

# Generating tensor based pairwise entity similarity
# dict of the form {(e1,e2):dot product}

def tensorSimilarity(tensor):
	# print("Creating tensorSimilarityDict")
	# similarityHash = {}
	X = sum(tensor) # summation over all slices
	# size = X.shape[0]
	# for rowIndex, rowVector in enumerate(X):
	# 	for nextRowIndex in range(rowIndex+1,size):
	# 		# pdb.set_trace()
	# 		# nextVector = X[nextRowIndex]
	# 		similarityHash.update({(rowIndex,nextRowIndex):float(dot(rowVector,X[nextRowIndex].T).toarray()[0][0])})
	# return similarityHash
	return dot(X,X.T)

def matrixSimilarity(matrix):
# 	print("Creating matrixSimilarityDict")
# 	similarityHash = {}
# 	size = matrix.shape[0]
# 	for rowIndex in range(size):
# 		rowVector = matrix[rowIndex]
# 		for nextRowIndex in range(rowIndex+1,size):
# 			# pdb.set_trace()
# 			# nextVector = matrix[nextRowIndex]
# 			similarityHash.update({(rowIndex,nextRowIndex):float(dot(rowVector,matrix[nextRowIndex].T).toarray()[0][0])})
# 	return similarityHash
	return dot(matrix,matrix.T)

start = timeit.default_timer()
tensorSimilarity = tensorSimilarity(XList)
end = timeit.default_timer()
print(end - start)
print("tensorSimilarityDict created")
# saving similarity matrix
fileHandle = open(os.path.join(cwd,inputFolderName,'tensorSimilarity'+'.pkl'),'wb')
pickle.dump(tensorSimilarity,fileHandle)
fileHandle.close()

start = timeit.default_timer()
matrixSimilarity = matrixSimilarity(D)
start = timeit.default_timer()
print(end - start)
print("matrixSimilarityDict created")

fileHandle = open(os.path.join(cwd,inputFolderName,'matrixSimilarity'+'.pkl'),'wb')
pickle.dump(matrixSimilarity,fileHandle)
fileHandle.close()

boolTensor = tensorSimilarity.toarray().astype(bool).astype(int)
boolMatrix = matrixSimilarity.toarray().astype(bool).astype(int)
diff = np.logical_xor(boolTensor,boolMatrix)

#printing xnor non-zero
print("the number of clashes: %s"%(len(np.flatnonzero(diff))))
