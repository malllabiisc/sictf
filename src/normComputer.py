import logging
import time
import numpy as np
from numpy import dot, zeros, array, eye, kron, prod
from numpy.linalg import norm, solve, inv, svd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
import math
import scipy as sp
import pdb
import multiprocessing
import timeit
import os 
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("TensorFolder", help = "Enter the name of the Tensor Folder :", type = str)
args = parser.parse_args()
inputFolder = args.TensorFolder

# load tensor
fname = 'listOfSlices' #input('Enter name of the file that is the source of Tensor :')
# TensorArray = np.load(os.getcwd()+'/'+inputFolder+fname+'.npy')
# X = [lil_matrix(TensorArray[k, :, :]) for k in range(TensorArray.shape[0])]
fileHandle = open(os.path.join(os.getcwd(),inputFolder,fname+'.pkl'),'rb')
XList = pickle.load(fileHandle)
fileHandle.close()
# Tensor = fromarray(TensorArray)

		

def tensorNorm(tensor):
	tensorNorm = 0.0
	for tensorSlice in tensor:
		sliceNorm = sum(tensorSlice.data**2)
		tensorNorm += sliceNorm
	return tensorNorm**0.5

print('The Frob norm of the tensor : %s'%(tensorNorm(XList)))

#load side info matrix
sideInfoMatrix = 'sideInfoMatrix'
fileHandle = open(os.path.join(os.getcwd(),inputFolder,sideInfoMatrix+'.pkl'),'rb')
D = pickle.load(fileHandle)
fileHandle.close()

def matrixNorm(sideInfoMatrix):
	return np.linalg.norm(sideInfoMatrix.toarray())

print('Frob norm of side info matrix: %s'%(matrixNorm(D)))