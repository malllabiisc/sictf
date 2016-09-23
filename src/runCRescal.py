#!/usr/bin/env python
import os
import re
import pickle
import datetime
import timeit
import logging
from scipy.io.matlab import loadmat
from random import *
# from sktensor import dtensor, cp_als, sptensor
# from sktensor.sptensor import fromarray
import numpy as np
from numpy import zeros, ones, array, arange, copy, ravel_multi_index, unravel_index
from numpy import setdiff1d, hstack, hsplit, vsplit, sort, prod, lexsort, unique, bincount
from scipy.sparse import lil_matrix, coo_matrix
from scipy.linalg import norm as spnorm
# from rescal import rescal_als
# from sktensor.rescal import als as rescal_als
# from rescal import als # From Standalone rescal
from Cnnrescal import nnr_als # From nnrescal
import argparse
from joblib import Parallel, delayed  
import multiprocessing
import pdb

num_cores_for_fit_computation = 4 #multiprocessing.cpu_count() # Dont take more than 4 cores if not using cheap fit
num_cores_for_rescal = multiprocessing.cpu_count()
parser = argparse.ArgumentParser()

Root = os.getcwd()
# Set logging to DEBUG to see CP-ALS information
logging.basicConfig(level=logging.DEBUG)

# Load Matlab data and convert it to dense tensor format

inputFolder = 'toRescal'
# outputFolderName = input('Enter the name of the Output Folder :')

fileData = 'filesFromData'

parser.add_argument("outputFolderName", help = "Enter the name of the Output Folder :", type = str)

# minRank = int(input('Enter Rank for min Rescal Decomposition :'))
parser.add_argument("minRank", help = "Enter Rank for min Rescal Decomposition :", type = int)
# maxRank = int(input('Enter Rank for max Rescal Decomposition :'))
parser.add_argument("maxRank", help = "Enter Rank for max Rescal Decomposition :", type = int)
# step = int(input('Enter step for Rank (Please ensure minRank +n*step = maxRank):'))
parser.add_argument("step", help = "Enter step for Rank (Please ensure minRank +n*step = maxRank):", type = int)
# Top = int(input('Enter cut-off for top Entities: '))
parser.add_argument("Top", help = "Enter cut-off for top Entities: ", type = int)
parser.add_argument("TopRC", help = "Enter cut-off for top RelMatrix Entries: ", type = int)
# maxIters = int(input('Enter Max no. of iterations: '))
parser.add_argument("maxIters", help = "Enter Max no. of iterations: ", type = int)
# fitFlag = eval(input('Enter True is fit computation desired, False if not, None for uncertainity: ')) # Make sure giving False works, otherwise give default and experimentally determine a good #iter
parser.add_argument("fitFlag", help = "Enter True is fit computation desired, False if not, None for uncertainity", type = str)

parser.add_argument("lA", help = "Enter Lambda A", type = float)
parser.add_argument("lR", help = "Enter Lambda R", type = float)
parser.add_argument("lV", help = "Enter Lambda V", type = float)
parser.add_argument("Ws", help = "Enter Side Info term weight", type = float)
parser.add_argument("Wrs", help = "Enter Relation Similarity term weight", type = float)

args = parser.parse_args()


outputFolderName = args.outputFolderName
minRank = args.minRank
maxRank = args.maxRank
step = args.step
Top = args.Top
TopRC = args.TopRC
maxIters = args.maxIters
fitFlag = eval(args.fitFlag)
lA = args.lA
lR = args.lR
lV = args.lV
Ws = args.Ws
print(Ws)
Wrs = args.Wrs
print(Wrs)
inputHandle = open(os.path.join(os.getcwd(),fileData,'domainNounPhrases.pkl'),'rb')
dictNPtoIndex = pickle.load(inputHandle)
dictIndextoNP = pickle.load(inputHandle)
inputHandle.close()


inputHandle = open(os.path.join(os.getcwd(),fileData,'verbPhrase.pkl'),'rb')
dictVPtoIndex = pickle.load(inputHandle)
dictIndextoVP = pickle.load(inputHandle)
inputHandle.close()



if outputFolderName not in os.listdir(Root):
	os.mkdir(outputFolderName)

fname = 'listOfSlices' #input('Enter name of the file that is the source of Tensor :')
# TensorArray = np.load(os.getcwd()+'/'+inputFolder+fname+'.npy')
# X = [lil_matrix(TensorArray[k, :, :]) for k in range(TensorArray.shape[0])]
fileHandle = open(os.path.join(os.getcwd(),inputFolder,fname+'.pkl'),'rb')
XList = pickle.load(fileHandle)
fileHandle.close()
# Tensor = fromarray(TensorArray)

sideInfoMatrix = 'sideInfoMatrix'
fileHandle = open(os.path.join(os.getcwd(),inputFolder,sideInfoMatrix+'.pkl'),'rb')
D = pickle.load(fileHandle)
fileHandle.close()

# D = np.load(os.path.join(os.getcwd(),inputFolder,sideInfoMatrix+'.npy'))

relSimMatrix = 'relSimMatrix'
# fileHandle = open(os.path.join(os.getcwd(),inputFolder,relSimMatrix+'.pkl'),'rb')
# RS = pickle.load(fileHandle)
# fileHandle.close()

RS = np.load(os.path.join(os.getcwd(),inputFolder,relSimMatrix+'.npy'))


os.chdir(outputFolderName) # ***** Dir Change *****
subRoot = os.getcwd()

print(lA,lR,lV,Ws,Wrs)
lambdaFolderName = 'lA-'+str(lA)+'_'+'lR-'+str(lR)+'_'+'lV-'+str(lV)+'_'+'Ws-'+str(Ws)+'_'+'Wrs-'+str(Wrs)
if lambdaFolderName not in os.listdir(subRoot):
	os.mkdir(lambdaFolderName)

# pdb.set_trace()
### Functions for Understanding
### 1) Print Top k phrases in a latent factor of A matrix
### 2) Generating Relation Matrices file with Top m participating matrix indices, 
###		VP = <VP> Fit: <Num> Same format as before
###		Top m Indices in rev sorted order and empty line
###		Computing a faster fit ( or memory efficient)
###		Doing fit computations in parallel (if memory efficient is possible)
### 3) Generating Relation Report in Rev sorted format: with parallel computations of fits.
### 	VP = <VP> VPI = <VPI> Slice Norm = <Val> Error Norm = <Val> SliceFit = <Val>
###		Generate total fit of Rescal 
###
###
###
### pdb.set_trace()


def TopPhrasesForAllLatentComponent(matrixA,dictNPtoIndex,dictIndextoNP,TopK,lambdaFolderName,runDir):
	sA_V, sA_I = np.flipud(np.sort(matrixA,axis = 0,kind = 'mergesort')),np.flipud(np.argsort(matrixA,axis = 0,kind = 'mergesort'))
	NPCount, LCCount = matrixA.shape
	# print(NPCount,LCCount) # Tally with tensor Creator

	fileHandle = open(os.path.join(os.getcwd(),lambdaFolderName,runDir,'Top'+str(TopK)+'EntitesForALatentComponent.txt'),'w')
	for LCIndex in range(0,LCCount):
		fileHandle.write('Latent Factor : '+str(LCIndex))
		for rowNum in range(0,TopK):
			fileHandle.write('\t'+str(dictIndextoNP[sA_I[rowNum,LCIndex]])+'\t'+str(sA_V[rowNum,LCIndex]))
		fileHandle.write('\n')
	fileHandle.close()




def RelationMatrix(Tensor,RelationTensor,matrixA,sideInfo,sideInfoBasis,dictIndextoVP,dictVPtoIndex,RelationReportData,TopRC,lambdaFolderName,runDir): # Fit thresholding can be done in later stages
	numVP = len(RelationReportData) # verify
	# if numVP == len(RelationTensor):
	# 	print('Number of Relations agree')
	# else:
	# 	print('Check Number of Relations and other dimensions')
	# generating Fit.txt
	def generateFitFile(RelationReportData):
		# RRD = [(VP,#VP,||Xk||,||Xk-ARkAt||,1-||Xk-ARkAt||/||Xk||), and so on]
		# pdb.set_trace()
		squaredErrorSliceNormList = [info[3]**2 for info in RelationReportData ]
		# pdb.set_trace()
		totalResidueNorm = sum(squaredErrorSliceNormList)**0.5
		# pdb.set_trace()
		squaredTensorNorm = [info[2]**2 for info in RelationReportData ]
		# pdb.set_trace()
		tensorNorm = sum(squaredTensorNorm)**0.5
		# pdb.set_trace()
		Fit = 1 - totalResidueNorm/tensorNorm
		print('\n****\n')
		print('Fit = ',Fit)
		print('\n****\n')
		return Fit
	def generateSideInfoFit(sideInfo,matrixA,sideInfoBasis):
		residueMatrix = sideInfo.toarray() - matrixA.dot(sideInfoBasis)
		residueMatrixNorm = np.linalg.norm(residueMatrix)
		FitSide = 1 - residueMatrixNorm/np.linalg.norm(sideInfo.toarray())
		print('\n****\n')
		print('Fit Side Info = ',FitSide)
		print('\n****\n')
		return FitSide

	Fit = generateFitFile(RelationReportData)
	SideInfoFit = generateSideInfoFit(sideInfo,matrixA,sideInfoBasis)
	FitHandle = open(os.path.join(os.getcwd(),lambdaFolderName,runDir,'Fit.txt'),'w')
	FitHandle.write(str('Fit over Tensor: '+str(Fit)+'\n'+'Side Info Fit :'+str(SideInfoFit)+'\n'))

	def flattenMatrix(matrix):
		''' Returns a list of tuples of the form [(row index,col index, value at the index),] '''
		flattenedMatrix = [] # containing the matrix info
		numRow,numCol = matrix.shape
		for currentRowIndex in range(0,numRow):
			for currentColIndex in range(0,numCol):
				flattenedMatrix.append((currentRowIndex,currentColIndex,matrix[currentRowIndex][currentColIndex]))
		return flattenedMatrix

	def generatePrintDataForSlice(flattenedMatrix,TopRC):
		flattenedMatrix.sort(key = lambda x:x[2], reverse = True)
		# print(flattenedMatrix)
		text = 'Values at Indicies of the Relation Matrix in descending order (Top '+str(TopRC)+' ).\n'
		index = 0
		while(index<TopRC):
			text += 'Value at ( '+str(flattenedMatrix[index][0])+', '+str(flattenedMatrix[index][1])+') = '+str(flattenedMatrix[index][2])+'\n'
			index += 1
		return text

	fileHandle = open(os.path.join(os.getcwd(),lambdaFolderName,runDir,'RelationMatrices.txt'),'w')
	for info in RelationReportData:
		Line1 = 'Verb Phrase: '+str(info[0])+'\t'+'Fit: '+str(info[4])+'\n'
		fileHandle.write(Line1)
		Line2 = generatePrintDataForSlice(flattenMatrix(RelationTensor[info[1]]),TopRC)
		fileHandle.write(Line2+'\n')
	fileHandle.close()


def SummaryGen(A,R,RelationReportData,TopK,TopRC,TopI = 3,CutOffForFit = 1):
	def flattenSortedMatrix(matrix):
		''' Returns a list of tuples of the form [(row index,col index, value at the index),] '''
		flattenedMatrix = [] # containing the matrix info
		numRow,numCol = matrix.shape
		for currentRowIndex in range(0,numRow):
			for currentColIndex in range(0,numCol):
				flattenedMatrix.append((currentRowIndex,currentColIndex,matrix[currentRowIndex][currentColIndex]))
		flattenedMatrix.sort(key = lambda x:x[2], reverse = True)
		return flattenedMatrix

	def TopPhrasesForALatentComponent(matrixA,dictIndextoNP,LC,Top = 4):
		sA_V, sA_I = np.flipud(np.sort(matrixA,axis = 0,kind = 'mergesort')),np.flipud(np.argsort(matrixA,axis = 0,kind = 'mergesort'))
		NPCount, LCCount = matrixA.shape
		# print(NPCount,LCCount) # Tally with tensor Creator

		Line1 = 'Latent Factor : '+str(LC)
		Line2 = ''
		for rowNum in range(0,Top):
			Line2 += '\t'+str(dictIndextoNP[sA_I[rowNum,LC]])+'\t'+str(sA_V[rowNum,LC])
		Line2 += '\n'
		Line = Line1 + Line2
		return Line

	cutOff = CutOffForFit/100
	fileHandle = open(os.path.join(os.getcwd(),lambdaFolderName,runDir,'Summary.txt'),'w')
	# print(RelationReportData)
	for info in RelationReportData:
		# print(info)
		# if info[4] > cutOff: #For all fits generate summary
		Line1 = 'Verb Phrase: '+str(info[0])+'\t'+'Fit: '+str(info[4])+'\n'
		fileHandle.write(Line1)
		flatRSlice = flattenSortedMatrix(R[info[1]])
		candidateIndicesForSummary = flatRSlice[:TopI]
		# print(candidateIndicesForSummary)
		for infoTuples in candidateIndicesForSummary:
			# print(infoTuples)
			fileHandle.write('Value at ( '+str(infoTuples[0])+', '+str(infoTuples[1])+') = '+str(infoTuples[2])+'\n')
			fileHandle.write(TopPhrasesForALatentComponent(A,dictIndextoNP,infoTuples[0]))
			fileHandle.write(TopPhrasesForALatentComponent(A,dictIndextoNP,infoTuples[1])+'\n')
			# pdb.set_trace()
		# else:
		# 	break
	fileHandle.close()



def computeSliceFit(TensorSlice,RelationSlice,matrixA,dictIndextoVP,relNum):
		# Fit = 1 - ||Ek||/||Xk||
		# Ek = Xk - ARkA.T
		# X_k = ArkA.T
		Rk = RelationSlice
		A = matrixA
		At = A.T
		tensorSliceNorm = sum(TensorSlice.data**2)**0.5
		X_k = A.dot(Rk.dot(At)) # this is a dense matrix, takes up ram
		normRecon = np.linalg.norm(X_k) # This takes up ram
		verbPhrase = dictIndextoVP[relNum]
			
		Ek = TensorSlice - X_k # RAM Intensive
		normResidueSlice = np.linalg.norm(Ek)
		Fit = 1 - normResidueSlice/tensorSliceNorm
		return verbPhrase, relNum, tensorSliceNorm, normResidueSlice, Fit

def cheaplyComputeSliceFit(TensorSlice,RelationSlice,matrixA,dictIndextoVP,dictIndextoNP,relNum):
	Rk = RelationSlice
	tensorSliceNorm = sum(TensorSlice.data**2)**0.5
	# Either implement index extraction in csr format or convert tensor slice to COO format
	# Converting tensor slice to COO format
	COOTensorSlice = coo_matrix(TensorSlice)
	nonZeroRow = COOTensorSlice.row
	# print(nonZeroRow[:10])
	nonZeroColumn = COOTensorSlice.col
	# print(nonZeroColumn[:10])
	nonZeroVal = COOTensorSlice.data
	# print(nonZeroVal[:10])

	def nonZeroInfo(nonZeroRow,nonZeroColumn,nonZeroVal):
		''' return [(r1,c1,v1),(r2,c2,v2),...]'''
		returnData = []
		numNonZeros = len(nonZeroVal)
		for idx in range(0,numNonZeros):
			triplet = (nonZeroRow[idx],nonZeroColumn[idx],nonZeroVal[idx])
			# print(triplet)
			returnData.append(triplet)
		return returnData
	nonZeroData = nonZeroInfo(nonZeroRow,nonZeroColumn,nonZeroVal)

	# Fit computation part
	TotalSliceReconSquaredError = 0
	for idx in nonZeroData:
		subjectIndex = idx[0]
		objectIndex = idx[1]
		verbPhrase = dictIndextoVP[relNum]
		# print("Triple being calculated : \n")
		# print("Triple Score : %f" %idx[2])
		# print("Subject : %s" %dictIndextoNP[subjectIndex])
		# print("Relation : %s" %verbPhrase)
		# print("Object : %s" %dictIndextoNP[objectIndex])
		tripletReconValue = A[subjectIndex].dot(Rk.dot(A[objectIndex])) # reconstruction value of  triplet after factorization
		# print("Triple reconstruction score %f" %tripletReconValue)
		# <ai,Rk,aj>
		tripletReconSquaredError = (idx[2] - tripletReconValue)**2
		TotalSliceReconSquaredError += tripletReconSquaredError
	reconErrorNorm = TotalSliceReconSquaredError**0.5
	Fit = 1 - reconErrorNorm/tensorSliceNorm
	return verbPhrase,relNum, tensorSliceNorm, reconErrorNorm, Fit

def RelationReport(Tensor,RelationTensor,matrixA,dictIndextoVP,dictVPtoIndex,lambdaFolderName,runDir,num_cores_for_fit_computation):
	numVP = len(RelationTensor)
	# dummy code to help parallelize
	RelIndexFitReport = [] # List of index to fit, indices to be sorted based on fit [(verbPhrase, relNum, tensorSliceNorm, normResidueSlice, Fit), tuples]
	# for relIndex in range(0,numVP):
	# 	verbPhrase,relNum,tensorSliceNorm, normResidueSlice, Fit = computeSliceFit(Tensor[relIndex],RelationTensor[relIndex],matrixA,dictIndextoVP,relIndex)
	# 	RelIndexFitReport.append((verbPhrase,relNum,tensorSliceNorm, normResidueSlice, Fit))
	RelIndexFitReport = Parallel(n_jobs=num_cores_for_fit_computation, verbose=1)(delayed(computeSliceFit)(Tensor[relIndex],RelationTensor[relIndex],matrixA,dictIndextoVP,relIndex) for relIndex in range(0,numVP))
	RelIndexFitReport.sort(key = lambda x:x[4],reverse=True) # sort based on fit of relations
	# print(RelIndexFitReport) # check whether sorted.
	# print('Printing Path')
	# print(os.path.join(lambdaFolderName,runDir,'RelationReport.txt'))
	# Writing old relation Report to a file	
	RelationReportHandle = open(os.path.join(os.getcwd(),lambdaFolderName,runDir,'RelationReport.txt'),'w')
	for lineInfo in RelIndexFitReport:
		line = 'Relation: '+ str(lineInfo[0])+'\t' +' Relation Number: '+ str(lineInfo[1])+'\t' +' sliceNorm: '+str(lineInfo[2])+'\t' +'errorNorm: '+str(lineInfo[3])+'\t'+' SlicewiseFit: '+str(lineInfo[4])+'\n'
		print(line)
		RelationReportHandle.write(line)
	RelationReportHandle.close()
	return RelIndexFitReport

def wordIntrusionFileCreator(A, dictNPtoIndex, dictIndextoNP, TopK, lambdaFolderName, runDir):
	''' Creates a file for word intrusion evaluation in the following format
	Latent Factor <Number> : <tab><phrase1><tab><phrase 2>.....<line break>'''
	sA_I = np.flipud(np.argsort(A,axis = 0,kind = 'mergesort'))
	NPCount, LCCount = A.shape
	numOfPurePhrases = 4 #TopK/2 
	phrasesForFile = []

	fileHandle = open(os.path.join(os.getcwd(),lambdaFolderName,runDir,'wordIntrusionFile.txt'),'w')
	answerHandle = open(os.path.join(os.getcwd(),lambdaFolderName,runDir,'wordIntrusionAnswer.txt'),'w')
	for LCIndex in range(0,LCCount):
		# pdb.set_trace()
		fileHandle.write('Latent Factor : '+str(LCIndex))
		pureSampleList = [dictIndextoNP[index] for index in list(sA_I[0:numOfPurePhrases,LCIndex])] 
		# pick and intruder randomly from the bottom NPCount/10
		lotteryNum = randint(int(-NPCount/10),-1)
		answerHandle.write(dictIndextoNP[sA_I[lotteryNum,LCIndex]]+'\n')
		pureSampleList.append(dictIndextoNP[sA_I[lotteryNum,LCIndex]]) ## Adding intruder
		shuffle(pureSampleList)
		phrasesForFile = pureSampleList
		for phrase in phrasesForFile:
			fileHandle.write('\t'+phrase)
		fileHandle.write('\n')
		# pdb.set_trace()
	answerHandle.close()
	fileHandle.close()

def relationSimilarityEvaluator(R, dictVPtoIndex, dictIndextoVP, parentFolder, lambdaFolderName, runDir):
	''' Printing || Xk - Xj || for 2 verb phrases indexed k and j, where both were deemed similar 
	input data is from data/similarRelations.txt'''
	''' Output will be of the form D(VP1,VP2) = <value>\n , where VP1 will be the "left verb phrase"'''
	# loading similar relations.txt
	fileHandle = open(os.path.join(parentFolder,'data','similarRelations.txt'),'r')
	lines = fileHandle.readlines()
	similarVerbPhrasesList = [line.strip() for line in lines]
	print(similarVerbPhrasesList[-1])
	# input('press any key')
	# del lines
	fileHandle.close()

	# computing norms
	fileHandle = open(os.path.join(os.getcwd(),lambdaFolderName,runDir,'RelationSimilarityReport.txt'),'w')
	for data in similarVerbPhrasesList:
		info = data.split('\t')
		l_VerbPhrase = info[0]
		try:
			index_l_VerPhrase = dictVPtoIndex[l_VerbPhrase]
			Rk = R[index_l_VerPhrase]
		except KeyError:
			continue
		for similarRel in info[1:]:
			try:
				indexRel = dictVPtoIndex[similarRel]
				Rj = R[indexRel]
				distance = np.linalg.norm(Rk-Rj)
				fileHandle.write('D(%s,%s) = %f' %(l_VerbPhrase,similarRel,distance))
				fileHandle.write('\n')
			except KeyError:
				continue
		fileHandle.write('\n')
	fileHandle.close()






# Decompose tensor using Rescal
for itr in range(minRank,maxRank+1,step):
	print('NNDSVD Initialization. compute_fit = %s'%(str(fitFlag)))
	# add options for positive and normal rescal
	start = timeit.default_timer()
	A, R, V, fval, iter1, exectimes = nnr_als(XList, D, RS, itr, maxIter = maxIters, compute_fit = fitFlag, lambda_A = lA, lambda_R = lR, lambda_V = lV, Ws = Ws, Wrs = Wrs)
	endFac = timeit.default_timer()
	# Immediately change the working directory to OutFolder/rank and after the Decomp is terminated. cd .. to outfolder. Repeat
	TopRC = min(TopRC,itr**2)


	runDir = 'Dimension'+str(itr)
	if runDir not in os.listdir(os.path.join(subRoot,lambdaFolderName)):
		os.chdir(lambdaFolderName)
		os.mkdir(runDir)
		os.chdir(subRoot) # ***** Dir Change *****
		print(os.getcwd()) # To check - Answer = subRoot

	# Saving np arrays for easy retrieval for word intrusion, relation report etc [ Evaluation Work along with the mappings of Phrases ]
	np.save(os.path.join(os.getcwd(),lambdaFolderName,runDir,'matrixA'),A) # Saving A matrix in subRoot/lambda/DimX/
	np.save(os.path.join(os.getcwd(),lambdaFolderName,runDir,'RelationTensor'),R) # Saving R Tensor in subRoot/lambda/DimX/
	np.save(os.path.join(os.getcwd(),lambdaFolderName,runDir,'sideInfoBasis'),V)

	fileHandle = open(os.path.join(os.getcwd(),lambdaFolderName,runDir,'output.pkl'),'wb') # Saving A and R in a pickle dump
	pickle.dump(A,fileHandle)
	pickle.dump(R,fileHandle)
	pickle.dump(V,fileHandle)
	fileHandle.close()
	# Saving ends
	try:
		# Printing TopK phrases for a LC in A
		TopPhrasesForAllLatentComponent(A,dictNPtoIndex,dictIndextoNP,Top,lambdaFolderName,runDir)
		#printing done


		# Generating Relation Report
		RelationReportData = RelationReport(XList,R,A,dictIndextoVP,dictVPtoIndex,lambdaFolderName,runDir,num_cores_for_fit_computation)
		#done
		
		# Generating Relation Matrices File and Fit File
		RelationMatrix(XList,R,A,D,V,dictIndextoVP,dictVPtoIndex,RelationReportData,TopRC,lambdaFolderName,runDir)
		# Done

		# Generating Summary.txt
		SummaryGen(A,R,RelationReportData,Top,TopRC)
		#done

		# word intrusion test.
		wordIntrusionFileCreator(A, dictNPtoIndex, dictIndextoNP, Top, lambdaFolderName, runDir)
		#done

		# Generate a report to gauge achieved relation similarity
		relationSimilarityEvaluator(R, dictVPtoIndex, dictIndextoVP, Root, lambdaFolderName, runDir)
			#done
	except:
		pass
	end = timeit.default_timer()
	FullRuntime = end - start
	fileHandle = open(os.path.join(os.getcwd(),lambdaFolderName,runDir,'timeTaken.text'),'w')
	fileHandle.write("Time Taken (seconds): %s\n"%str(FullRuntime))
	fileHandle.close()