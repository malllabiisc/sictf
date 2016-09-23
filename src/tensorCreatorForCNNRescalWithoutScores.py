import os
import re
import pickle
import datetime
import timeit

import numpy as np
from numpy import zeros, ones, array, arange, copy, ravel_multi_index, unravel_index, linalg
from numpy import setdiff1d, hstack, hsplit, vsplit, sort, prod, lexsort, unique, bincount
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import argparse

startOfProgram = timeit.default_timer() # Starting time of the program

parser = argparse.ArgumentParser() # argparser object

patternOfInterest = '(.+)\t(.+)\t(.+)' # 3 fields, NP    VP    NP
pattern = re.compile(patternOfInterest) # regex object

Root = os.getcwd() # Root path
cwd = Root

#Arguments
parser.add_argument("inputFolder", help = "Enter the name of folder which containts the data file :",type = str)

parser.add_argument("fname", help = "Enter name of the file that is the source of triplets :",type = str)

parser.add_argument("sideInfoMatrix", help = "Enter the name the file that contains side information :",type = str)

parser.add_argument("relSim", help = "Enter the name of the file that contains Verb phrase Similarity information :", type =str)


args = parser.parse_args()

inputFolder = args.inputFolder
fname = args.fname
sideInfoMatrix = args.sideInfoMatrix
relSim = args.relSim


# Folders output by the program
outputFolderName = "filesFromData"

rescalDataDir = "toRescal"


if outputFolderName not in os.listdir(Root):
	os.mkdir(outputFolderName)
if rescalDataDir not in os.listdir(Root):
	os.mkdir(rescalDataDir)


# This part is for reading text from a file and finding triples
def readFile(cwd,filePath, fileName):
	''' Returns the contents of the file as a string'''
	fileHandle = open(os.path.join(cwd,filePath,fileName),'r') # opening the file to read
	text = fileHandle.read()
	fileHandle.close()
	return text

def findAll(pattern,text):
	''' Returns a list of tuples where each tuple is a NP1, VP , NP2 for a sentence'''	
	match = pattern.findall(text)
	return match

def readLines(cwd,filePath,fileName):
	''' Returns lines of file in a list with new line stripped'''
	with open(os.path.join(cwd,filePath,fileName),'r') as fileHandle:
		listOfLines = [line.strip() for line in fileHandle.readlines()]
	print(listOfLines)
	return listOfLines

fileText = readFile(cwd,inputFolder,fname)
matches = findAll(pattern, fileText)
matchesIndexed = []

dictNPtoIndex = {}
dictVPtoIndex = {}

ctrNP = 0
ctrVP = 0
for ext in matches:
	#debug statement
	# print(ext)
	if ext[0] not in dictNPtoIndex:
		dictNPtoIndex[ext[0]] = ctrNP
		IndexNP1 = ctrNP
		ctrNP += 1
	else:
		IndexNP1 = dictNPtoIndex[ext[0]]

	if ext[1] not in dictVPtoIndex:
		dictVPtoIndex[ext[1]] = ctrVP
		IndexVP = ctrVP
		ctrVP += 1
	else:
		IndexVP = dictVPtoIndex[ext[1]]
	if ext[2] not in dictNPtoIndex:
		dictNPtoIndex[ext[2]] = ctrNP
		IndexNP2 = ctrNP
		ctrNP += 1
	else:
		IndexNP2 = dictNPtoIndex[ext[2]]
	# tripleScore = ext[3]
	matchedExt = (IndexNP1,IndexVP,IndexNP2)
	# print(matchedExt)
	matchesIndexed.append(matchedExt)



dNP = len(dictNPtoIndex)
VP = len(dictVPtoIndex)
Triples = len(matches)

details = [str(dNP), str(VP) , str(Triples)]
print('Details :', details)

print('Saving files to Folder: %s\n'%(outputFolderName))
# Creating a Tab seprated Indexed Triples file for verification

fileHandle = open(os.path.join(cwd,outputFolderName,'TriplesIndexed'),'w')
for iEXT in matchesIndexed:
	line = str(iEXT[0])+'\t'+str(iEXT[1])+'\t'+str(iEXT[2])+'\n'
	fileHandle.write(line)
fileHandle.close()

# Saving matches Indexed
fileHandle = open(os.path.join(cwd,outputFolderName,'matchesIndexed.pkl'),'wb')
pickle.dump(matchesIndexed,fileHandle)
fileHandle.close()


# creating Index to NP and Index to VP from respective phrase to Index dicts

dummyList = list(dictNPtoIndex.keys())
dictIndextoNP = {}
for key in dummyList:
	val = dictNPtoIndex[key]
	dictIndextoNP[val] = key


dummyList = list(dictVPtoIndex.keys())
dictIndextoVP = {}
for key in dummyList:
	val = dictVPtoIndex[key]
	dictIndextoVP[val] = key

#Saving Dicts
fileHandle = open(os.path.join(cwd,outputFolderName,'domainNounPhrases.pkl'),'wb')
pickle.dump(dictNPtoIndex,fileHandle)
pickle.dump(dictIndextoNP,fileHandle)
fileHandle.close()

fileHandle = open(os.path.join(cwd,outputFolderName,'verbPhrase.pkl'),'wb')
pickle.dump(dictVPtoIndex,fileHandle)
pickle.dump(dictIndextoVP,fileHandle)
fileHandle.close()


# Saving triples indices that are sorted by relation phrase index
indexsSortedByRelation = sorted(matchesIndexed, key=lambda tup: tup[1])

fileHandle = open(os.path.join(cwd,outputFolderName,'indexsSortedByRelation.pkl'),'wb')
pickle.dump(indexsSortedByRelation,fileHandle)
fileHandle.close()


### OLD code tryout ( write afresh )

listOfSlices = []
for numRel in range(0,VP):
	mIR = [] # Triples that exist for relation number numRel
	[mIR.append(triplet) for triplet in indexsSortedByRelation if triplet[1] == numRel]
	mode1Elem = []
	[mode1Elem.append(triplet[0]) for triplet in mIR]
	mode2Elem = []
	[mode2Elem.append(triplet[2]) for triplet in mIR]
	data = np.ones(len(mIR))
	relSlice = coo_matrix((data,(mode1Elem,mode2Elem)),shape = (dNP,dNP)) # Sparse Slice
	CSRM = csr_matrix(relSlice) # Compressed Row form sparse slice
	listOfSlices.append(CSRM) # Add to raw list of slices


print('Saving files to Folder: toRescal\n')


fileHandle = open(os.path.join(cwd,rescalDataDir,'listOfSlices.pkl'),'wb')
pickle.dump(listOfSlices,fileHandle)
fileHandle.close()

endTensorcreation = timeit.default_timer()
print('Time Taken to create Tenro and write files: ', endTensorcreation - startOfProgram)

## Test the tensor properly - 

## ------------------- Side info matrix creation ----------------------

## Input : A text file with lines as follows -:
## 		<#noun phrases><white space><Number of attributes><\n> ???
## 		<Noun Phrase with whitespaces><tab><attr1Val><whiteSpace><attr2Val>....<whiteSpace><attrNVal><\n>
## 		.
## 		.
## 		.
## 		EOF

sideInfoData = readFile(cwd,inputFolder,sideInfoMatrix)
sideInfoData = sideInfoData.strip()
sideInfoData = sideInfoData.split('\n')


numAttr = len((sideInfoData[0].split('\t')[1]).split())
print('Number of Attributes: ',numAttr)


D = np.zeros([dNP,numAttr]) # the side info matrix : make this coo for large data

def populateSideInfoMatrix(data,matrix,dictNPtoIndex):
	''' data is a list of lines of the form
	<Noun Phrase with whitespaces><tab><attr1Val><whiteSpace><attr2Val>....<whiteSpace><attrNVal><\n>
	matrix is a np matrix of zeros.
	Return the matrix '''

	for line in data:
		lineContent = line.split('\t')
		NounPhrase = lineContent[0]
		try:
			IndexNP = dictNPtoIndex[NounPhrase]
		except KeyError:
			print(NounPhrase+'\n')
			continue
		nounPhraseVector = lineContent[1].split()
		matrix[IndexNP][:] = nounPhraseVector

	return matrix

D = populateSideInfoMatrix(sideInfoData,D,dictNPtoIndex)

fileHandle = open(os.path.join(cwd,rescalDataDir,'sideInfoMatrix.pkl'),'wb')
pickle.dump(D,fileHandle)
fileHandle.close()

np.save(os.path.join(cwd,rescalDataDir,'sideInfoMatrix'),D)


## ---------- Relation Similarity information construction from Rel Sim file ----------

similarVerbPhrases = readLines(cwd,inputFolder,relSim)
# print(similarVerbPhrases)

numVP = VP

def createRelSimMatrix(similarVerbPhrases, dictVPtoIndex, numVP):
	''' Returns an #VP x #VP matrix with entries at a particular index i,j 1 if Ri is similar to Rj '''
	relSimMatrix = np.zeros((numVP,numVP))
	listOfListOfPhrasesPerLine = [line.split('\t') for line in similarVerbPhrases]
	# print(listOfListOfPhrasesPerLine)
	for listOfPhrasesInALine in listOfListOfPhrasesPerLine:
		pivotVerbPhrase = listOfPhrasesInALine[0] # The verb phrase to which all others are similar
		try:
			pivotVerbPhraseIndex = dictVPtoIndex[pivotVerbPhrase]
		except KeyError:
			print(pivotVerbPhrase)
			continue
		for verbPhrase in listOfPhrasesInALine:
			try:
				relSimMatrix[pivotVerbPhraseIndex][dictVPtoIndex[verbPhrase]] = 1
				relSimMatrix[dictVPtoIndex[verbPhrase]][pivotVerbPhraseIndex] = 1
			except KeyError:
				print(verbPhrase)
				continue

	return relSimMatrix


def createRelSimEvaluator(similarVerbPhrases):
	''' Reads data from similarity file and outputs only those relations for which a similar verb phrase exists,
	 to be used to evaluate generation of predicates by relation similarity'''
	listOfSimilarPhrases = [ line+'\n' for line in similarVerbPhrases if len(line.split('\t')) > 1]
	return listOfSimilarPhrases


fileHandle = open(os.path.join(cwd,inputFolder,'similarRelations.txt'),'w')
fileHandle.writelines(createRelSimEvaluator(similarVerbPhrases))
fileHandle.close()

relSimMatrix = createRelSimMatrix(similarVerbPhrases,dictVPtoIndex,numVP)

fileHandle = open(os.path.join(cwd,rescalDataDir,'relSimMatrix.pkl'),'wb')
pickle.dump(relSimMatrix,fileHandle)
fileHandle.close()

np.save(os.path.join(cwd,rescalDataDir,'relSimMatrix'),relSimMatrix)




end = timeit.default_timer()
print('Time Taken to create and write side info matrix and relation Similarity Matrix: ', end - endTensorcreation)