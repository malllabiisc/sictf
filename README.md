# Relation Schema Induction

Given a set of documents from a specific domain (e.g., medical research journals), how do we automatically build a Knowledge Graph (KG) for that domain? Automatic identification of relations and their schemas, i.e., type signature of arguments of relations (e.g., undergo(Patient, Surgery)), is an important first step towards this goal. We refer to this problem as Relation Schema Induction (RSI).

## SICTF

SICTF solves the problem of Relation Schema Induction, defined below. Given the surface triples as a Tensor, Noun Phrase Side Information as a Matrix and Relation similarity as a Matrix, SICTF performs a joint factorization. 


## Dependencies:

Please install the following dependencies for SCITF to run:
numpy, scipy, python3.3 or above.

## Help
Help for running the setup.

Place the triples and side information in a folder at the same level as *.py scripts

Run the Following commands sequentially afterwards ( with -h flag for help)

python3 tensorCreatorForCNNRescalWithScores.py

usage: tensorCreatorForCNNRescalWithScores.py [-h]
                                              inputFolder fname sideInfoMatrix
                                              relSim

positional arguments:
  * inputFolder     Enter the name of folder which containts the data file
  * fname           Enter name of the file that is the source of triplets
  * sideInfoMatrix  Enter the name the file that contains side information
  * relSim          Enter the name of the file that contains Verb phrase Similarity information

optional arguments:
  -h, --help      show this help message and exit


##########

python3 runCRescal.py -h

usage: runCRescal.py [-h]
                     outputFolderName minRank maxRank step Top TopRC maxIters
                     fitFlag lA lR lV Ws Wrs

positional arguments:
  * outputFolderName  Enter the name of the Output Folder :
  * minRank           Enter Rank for min Rescal Decomposition :
  * maxRank           Enter Rank for max Rescal Decomposition :
  * step              Enter step for Rank (Please ensure minRank +n*step =
                    maxRank):
  * Top               Enter cut-off for top Entities:
  * TopRC             Enter cut-off for top RelMatrix Entries:
  * maxIters          Enter Max no. of iterations:
  * fitFlag           Enter True is fit computation desired, False if not, None
                    for uncertainity (Advised to keep as False)
  * lA                Enter Lambda A
  * lR                Enter Lambda R
  * lV                Enter Lambda V
  * Ws                Enter Side Info term weight
  * Wrs               Enter Relation Similarity term weight

optional arguments:
  * -h, --help        show this help message and exit

## Reference
[1] Madhav Nimishakavi, Uday Singh Saini and Partha Talukdar. Relation Schema Induction using Tensor Factorization with Side Information. November 2016. Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016).
