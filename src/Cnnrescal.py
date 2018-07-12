# coding: utf-8
# rescal.py - python script to compute the RESCAL tensor factorization
# Copyright (C) 2013 Maximilian Nickel <mnick@mit.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import time
import numpy as np
from numpy import dot, zeros, array, eye, kron, prod
from numpy.linalg import norm, solve, inv, svd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from numpy.random import rand
import math
import scipy.sparse as sp
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
import pdb
from joblib import Parallel, delayed  
import multiprocessing
import timeit

global VarName
__version__ = "0.4"
__all__ = ['nnr-als']

_DEF_MAXITER = 100
_DEF_INIT = 'nndsvd'
_DEF_CONV = 1e-4
_DEF_LMBDA = 0
_DEF_Ws = 1
_DEF_Wrs = 1
_DEF_ATTR = []
_DEF_NO_FIT = 1e9
_DEF_FIT_METHOD = None

# pdb.set_trace()
#	-------------------- This part holds functions for NMF Init --------------------
_log = logging.getLogger('NNDSVD Initialization')
def safe_vstack(Xs):
	if any(sp.issparse(X) for X in Xs):
		return sp.vstack(Xs)
	else:
		return np.vstack(Xs)


def norm(x):
	"""Dot product-based Euclidean norm implementation

	See: http://fseoane.net/blog/2011/computing-the-vector-norm/
	"""
	return math.sqrt(squared_norm(x))


def trace_dot(X, Y):
	"""Trace of np.dot(X, Y.T)."""
	return np.dot(X.ravel(), Y.ravel())


def _sparseness(x):
	"""Hoyer's measure of sparsity for a vector"""
	sqrt_n = np.sqrt(len(x))
	return (sqrt_n - np.linalg.norm(x, 1) / norm(x)) / (sqrt_n - 1)


def check_non_negative(X, whom):
	X = X.data if sp.issparse(X) else X
	if (X < 0).any():
		raise ValueError("Negative values in data passed to %s" % whom)


def _initialize_nmf(X, n_components, variant=None, eps=1e-6,
					random_state=None):
	"""NNDSVD algorithm for NMF initialization.

	Computes a good initial guess for the non-negative
	rank k matrix approximation for X: X = WH

	Parameters
	----------

	X : array, [n_samples, n_features]
		The data matrix to be decomposed.

	n_components : array, [n_components, n_features]
		The number of components desired in the approximation.

	variant : None | 'a' | 'ar'
		The variant of the NNDSVD algorithm.
		Accepts None, 'a', 'ar'
		None: leaves the zero entries as zero
		'a': Fills the zero entries with the average of X
		'ar': Fills the zero entries with standard normal random variates.
		Default: None

	eps: float
		Truncate all values less then this in output to zero.

	random_state : numpy.RandomState | int, optional
		The generator used to fill in the zeros, when using variant='ar'
		Default: numpy.random

	Returns
	-------

	(W, H) :
		Initial guesses for solving X ~= WH such that
		the number of columns in W is n_components.

	References
	----------
	C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for 
	nonnegative matrix factorization - Pattern Recognition, 2008

	http://tinyurl.com/nndsvd
	"""
	check_non_negative(X, "NMF initialization")
	if variant not in (None, 'a', 'ar'):
		raise ValueError("Invalid variant name")

	U, S, V = randomized_svd(X, n_components)
	W, H = np.zeros(U.shape), np.zeros(V.shape)

	# The leading singular triplet is non-negative
	# so it can be used as is for initialization.
	W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
	H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

	for j in range(1, n_components):
		x, y = U[:, j], V[j, :]

		# extract positive and negative parts of column vectors
		x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
		x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

		# and their norms
		x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
		x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

		m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

		# choose update
		if m_p > m_n:
			u = x_p / x_p_nrm
			v = y_p / y_p_nrm
			sigma = m_p
		else:
			u = x_n / x_n_nrm
			v = y_n / y_n_nrm
			sigma = m_n

		lbd = np.sqrt(S[j] * sigma)
		W[:, j] = lbd * u
		H[j, :] = lbd * v

	W[W < eps] = 0
	H[H < eps] = 0

	if variant == "a":
		avg = X.mean()
		W[W == 0] = avg
		H[H == 0] = avg
	elif variant == "ar":
		random_state = check_random_state(random_state)
		avg = X.mean()
		W[W == 0] = abs(avg * random_state.randn(len(W[W == 0])) / 100)
		H[H == 0] = abs(avg * random_state.randn(len(H[H == 0])) / 100)

	return W, H

# ----------End of NMF----------

_log = logging.getLogger('Non-Negative RESCAL')

# pdb.set_trace()
def nnr_als(X, D, relSim, rank, **kwargs):
	"""
	RESCAL-ALS algorithm to compute the RESCAL tensor factorization.


	Parameters
	----------
	X : list
		List of frontal slices X_k of the tensor X.
		The shape of each X_k is ('N', 'N').
		X_k's are expected to be instances of scipy.sparse.csr_matrix
	D : Side information matrix.
		The shape of Side information matrix should be ('N','C')
		Where N is the number of noun-phrases and C is the cardinaltiy 
		of side information category. D is expected to be an instance of scipy.sparse.csr_matrix
		or a numpy matrix, depending on the sparsity and size of side information
	relSim : Relation Similarity information
		The entries i,j of the matrix S signify whether the ith and the jth relation are similar or nTo
		1 for similar, 0 for no.
		dtype = int
		size = number of Verb noun-phrases x number of Verb noun-phrases
	rank : int
		Rank of the factorization
	lmbdaA : float, optional
		Regularization parameter for A factor matrix. 0 by default
	lmbdaR : float, optional
		Regularization parameter for R_k factor matrices. 0 by default
	lmbdaV : float, optional
		Regularization parameter for V_l factor matrices. 0 by default
	Ws : Weight assigned to side information term. Default 1
	Wrs: Weight assigned to relational side information. Default 1
	attr : list, optional
		List of sparse ('N', 'L_l') attribute matrices. 'L_l' may be different
		for each attribute
	init : string, optional
		Initialization method of the factor matrices. 'nvecs' (default)
		initializes A based on the eigenvectors of X. 'random' initializes
		the factor matrices randomly.
	compute_fit : boolean, optional
		If true, compute the fit of the factorization compared to X.
		For large sparse tensors this should be turned of. None by default.
	maxIter : int, optional
		Maximium number of iterations of the ALS algorithm. 500 by default.
	conv : float, optional
		Stop when residual of factorization is less than conv. 1e-5 by default

	Returns
	-------
	A : ndarray
		array of shape ('N', 'rank') corresponding to the factor matrix A
	R : list
		list of 'M' arrays of shape ('rank', 'rank') corresponding to the
		factor matrices R_k
	V : Combination of Latent components that factorize D to A
	f : float
		function value of the factorization
	itr : int
		number of iterations until convergence
	exectimes : ndarray
		execution times to compute the updates in each iteration

	Examples
	--------
	>>> X1 = csr_matrix(([1,1,1], ([2,1,3], [0,2,3])), shape=(4, 4))
	>>> X2 = csr_matrix(([1,1,1,1], ([0,2,3,3], [0,1,2,3])), shape=(4, 4))
	>>> A, R, _, _, _ = rescal([X1, X2], 2)

	See
	---
	For a full description of the algorithm see:
	.. [1] Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
		"A Three-Way Model for Collective Learning on Multi-Relational Data",
		ICML 2011, Bellevue, WA, USA

	.. [2] Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
		"Factorizing YAGO: Scalable Machine Learning for Linked Data"
		WWW 2012, Lyon, France
	"""
	global VarName 
	VarName = relSim
	# ------------ init options ----------------------------------------------
	ainit = kwargs.pop('init', _DEF_INIT)
	maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
	conv = kwargs.pop('conv', _DEF_CONV)
	lmbdaA = kwargs.pop('lambda_A', _DEF_LMBDA)
	lmbdaR = kwargs.pop('lambda_R', _DEF_LMBDA)
	lmbdaV = kwargs.pop('lambda_V', _DEF_LMBDA)
	Ws = kwargs.pop('Ws', _DEF_Ws)
	Wrs = kwargs.pop('Wrs', _DEF_Wrs)
	compute_fit = kwargs.pop('compute_fit', _DEF_FIT_METHOD)
	P = kwargs.pop('attr', _DEF_ATTR)
	dtype = kwargs.pop('dtype', np.float)

	# ------------- check input ----------------------------------------------
	if not len(kwargs) == 0:
		raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

	# check frontal slices have same size and are matrices
	sz = X[0].shape
	for i in range(len(X)):
		if X[i].ndim != 2:
			raise ValueError('Frontal slices of X must be matrices')
		if X[i].shape != sz:
			raise ValueError('Frontal slices of X must be all of same shape')
		#if not issparse(X[i]):
			#raise ValueError('X[%d] is not a sparse matrix' % i)

	if compute_fit is None:
		if prod(X[0].shape) * len(X) > _DEF_NO_FIT:
			_log.warn('For large tensors automatic computation of fit is disabled by default\nTo compute the fit, call rescal_als with "compute_fit=True" ')
			compute_fit = False
		else:
			compute_fit = True

	n = sz[0]
	k = len(X)

	_log.debug(
		'[Config] rank: %d | maxIter: %d | conv: %7.1e | lmbda: %7.1e' %
		(rank, maxIter, conv, lmbdaA)
	)
	_log.debug('[Config] dtype: %s / %s' % (dtype, X[0].dtype))

	# ------- convert X and P to CSR ------------------------------------------
	for i in range(k):
		if issparse(X[i]):
			X[i] = X[i].tocsr()
			X[i].sort_indices()
	for i in range(len(P)):
		if issparse(P[i]):
			P[i] = P[i].tocoo().tocsr()
			P[i].sort_indices()
	# pdb.set_trace()
	# ---------- initialize A ------------------------------------------------
	_log.debug('Initializing A')
	if ainit == 'random':
		A = array(rand(n, rank), dtype=dtype)
	elif ainit == 'nndsvd':
		_log.debug('Initialization of A = NNDSVD')
		S = csr_matrix((n, n), dtype=dtype)
		for i in range(k):
			S = S + X[i]
			S = S + X[i].T
		# _, A = eigsh(csr_matrix(S, dtype=dtype, shape=(n, n)), rank)
		W,H = _initialize_nmf(S, rank)
		A = W
		# pdb.set_trace()
		A = array(A, dtype=dtype)
		# if A.min() < 0 :
		# 	print("Negative Rn!")
		# 	raw_input("Press Enter: ")
		_log.debug('Initialized A: NNDSVD')
	else:
		raise ValueError('Unknown init option ("%s")' % ainit)

	# pdb.set_trace()

	_log.debug('Initializing V')
	V = _initialize_nmf(D,rank)[1]
	print('Side Info Shape and basis shape respectively :\n')
	print(D.shape)
	print(V.shape)
	# pdb.set_trace()
	_log.debug('Initialized V')

	# ------- initialize R and Z ---------------------------------------------
	R = _initR(X, A, lmbdaR)
	# Z = _updateZ(A, P, lmbdaV)
	# pdb.set_trace()
	#  ------ compute factorization ------------------------------------------
	fit = fitchange = fitold = f = 0
	# global relSim
	exectimes = []
	for itr in range(maxIter):
		tic = time.time()
		fitold = fit
		# pdb.set_trace()
		A = _updateA(X, A, R, D, V, lmbdaA, Ws)
		# if A.min() < 0 :
		# 	print("Negative Rn!")
		# 	raw_input("Press Enter: ")
		# pdb.set_trace()
		RStart = timeit.default_timer()
		R = _updateR(X, A, R, lmbdaR, Wrs)
		REnd = timeit.default_timer()
		print(REnd - RStart)
		print('Time take for 1 run of R: ')
		# input('...: ')
		# if R[0].min() < 0 :
		# 	print("Negative Rn!")
		# 	raw_input("Press Enter: ")
		# Z = _updateZ(A, P, lmbdaV)

		V = _updateV(D,A,V, lmbdaV, Ws)

		# compute fit value
		if compute_fit:
			fit = _compute_fit(X, A, R, P, lmbdaA, lmbdaR)
		else:
			fit = itr

		fitchange = abs(fitold - fit)

		toc = time.time()
		exectimes.append(toc - tic)

		_log.info('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (
			itr, fit, fitchange, exectimes[-1]
		))
		if itr > 0 and fitchange < conv:
			break
	return A, R, V, f, itr + 1, array(exectimes)


# ------------------ Update A ------------------------------------------------

def FparA(Xk, A, AtA, Rk):
	ep = 1e-9
	F = Xk.dot(dot(A, Rk.T)) + Xk.T.dot(dot(A, Rk))
	return F


def EparA(Xk, A, AtA, Rk):
	ep = 1e-9
	E = dot(Rk, dot(AtA, Rk.T)) + dot(Rk.T, dot(AtA, Rk))
	return E





def _updateA(X, A, R,  D, V, lmbdaA, Ws):
	"""Update step for A"""
	_log.debug('Updating A')
	n, rank = A.shape
	F = zeros((n, rank), dtype=A.dtype)
	E = zeros((rank, rank), dtype=A.dtype)
	# pdb.set_trace()
	num_cores = multiprocessing.cpu_count()
	ep = 1e-9
	print(Ws)
	AtA = dot(A.T, A)
	VVt = dot(V,V.T)

	# for i in range(len(X)): #parallelize
	# 	F += X[i].dot(dot(A, R[i].T)) + X[i].T.dot(dot(A, R[i]))
	# 	E += dot(R[i], dot(AtA, R[i].T)) + dot(R[i].T, dot(AtA, R[i]))

	# regularization
	I = lmbdaA * eye(rank, dtype=A.dtype)

	Farray = Parallel(n_jobs=num_cores, verbose=1)(delayed(FparA)(X[i], A, AtA, R[i]) for i in range(len(X)))
	Earray = Parallel(n_jobs=num_cores, verbose=1)(delayed(EparA)(X[i], A, AtA, R[i]) for i in range(len(X)))

	F = sum(Farray)
	E = sum(Earray)






	# Denominator and Numerator
	Denominator = dot(A,(E+I + Ws*VVt)).__iadd__(ep)
	# pdb.set_trace()
	Numerator = F + Ws*D.dot(V.T) # Removing epsilon addtion from numerator
	Multiplicand = np.divide(Numerator,Denominator)

	A = np.multiply(A,Multiplicand)


	# attributes : Z not Needed
	# for i in range(len(Z)):
	#     F += P[i].dot(Z[i].T)
	#     E += dot(Z[i], Z[i].T)

	# finally compute update for A
	# A = solve(I + E.T, F.T).T
	#A = dot(F, inv(I + E))
	#_log.debug('Updated A lambda_A:%f, dtype:%s' % (lmbdaA, A.dtype))
	return A


# ------------------ Update R ------------------------------------------------

def parSliceR(Xk, A, At, AtA, Rk, lmbdaR, Wrs, i):
	global VarName
	global aliasR
	# global R
	ep = 1e-9
	rank = Rk.shape[0]
	k = i
	# pdb.set_trace()
	numerator1 = dot(At,Xk.dot(A)) # Removing epsilon addtion from numerator
	sumSimilarSlices = np.zeros((rank,rank)) # To account for summation over all j Skj*Rj . j!= k constraint not imposed yet.
	for relNum in range(len(aliasR)):
		if VarName[k][relNum] != 0 and k != relNum: # later k != relNum can be added
			sumSimilarSlices += aliasR[relNum]
	numerator2 = Wrs * sumSimilarSlices
	numerator = numerator1 + numerator2

	# pdb.set_trace()
	denominator1 = (dot(AtA,dot(Rk,AtA)) + lmbdaR*Rk)
	denominator2 = Wrs * sum(VarName[k]) * Rk
	denominator = denominator1 + denominator2
	# pdb.set_trace()
	Multiplicand = np.divide(numerator,denominator.__iadd__(ep))

	Rko = np.multiply(Rk,Multiplicand)
	return Rko



global aliasR
def _updateR(X, A, R, lmbdaR, Wrs):
	global aliasR
	aliasR = R
	_log.debug('Updating R , lambda R: %s' % str(lmbdaR))
	num_cores = multiprocessing.cpu_count()
	rank = A.shape[1]
	# U, S, Vt = svd(A, full_matrices=False)
	# Shat = kron(S, S)
	# Shat = (Shat / (Shat ** 2 + lmbdaR)).reshape(rank, rank)
	ep = 1e-9
	At = A.T
	AtA = dot(At, A)
	# pdb.set_trace()
	# Ro = []
	# for i in range(len(X)): # parallelize
	# 	# pdb.set_trace()
	# 	numerator = dot(At,X[i].dot(A)).__iadd__(ep)
	# 	# pdb.set_trace()
	# 	denominator = (dot(AtA,dot(R[i],AtA)) + lmbdaR*R[i]).__iadd__(ep)
	# 	# pdb.set_trace()
	# 	Multiplicand = np.divide(numerator,denominator)

	# 	Ri = np.multiply(R[i],Multiplicand)

	# 	# Rn = Shat * dot(U.T, X[i].dot(U))
	# 	# pdb.set_trace()
	# 	# Rn = dot(Vt.T, dot(Rn, Vt))
	# 	Ro.append(Ri)

	# print(relSim.shape)
	# print(R.shape)
	# input('...: ')

	#Ro = Parallel(n_jobs=num_cores, verbose=1)(delayed(parSliceR)(X[i], A, At, AtA, R[i], lmbdaR, Wrs, i) for i in range(len(X)))
	Ro = Parallel(n_jobs=num_cores, backend= 'multiprocessing',verbose=1)(delayed(parSliceR)(X[i], A, At, AtA, R[i], lmbdaR, Wrs, i) for i in range(len(X)))
	return Ro


# ------------------ Update Z ------------------------------------------------
# def _updateZ(A, P, lmbdaZ):
# 	Z = []
# 	if len(P) == 0:
# 		return Z
# 	#_log.debug('Updating Z (Norm EQ, %d)' % len(P))
# 	pinvAt = inv(dot(A.T, A) + lmbdaZ * eye(A.shape[1], dtype=A.dtype))
# 	pinvAt = dot(pinvAt, A.T).T
# 	for i in range(len(P)):
# 		if issparse(P[i]):
# 			Zn = P[i].tocoo().T.tocsr().dot(pinvAt).T
# 		else:
# 			Zn = dot(pinvAt.T, P[i])
# 		Z.append(Zn)
# 	return Z


def _compute_fit(X, A, R, P, lmbdaA, lmbdaR):
	"""Compute fit for full slices"""
	f = 0
	# precompute norms of X
	normX = [sum(M.data ** 2) for M in X]
	sumNorm = sum(normX)

	for i in range(len(X)):
		ARAt = dot(A, dot(R[i], A.T))
		f += norm(X[i] - ARAt) ** 2

	return 1 - f / sumNorm

# ------------------ Initializing R ------------------------------------------------

def _initR(X, A, lmbdaR):
	_log.debug('Initializing R (SVD) lambda R: %s' % str(lmbdaR))
	rank = A.shape[1]
	U, S, Vt = svd(A, full_matrices=False)
	Shat = kron(S, S)
	Shat = (Shat / (Shat ** 2 + lmbdaR)).reshape(rank, rank)
	R = []
	ep = 1e-9
	for i in range(len(X)): # parallelize
		Rn = Shat * dot(U.T, X[i].dot(U))
		Rn = dot(Vt.T, dot(Rn, Vt))
 
		negativeVal = Rn.min()
		Rn.__iadd__(-negativeVal+ep)
		# if Rn.min() < 0 :
		# 	print("Negative Rn!")
		# 	raw_input("Press Enter: ")
		# Rn = np.eye(rank)
		# Rn = dot(A.T,A)

		R.append(Rn)
	print('Initialized R')
	return R

# ------------------ Update V ------------------------------------------------

def _updateV(D,A,V, lmbdaV, Ws):
	"""Update step for V"""
	_log.debug('Updating V')
	rank , n = V.shape
	print(rank)
	AtA = dot(A.T,A)
	ep = 1e-9
	print(Ws)
	# pdb.set_trace()
	numerator = Ws*dot(A.T,D.toarray()) # A.T . D #method toarray() is a 'fix' for sparse and dense matrix multiplication
	denominator = Ws*dot(AtA,V) + lmbdaV*V #(A.T . A + lmbdaV*I) . V

	Multiplicand = np.divide(numerator,denominator.__iadd__(ep))

	V = np.multiply(V,Multiplicand)

	return V
