#!/usr/bin/env python
# encoding: utf-8
# File Name: eigen.py
# Author: Jiezhong Qiu
# Create Time: 2017/07/13 16:05
# TODO:

import scipy.sparse as sparse
from scipy.sparse import csgraph
from sklearn import preprocessing
from scipy.special import iv
from scipy import linalg
import numpy as np
import time
import logging
import networkx as nx

logger = logging.getLogger(__name__)

def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    evals = np.maximum(evals, 0)
    logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f",
            np.max(evals), np.min(evals))
    return evals

def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} W D^{-1/2}
    X = sparse.identity(n) - L
    logger.info("Eigen decomposition...")
    #evals, evecs = sparse.linalg.eigsh(X, rank,
    #        which=which, tol=1e-3, maxiter=300)
    evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
    logger.info("Computing D^{-1/2}U..")
    D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU

def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    Y = X.dot(X.T)*(vol/b)
    Y = np.log(np.maximum(1, Y))
    logger.info("Computed DeepWalk matrix with %d non-zero elements",
            np.count_nonzero(Y))
    return sparse.csr_matrix(Y)

def svd_deepwalk_matrix(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sparse.diags(np.sqrt(s)).dot(u.T).T


def netmf_large(G, window, rank, dim, negative):
    logger.info("Running NetMF for a large window size...")
    logger.info("Window size is set to be %d", window)
    # load adjacency matrix
    A = nx.adjacency_matrix(G)
    vol = float(A.sum())
    # perform eigen-decomposition of D^{-1/2} A D^{-1/2}
    # keep top #rank eigenpairs
    evals, D_rt_invU = approximate_normalized_graph_laplacian(A, rank=rank, which="LA")

    # approximate deepwalk matrix
    deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU,
            window=window,
            vol=vol, b=negative)

    # factorize deepwalk matrix with SVD
    deepwalk_matrix.eliminate_zeros()
    logger.info("running SVD with %d nnz", deepwalk_matrix.nnz)
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)

    #  deepwalk_embedding = preprocessing.normalize(deepwalk_embedding, "l2")
    return deepwalk_embedding


def direct_compute_deepwalk_matrix(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} A D^{-1/2}
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        logger.info("Compute matrix %d-th power", i+1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    M = np.log(np.maximum(1, M))
    return M

def ChebyshevGaussian(A, a, order=5, mu=0.5, s=0.5):
    n = A.shape[0]
    logger.info('Chebyshev Series -----------------')
    t1 = time.time()

    if order == 1:
        return a

    A = sparse.eye(n) + A
    DA = preprocessing.normalize(A, norm='l1')
    L = sparse.eye(n) - DA

    M = L - mu * sparse.eye(n)

    Lx0 = a
    Lx1 = M.dot(a)
    Lx1 = 0.5* M.dot(Lx1) - a

    conv = iv(0,s)*Lx0
    conv -= 2*iv(1,s)*Lx1
    for i in range(2, order):
        Lx2 = M.dot(Lx1)
        Lx2 = (M.dot(Lx2) - 2*Lx1) - Lx0
        if i%2 ==0:
            conv += 2*iv(i,s)*Lx2
        else:
            conv -= 2*iv(i,s)*Lx2
        Lx0 = Lx1
        Lx1 = Lx2
        del Lx2
        logger.info('Bessell time %d %f', i, time.time()-t1)
    return A.dot(a-conv)

def spectral_propagation(emb, G, dim, pro_steps=10, pro_mu=0.2, pro_theta=0.5):
    # see section 3.2 of https://www.ijcai.org/proceedings/2019/0594.pdf
    A = nx.adjacency_matrix(G)
    emb = ChebyshevGaussian(A, emb, order=pro_steps, mu=pro_mu, s=pro_theta)
    t1 = time.time()
    U, s, Vh = linalg.svd(emb, full_matrices=False,  check_finite=False, overwrite_a=True)
    U = np.array(U)
    U = U[:, :dim]
    s = s[:dim]
    print(s[:10])
    s = np.sqrt(s)
    U = U * s
    U = preprocessing.normalize(U, "l2")
    logger.info('Densesvd time %f', time.time() - t1)

    return U


