"""
COLPP - Class-constrained Orthogonal Locality Preserving Projections
====================================================================

Linear-algebra core shared by OHRP and S-OHRP. It builds the orthogonal,
locality-preserving subspace onto which asset returns are projected before
hierarchical allocation.

Pipeline
--------
    affinity_matrix    -> heat-kernel similarity over a k-NN graph.
                          `olpp=True`  ignores labels -> classic OLPP (OHRP)
                          `olpp=False` honours labels -> sector-constrained (S-OHRP)
    diagonal           -> degree matrix D of the locality graph
    SVD / cut_on_ratio -> PCA stage, keeping a fraction `r` of the variance
    build_weights      -> orthogonal basis, one direction at a time, each solving
                          the LPP eigenproblem restricted to the orthogonal
                          complement of the directions already found
    project_data       -> maps the returns onto the resulting subspace

The heat-kernel bandwidth is estimated by `find_optimal` as the mean pairwise
Euclidean distance across all return series.
"""

import pandas
import numpy

from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from scipy.linalg import svd
from scipy.linalg import cholesky
from scipy.linalg import eig
from scipy.sparse import spdiags

class COLPP():

    def find_optimal(df):
        nSmp = df.shape[0]
        if nSmp > 3000:
            pass
        else:
            t = squareform(pdist(df, metric = "euclidean"))
            t = numpy.mean(numpy.mean(t, axis = 1))
        return t

    def __init__(self) -> None:
        pass

    def max_with_transpose(matrix):
        matrix[matrix.T > matrix] = matrix.T
        return matrix.copy()

    def diagonal(affinity):
        D = numpy.diag(affinity.sum(axis = 1))
        return D
    
    def svd_decompose(matrix):
        usv = svd(matrix, full_matrices = False)
        return (
            pandas.DataFrame(usv[0]), 
            pandas.DataFrame(numpy.diag(usv[1])), 
            pandas.DataFrame(usv[2])
        )
    
    def cut_on_ratio(U, S, V, pca_ratio = 1.0):
        eigenvalue_pca =  numpy.diag(S)
        sum_eigen = numpy.sum(eigenvalue_pca)
        sum_eigen *= pca_ratio
        dynamic_sum = 0
        for d in range(len(eigenvalue_pca)):
            dynamic_sum += eigenvalue_pca[d]
            if dynamic_sum >= sum_eigen:
                break
        d += 1

        U = U.iloc[:, 0 : d]
        S = S.iloc[0 : d, 0 : d]
        V = V.iloc[:, 0 : d]
        return U, S, V
    
    def affinity_matrix(df, labels, ops, self_connection = False, olpp = False):
        labels_cp = labels.copy()
        # MATRIZ DE AFINIDADE ENTRE VIZINHOS DA MESMA CLASSE
        if olpp: # VERSÃO CLASSICA DO OLPP
            labels_cp *= 0

        Label = labels_cp.label.unique().tolist() # LABELS INDIVIDUAIS
        nLabel = len(Label)                    # NUMERO DE LABELS
        nSmp = df.shape[0]                     # TOTAL SAMPLES ON DF
        k = ops["k"]
        if ops["NeighborMode"] == "NonSupervised": #TODO CRIAR VERSAO OLPP (N/SUPERVISIONADA)
            # PROCESSO SUPERVISIONADO (COM LABELS)
            if ops["WeightMode"] == "HeatKernel": # USE HEAT KERNELS
                W = pandas.DataFrame()
                # ITER OVER CLASS
                for i in range(nLabel):
                    classIdx = labels_cp[labels_cp.label == i].index # INDEX OF EACH LABEL
                    label_data = df.copy().loc[classIdx,:]
                    D = squareform(pdist(label_data, metric = "euclidean")) ** 2
                    for t in range(D.shape[0]):
                        aux = D[t,:]
                        trunc = numpy.min([aux.shape[0] - 1, k]) # EVITA BUGS EM CONJUNTOS PEQUENOS
                        if (self_connection) or (trunc == 0):
                            aux[aux > sorted(aux)[trunc]] = numpy.nan
                        else:
                            aux[(aux > sorted(aux)[trunc]) | (aux == 0)] = numpy.nan
                        D[t,:] = aux
                    D = pandas.DataFrame(
                        numpy.e**((-1 * D)/(2*(ops["t"]**2))),
                        index = classIdx,
                        columns = classIdx
                    )
                    W = pandas.concat([W, D], axis = 1).fillna(0)
        elif ops["NeighborMode"] == "Supervised":
            # PROCESSO SUPERVISIONADO (COM LABELS)
            if ops["WeightMode"] == "HeatKernel": # USE HEAT KERNELS
                W = pandas.DataFrame()
                # ITER OVER CLASS
                for i in range(nLabel):
                    classIdx = labels_cp[labels_cp.label == i].index # INDEX OF EACH LABEL

                    label_data = df.copy().loc[classIdx,:]

                    D = squareform(pdist(label_data, metric = "euclidean")) ** 2
                    for t in range(D.shape[0]):
                        aux = D[t,:]
                        if pandas.isna(k):
                            trunc = aux.shape[0] - 1
                        else:
                            trunc = numpy.min([aux.shape[0]-1, k]) # EVITA BUGS EM CONJUNTOS PEQUENOS
                        if (self_connection) or (trunc == 0):
                            aux[aux > sorted(aux)[trunc]] = numpy.nan
                        else:
                            aux[(aux > sorted(aux)[trunc]) | (aux == 0)] = numpy.nan
                        D[t,:] = aux
                    D = pandas.DataFrame(
                        D, index = classIdx,
                        columns = classIdx
                    )

                    W = pandas.concat([W, D], axis = 1).fillna(0)
                    
        else:
            pass

        W = COLPP.max_with_transpose(matrix = W).reset_index(drop = True)
        W.columns = [i for i in range(W.shape[1])] # EVITA BUG FUTURO
        return W
    
    def SVD(X, reduced_dim = 0):

        numpy.random.seed(72435)

        MAX_MATRIX_SIZE = 1600
        EIGVECTOR_RATIO = 0.1
        size = X.shape

        if size[1]/size[0] > 1.0713:

            ddata = X @ X.T
            ddata = COLPP.max_with_transpose(ddata)
            dimMatrix = ddata.shape[1]

            if (reduced_dim > 0) and (dimMatrix > MAX_MATRIX_SIZE) and (reduced_dim < (dimMatrix*EIGVECTOR_RATIO)):
                pass
            else:
                eigvalue, basic_U = eig(ddata)
                eigvalue = numpy.real(eigvalue).tolist()
                ranked_eigvalue = sorted(eigvalue)[::-1]
                U = numpy.zeros(basic_U.shape)
                for t in range(U.shape[1]):
                    previous_index = eigvalue.index(ranked_eigvalue[t])
                    U[:, t] = basic_U[:, previous_index]
                eigvalue = pandas.DataFrame(ranked_eigvalue)

            maxEigValue = numpy.max(numpy.abs(eigvalue))
            eigvalue = eigvalue[numpy.abs(eigvalue)/maxEigValue > 1E-10]
            eigvalue = numpy.array(eigvalue.transpose().dropna(axis = 1))
            U = U[:,:eigvalue.shape[1]]

            if (reduced_dim > 0) and (reduced_dim < eigvalue.shape[1]):
                eigvalue = eigvalue[:,:reduced_dim]
                U = U[:,:reduced_dim]

            eigvalue_Half = eigvalue**0.5
            S = spdiags(eigvalue_Half, 0, eigvalue_Half.shape[1], eigvalue_Half.shape[1]).todense()

            eigvalue_MinusHalf = eigvalue_Half**-1

            V = numpy.zeros((U.shape[0], eigvalue_MinusHalf.shape[1]))
            for t in range(U.shape[0]):
                V[t, :] = eigvalue_MinusHalf
            V *= U
            V = X.T @ V
            return pandas.DataFrame(U), pandas.DataFrame(S), pandas.DataFrame(V)

        else:
            ddata = X.T @ X
            ddata = COLPP.max_with_transpose(ddata)
            eigenvalues, eigenvectors = eigh(ddata)
            eigenvalues = eigenvalues[::-1]
            eigenvectors = numpy.asarray(eigenvectors)[:, ::-1]

            max_eig_val = numpy.max(numpy.abs(eigenvalues))
            keep = numpy.abs(eigenvalues) / max_eig_val >= 1E-10
            eigenvalues  = eigenvalues[keep]
            eigenvectors = eigenvectors[:, keep]

            if reduced_dim > 0 and reduced_dim < len(eigenvalues):
                eigenvalues  = eigenvalues[:reduced_dim]
                eigenvectors = eigenvectors[:, :reduced_dim]

            eigenvalues_half = eigenvalues ** 0.5
            S = numpy.diag(eigenvalues_half)
            eigenvalues_minus_half = eigenvalues_half ** -1

            U = eigenvectors * eigenvalues_minus_half
            U = X @ U
            V_df = pandas.DataFrame(eigenvectors)
            if hasattr(X, 'columns') and V_df.shape[0] == X.shape[1]:
                V_df.index = X.columns
            return U, pandas.DataFrame(S), V_df
        
    def cholesky_with_fixed_column_signs(A, lower=True, eps=1e-12):
        """
        Compute Cholesky factor L of symmetric pos-def matrix A, then enforce
        a deterministic sign for each column:
        - find the element with largest absolute value in column j
        - if that element is negative, multiply the entire column by -1

        Returns L (same shape as cholesky(...)).
        """
        L = cholesky(A, lower=lower, check_finite=True)
        # iterate columns
        for j in range(L.shape[1]):
            col = L[:, j]
            # index of largest absolute entry
            idx = numpy.argmax(numpy.abs(col))
            val = col[idx]
            if numpy.abs(val) <= eps:
                # column near-zero: skip or force positive (choose skip)
                continue
            if val < 0:
                L[:, j] = -col
        return L
        
    def forceArpackConvergence(df, maxiter = 5000, tol = 1e-6):
        localMaxIter = maxiter
        localTol     = tol
        while True:
            try:
                eigV, eigVec = eigs(df.values, 1, which = "LR", v0 = numpy.ones(df.shape[0]), maxiter=localMaxIter, tol=localTol)
                return eigV, eigVec
            except Exception:
                localMaxIter *= 2
                localTol     *= 10

    def adjustedCholesky(M):
        try:
            return cholesky(M)
        except:
            eigenvalues = numpy.linalg.eigvalsh(M)

            if eigenvalues.min() <= 0:
                # Fix it
                M += (-eigenvalues.min() + 1e-8) * numpy.eye(M.shape[0])

            return cholesky(M, lower=True)

    def build_weights(U, S, V, D, W, reduced_dim = 2, bd = True):
        # Drop pandas labels for the pure-numeric kernel below: U may carry a
        # ticker index (SVD's "else" branch builds U via `X @ U`), while D and
        # the integer-relabelled W would then trigger pandas label-alignment
        # errors in `dataset.T @ D @ dataset` even though the shapes match.
        dataset = numpy.asarray(U @ S)
        eig_pca = V.copy()
        D_arr   = numpy.asarray(D)
        W_arr   = numpy.asarray(W)

        if bd: #TODO
            DPrime = pandas.DataFrame(dataset.T @ D_arr @ dataset)
        else:
            DPrime = pandas.DataFrame(dataset.T @ dataset)
        DPrime = COLPP.max_with_transpose(matrix = DPrime)

        WPrime = pandas.DataFrame(dataset.T @ W_arr @ dataset)
        WPrime = COLPP.max_with_transpose(matrix = WPrime)

        if (reduced_dim > WPrime.shape[1]):
            reduced_dim = WPrime.shape[1]

        rDPrime = COLPP.adjustedCholesky(DPrime)
        #rDPrime = COLPP.cholesky_with_fixed_column_signs(DPrime)
        lDPrime = rDPrime.T

        Q0 = numpy.linalg.inv(rDPrime) @ (numpy.linalg.inv(lDPrime) @ WPrime)
        Q = Q0.copy()

        eigvector = pandas.DataFrame()

        for k in range(reduced_dim):

            eigV, eigVec = COLPP.forceArpackConvergence(df = Q, maxiter = 5000, tol = 1e-6)
            eigVec = numpy.real(eigVec)
            eigV = numpy.real(eigV)

            if (numpy.abs(eigV[-1]) < 1E-6):
                break

            if eigvector.empty:
                eigvector = pandas.DataFrame(eigVec.copy())
            else:
                eigvector = pandas.concat([eigvector, pandas.DataFrame(eigVec.copy())], axis = 1)
            
            tmpD = numpy.linalg.inv(rDPrime) @ (numpy.linalg.inv(lDPrime) @ eigvector)
            
            DTran = eigvector.T
            tmptmpD = pandas.DataFrame(DTran @ tmpD)
            tmptmpD = COLPP.max_with_transpose(matrix = tmptmpD)
            rtmpttmpD = cholesky(tmptmpD)
            #rtmpttmpD = COLPP.cholesky_with_fixed_column_signs(tmptmpD)
            tmptmpD = numpy.linalg.inv(rtmpttmpD) @ (numpy.linalg.inv(rtmpttmpD.T) @ DTran)
            Q = ((-1 * numpy.array(tmpD)) @ tmptmpD)
            Q += numpy.eye(Q.shape[0])
            Q = Q @ Q0.copy()

        return pandas.DataFrame(eig_pca @ eigvector)
    
    def project_data(X, P, L):
        y = pandas.DataFrame(X @ P)
        y.columns = ["X" + str(i) for i in range(y.shape[1])]
        y["label"] = L
        return y