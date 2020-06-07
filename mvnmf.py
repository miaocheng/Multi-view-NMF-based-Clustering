# -------------------------------------------------------------------------------------------------------
# This python file contains the implementation of multi-view NMF algorithm
# 
# Reference:
# J. Liu, C. Wang, J. Gao, J. Han, Multi-View Clustering via Joint Nonnegative Matrix Factorization, 
# SIAM ICDM (SDM), 2013.
# Coded by Miao Cheng
# Date: 2020-02-22
# All rights reserved
# -------------------------------------------------------------------------------------------------------
import numpy as np

from cala import *


def preLabel(cV):
    nCls, nSam = np.shape(cV)
    
    B, index = iMax(cV, axis=0)   
    labels = index + 1
    
    return labels


def nonneg(Fea):
    nFea = len(Fea)
    
    for i in range(nFea):
        tmx = Fea[i]
        nRow, nCol = np.shape(tmx)
        
        mVal = np.min(np.min(tmx))
        tmx = tmx - mVal
        
        Fea[i] = tmx
        
    return Fea
    

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This function implements the multiplicative algorithm of NMF
# Reference:
# D. Lee and S. Seung, Algorithms for Non-negative Matrix Factorization,
# NIPS, 2000.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def nmf(X, r, nIter):
    xRow, xCol = np.shape(X)
    W = np.random.rand(xRow, r)
    W = justNorm(W)
    H = np.random.rand(r, xCol)
    H = justNorm(H)
    
    for ii in range(nIter):
        # +++++ Update H +++++
        tmp = np.dot(np.transpose(W), X)    # r * xCol
        tnp = np.dot(np.transpose(W), W)    # r * r
        tnp = np.dot(tnp, H)                # r * xCol
        tm = tmp / tnp
        
        H = H * tm      # r * xCol
        
        # +++++ Update W +++++
        tmp = np.dot(X, np.transpose(H))    # xRow * r
        tnp = np.dot(W, H)                  # xRow * xCol
        tnp = np.dot(tnp, np.transpose(H))  # xRow * r
        
        tm = tmp / tnp
        W = W * tm
        
        # +++++ Check the objective +++++
        tmp = np.dot(W, H)
        obj = X - tmp
        obj = norm(obj, 1)
        
        str = 'The %d-th iteration: ' %ii + '%f' %obj
        print(str)
        
        if obj < 1e-7:
            break
        
    return W, H


def totalObj(Fea, U, V, cV, lamda):
    nFea = len(Fea)
    obj = 0
    
    for i in range(nFea):
        tmx = Fea[i]
        tmu = U[i]
        tmv = V[i]
        tml = lamda[i]
        
        tmp = np.dot(tmu, tmv)
        tmp = tmx - tmp
        tm = norm(tmp, 1)
        
        q = np.sum(tmu, axis=0)
        Q = np.diag(q)      # r * r
        tmp = np.dot(Q, tmv)  # r * nCol
        tmp = tmp - cV
        tn = tml * norm(tmp, 1)
        
        tmn = tm + tn
        obj = obj + tmn
        
        
    return obj


def calObj(X, U, V, cV, Q, lamda):    
    tmp = np.dot(U, V)
    tmp = X - tmp
    tm = norm(tmp, 1)
    
    tmp = np.dot(Q, V)        # r * nCol
    tmp = tmp - cV            # r * nCol
    tn = lamda * norm(tmp, 1)
    
    obj = tm + tn
    
    return obj    
    

def pervNMF(X, U, V, cV, lamda, maxIter):
    nRow, nCol = np.shape(X)
    _, r = np.shape(U)
    
    obj = 1e7
    for ii in range(maxIter):
        # +++++ Update U +++++
        tmp = np.dot(X, np.transpose(V))    # nRow * r
        tmq = V * cV                        # r * nCol
        tmq = np.sum(tmq, axis=1)           # r * 1
        tmq = repVec(tmq, nRow)             # r * nRow
        tmq = np.transpose(tmq)             # nRow * r
        
        tm = tmp + lamda * tmq              
        
        tnp = np.dot(U, V)                  # nRow * nCol
        tnp = np.dot(tnp, np.transpose(V))  # nRow * r
        tnq = V ** 2
        tnq = np.sum(tnq, axis=1)           # r * 1
        tnq = repVec(tnq, nRow)             
        tnq = np.transpose(tnq)             # nRow * r
        tnq = U * tnq                       # nRow * r
        tnq = np.sum(tnq, axis=0)           # 1 * r
        tnq = repVec(tnq, nRow)             
        tnq = np.transpose(tnq)             # nRow * r
        
        tn = tnp + lamda * tnq
        
        tmn = tm / tn
        U = U * tmn
        
        # +++++ Normalize U and V +++++
        q = np.sum(U, axis=0)   # 1 * r
        Q = np.diag(q)
        tmp = q ** -1
        Qf = np.diag(tmp)
        
        U = np.dot(U, Qf)   # nRow * r
        V = np.dot(Q, V)    # r * nCol
        
        # +++++ Update V +++++
        tmp = np.dot(np.transpose(X), U)    # nCol * r
        tmq = np.transpose(cV)              # nCol * r
        tm = tmp + lamda * tmq
        
        tnp = np.dot(np.transpose(V), np.transpose(U))      # nCol * nRow
        tnp = np.dot(tnp, U)                                # nCol * r
        tnq = np.transpose(V)
        tn = tnp + lamda * tnq
        
        tmn = tm / tn
        tmn = np.transpose(tmn)         # r * nCol
        V = tmn * V                     # r * nCol
        
        # +++++ Check the objective +++++
        oldObj = obj
        obj = calObj(X, U, V, cV, Q, lamda)
        tmp = obj - oldObj
        delta = norm(tmp, 2)
        if delta < 1e-7:
            break
        
    str = 'The final objective: %f' %obj
    print(str)
        
    return U, V
    
    
def mvnmf(Fea, r, lamda, maxIter):
    nFea = len(Fea)
    n = len(lamda)
    assert nFea == n, 'The length of features and parameters are not identical !'
    
    tmx = Fea[0]
    nRow, nCol = np.shape(tmx)
    
    # +++++ Initialize the Matrices +++++
    U = []
    V = []
    for i in range(nFea):
        tmx = Fea[i]
        tmx = justNorm(tmx)
        Fea[i] = tmx
        
        tmu, tmv = nmf(tmx, r, 200)
        U.append(tmu)
        V.append(tmv)
        
        
    obj = 1e7
    # +++++ Iterative learning +++++
    for ii in range(maxIter):
        cV = np.zeros((r, nCol))
        # +++++ Calcuate cV +++++
        for i in range(nFea):
            tmu = U[i]
            tmv = V[i]
            q = np.sum(tmu, axis=0)
            Q = np.diag(q)
            
            tmp = np.dot(Q, tmv)    # r * nCol
            tmp = lamda[i] * tmp
            cV = cV + tmp
            
        tn = np.sum(lamda)
        cV = cV / tn
        
        # +++++ Update each view +++++
        for i in range(nFea):
            tmx = Fea[i]
            tmu = U[i]
            tmv = V[i]
            tml = lamda[i]
            
            tmu, tmv = pervNMF(tmx, tmu, tmv, cV, tml, maxIter)
            U[i] = tmu
            V[i] = tmv
            
        # +++++ Check total objective +++++
        oldObj = obj
        obj = totalObj(Fea, U, V, cV, lamda)
        str = 'The %d-th iteration: ' %ii + '%f' %obj
        print(str)
        
        tmp = obj - oldObj
        delta = norm(tmp, 2)
        if delta < 1e-7:
            break
        
        
    return U, V, cV
    
    
    
    
# ===============================================================================================
#def mvnmf(Fea, r, lamda, nIter):
    #nFea = len(Fea)
    #n = len(lamda)
    #assert nFea == n, 'The length of features and parameters are not identical !'
    
    #tmx = Fea[0]
    #nDim, nSam = np.shape(tmx)
    
    #U = np.zeros((nDim, r, nFea))
    #V = np.zeros((r, nSam, nFea))
    #for i in range(nFea):
        #tmx = Fea[i]
        #tmx = justNorm(tmx)
        #Fea[i] = tmx
        
        #tmu, tmv = nmf(tmx, r, 100)
        #U[:, :, i] = tmu
        #V[:, :, i] = tmv
        
        
    #oldL = 100
    #ii = 0
    ## +++++ Initialize parameters +++++   
    #while ii < nIter:
        #if ii == 0:
            #cV = V[:, :, 0]
        #else:
            #cV = lamda[0] * V[:, :, 0]
            #for jj in range(nFea):
                #if jj > 0:
                    #tmv = lamda[jj] * V[:, :, jj]
                    #cV = cV + tmv
                    
            #tmp = np.sum(lamda)
            #cV = cV / tmp
            
        #logL = 0
        #for i in range(nFea):
            #tmx = Fea[i]
            #tmu = U[:, :, i]
            #tmv = V[:, :, i]
            
            #tmp = np.dot(tmu, np.transpose(tmv))
            #tmp = tmx - tmp
            #tmq = tmv - cV
            #tm = norm(tmp, 1)
            #tn = lamda[i] * norm(tmq, 1)
            #logL = logL + tm + tn
            
        #str = 'The current logL: %f' %logL
        #print(str)
        
        #if oldL < logL:
            #U = oldU
            #V = oldV
            #logL = oldL
            #ii = ii - 1
            #str = 'Restart this iteration !'
            #print(str)
        
        ## +++++ Save the old data +++++
        #oldU = U
        #oldV = V
        #oldL = logL
        
        #for i in range(nFea):
            #tml = lamda[i]
            #tmx = Fea[i]
            #tmu = U[:, :, i]
            #tmv = V[:, :, i]
            
            #tmu, tmv = pervNMF(tmx, r, cV, tml, tmu, tmv, nIter)
            #U[:, :, i] = tmu
            #V[:, :, i] = tmv
            
            
            
    
    

