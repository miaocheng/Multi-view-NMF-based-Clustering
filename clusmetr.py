# -----------------------------------------------------------------------------------------------------
# This file implements the essential functions for calculation of clustering accuracy
# 
# Coded by Miao Cheng
# Email: miaocheng24@outlook.com
# Date: 2020-01-17
# All Rights Reserved by the Author
# -----------------------------------------------------------------------------------------------------
import numpy as np
from cala import *

#from sklearn import metrics

def accu(a, b):
    a = np.array(a)
    b = np.array(b)
    
    if a.ndim == 1:
        m = len(a)
        a = np.reshape(a, (m, 1))
    
    if b.ndim == 1:
        n = len(b)
        b = np.reshape(b, (n, 1))
        
    m, _ = np.shape(a)
    n, _ = np.shape(b)
    assert m == n, 'The length of cluster labels are not identical !'
    
    acc = 0
    for i in range(m):
        if a[i, 0] == b[i, 0]:
            acc = acc + 1
            
    accuracy = float(acc) / m
    
    return accuracy, acc


def intersect(g1, g2):
    m = len(g1)
    n = len(g2)
    inter = []
    nInter = 0
    
    for i in range(m):
        tmp = g1[i]
        ind = seArr(tmp, g2)
        
        if ind != []:
            inter.append(tmp)
            nInter = nInter + 1
    
    return inter, nInter


def iCount(a, b):
    m = len(a)
    n = len(b)
    assert m == n, 'The length of different labels are not identical !'
    
    ind = []
    acc = 0
    for i in range(m):
        if a[i] == b[i]:
           ind.append(a[i])
           acc = acc + 1
           
    return ind, acc
    

def f_core(a, b, alpha):
    m = len(a)
    n = len(b)
    assert m == n, 'The length of different labels are not identical !'
    
    nt = 0
    nh = 0
    ni = 0
    for i in range(m):
        tml = a[i]
        ida = seArr(tml, a)
        
        tml = b[i]
        idb = seArr(tml, b)
        
        nt = nt + len(ida)
        nh = nh + len(idb)
        
        inter, nInter = intersect(ida, idb)
        if inter != []:
            ni = ni + nInter
            
    p = 0
    r = 0
    f = 1
    assert nh > 0, 'The predicated labels should be nonzero !'
    p = ni / nh
    
    assert nt > 0, 'Are you ok ?'
    r = ni / nt
    
    tmp = alpha ** 2
    tmp = tmp * p + r
    if tmp == 0:
        f = 0
    else:
        tmq = (alpha ** 2 + 1)
        f = tmq * p * r
        f = float(f) / tmp
        
    return f, p, r
    

# ++++++++++ Normalization Mutual Information ++++++++++
def nomi(a, b):
    m = len(a)
    n = len(b)
    eps = 1e-6
    
    #if L1.ndim == 1:
        #m = len(L1)
        #L1 = np.reshape(L1, (m, 1))
    
    #if L2.ndim == 1:
        #n = len(L2)
        #L2 = np.reshape(L2, (n, 1))
        
    ua = np.unique(a)
    na = len(ua)
    ub = np.unique(b)
    nb = len(ub)
    
    #if nL2 < nL1:
        #L1 = np.row_stack((L1, uL1))
        #L2 = np.row_stack((L2, uL1))
        
    #elif nL2 > nL1:
        #L1 = np.row_stack((L1, uL2))
        #L2 = np.row_stack((L2, uL2))
        
    #G = np.zeros((nL1, nL1))
    
    # +++++ Mutual Information +++++
    MI = 0
    for i in range(na):
        tml = ua[i]
        ida = seArr(tml, a)
        nr1 = len(ida)
        
        for j in range(nb):
            tml = ub[j]
            idb = seArr(tml, b)
            nr2 = len(idb)
            
            inter, ni = intersect(ida, idb)
            
            px = nr1 / m
            py = nr2 / m
            pxy = ni / m
            
            tmp = px * py
            tmp = pxy / tmp + eps
            tmp = np.log2(tmp)
            tmp = pxy * tmp
            
            MI = MI + pxy
            
    # +++++ Normalization Mutual Information +++++
    Hx = 0
    for i in range(na):
        tml = ua[i]
        ind = seArr(tml, a)
        nr = len(ind)
        
        tmp = float(nr) / m + eps
        tmp = np.log2(tmp)
        tmq = float(nr) / m
        Hx = Hx - tmq * tmp
        
    Hy = 0
    for i in range(nb):
        tml = ub[i]
        ind = seArr(tml, b)
        nr = len(ind)
        
        tmp = float(nr) / m + eps
        tmp = np.log2(tmp)
        tmq = float(nr) / m
        Hy = Hy - tmq * tmp
        
        
    momi = 2 * MI / (Hx + Hy)
    
    return momi
    
    
# ++++++++++ Adjusted Rand Index ++++++++++
def ari(a, b):
    n_samples = a.shape[0]
    classes = np.unique(a)
    clusters = np.unique(b)
    
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(a)):
        return 1.0
    
    contingency = contingency_matrix(a, b)
    
    # Compute the ARI using the contingency data
    sum_comb_c = sum(comb2(n_c) for n_c in contingency.sum(axis=1)) 
    sum_comb_k = sum(comb2(n_k) for n_k in contingency.sum(axis=0))
    
    sum_comb = sum(comb2(n_ij) for n_ij in contingency.flatten())
    prod_comb = (sum_comb_c * sum_comb_k) / float(comb(n_samples, 2))
    mean_comb = (sum_comb_k + sum_comb_c) / 2
    
    tmp = sum_comb - prod_comb 
    tmq = mean_comb - prod_comb
    ari = tmp / tmq
    
    return ari
    
    
    