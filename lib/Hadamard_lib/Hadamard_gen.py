from typing import List
import numpy as np
from scipy.linalg import toeplitz, hankel
import os

def nextHadamard(n: int):
    '''
    nextHadamard Returns a Hadamard matrix.
    nextHadamard(N) gives a smallest Hadamard matrix with at least N+1 rows
    in normalized form. For N up to 2^8, there are at most N+4 rows. For N
    higher than 2^8, excess rows appear to be bounded by sqrt(N).

    Some known hadamard matrices have been included in this library,
    obtained from http://neilsloane.com/hadamard/ , made available by
    N.J.A. Sloane.

    See also hadamard.   
    2016 Vicente Parot
    Cohen Lab - Harvard University

    Parameters
    ----------
    n : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    ''' 
    assert type(n)==int, 'input type of n should be int'
    if n < 2:
        h = Hadamard(n+1)
    else:
        h = np.array([])
        n = n//4*4
        while len(h)==0:
            n += 4
            h = libHadamard(n)
    return h


def libHadamard(n: int):
    '''
    libHadamard constructs a Hadamard matrix.
    libHadamard(N) returns a normalized Hadamard matrix of size N, or empty
    matrix if the algorithm doesn't make that size of Hadamard matrix.
    libHadamard(N,testOnly) returns true or false depending on whether the
    algorithm makes that size of Hadamard matrix.

    Some known hadamard matrices have been included in this library,
    obtained from http://neilsloane.com/hadamard/ , made available by
    N.J.A. Sloane.

    See also hadamard.
    2016 Vicente Parot
    Cohen Lab - Harvard University

    Parameters
    ----------
    n : int
        _description_
    '''   
    # factors = pickFactor(n)
    # list of some stored Hadamard matrix
    hlib = [28, 36, 44, 52, 56, 60, 68, 72, 76, 84, 88, 92, 100, 104, 108, 112, 116, 120,
            124, 132, 136, 140, 144, 148, 152, 156, 164, 168, 172, 176, 180, 184, 188,
            196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252]

    if n in hlib: 
        parent_path = os.path.dirname(__file__)
        lib_path = os.path.join(parent_path, 'hadlib')
        fname = [file for file in os.listdir(lib_path) if f'had.{n}' in file].pop()
        fpath = os.path.join(lib_path, fname)

        # read Hadamard matrix stored .txt file
        h = []
        with open(fpath) as f:
            lines = f.readlines()
            for line in lines:
                h.append(list(line.replace('+', '1').replace('-', '0').strip()))
        h = np.array(h, dtype=int)
        h[h==0] = -1

    elif ((np.log2(n) % 1 == 0) and (np.log2(n) >= 0)) or \
        ((np.log2(n/12) % 1 == 0) and (np.log2(n/12) >= 0)) or \
        ((np.log2(n/20) % 1 == 0)) and (np.log2(n/20) >= 0):
        h = Hadamard(n)
    else:
        h = np.array([])

    return h

def Hadamard(n: int):
    if (np.log2(n) % 1 == 0) and (np.log2(n) >= 0):
        H = np.ones(1)
        num_loop = int(np.log2(n))
    elif (np.log2(n/12) % 1 == 0) and (np.log2(n/12) >= 0):
        H = np.block([[np.ones((1, 12))], 
            [np.ones((11, 1)), toeplitz([-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],[-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1])]])
        num_loop = int(np.log2(n/12))
    elif (np.log2(n/20) % 1 == 0) and (np.log2(n/20) >= 0):
        H = np.block([[np.ones((1, 20))], 
            [np.ones((19, 1)), hankel([-1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1], [1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1])]])
        num_loop = int(np.log2(n/20))
    else:
        return np.array([[]])

    for i in range(num_loop):
        H = np.block([[H, H],
                    [H, -H]])
    return H            

        
def pickFactor(product: int, possibleFactors: List=[]):
    '''
    pickFactors Factorizes into dimensions of known Hadamard matrix sizes.
    pickFactors(N) returns a factorization of the input product into
    factors in the set of known hadamard matrix sizes [2 4:4:2^8]. If this
    is not possible, returns an empty array. The algorithm is greedy: from
    multiple options, it will select the one with largest factors.

    Some known hadamard matrices have been included in this library,
    obtained from http://neilsloane.com/hadamard/ , made available by
    N.J.A. Sloane.

    See also nextHadamard.   
    2016 Vicente Parot
    Cohen Lab - Harvard University

    Parameters
    ----------
    prodcut : _type_
        _description_
    possibleFactors : List, optional
        _description_, by default []
    '''   
    if product <= 0:
        return 0

    if len(possibleFactors)==0:
        possibleFactors = np.array([2]+[i for i in range(4, 2**8+4, 4)])
    
    possibleFactors = possibleFactors[possibleFactors!=1]

    divIdx = np.argwhere(product % possibleFactors==0).reshape(-1).tolist()
    
    if divIdx==[]:
        return 0
    else:
        for idx in divIdx[::-1]:
            lowerFactors = pickFactor(product / possibleFactors[idx], possibleFactors)

            if lowerFactors==0:
                factors = possibleFactors[idx]
            else:
                factors = [lowerFactors, possibleFactors[idx]]
            
            if product==np.prod(factors):
                break
            else:
                factors = 0
        return factors

