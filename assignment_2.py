# -*- coding: utf-8 -*-

from numpy import asarray
import numpy as np
import sys


#STOCK_PRICES = [100, 113, 110, 85, 105, 102, 86, 63, 81, 101, 94, 106, 101, 79, 94, 90, 97]
STOCK_PRICE_CHANGES = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
B = [[1, 2, 3, 4], [5, 6, 7, 8], [3, 4, 5, 6], [7, 8, 9, 5]]
D = [[2, 9, 8, 3], [5, 6, 0, 1], [1, 3, 2, 3], [1, 1, 2, 0]]
E = [[1, 2, 3, 4], [5, 6, 7, 8], [3, 4, 5, 6], [7, 8, 9, 5]]
F = [[2, 9, 8, 3], [5, 6, 0, 1], [1, 3, 2, 3], [1, 1, 2, 0]]
#==============================================================


# The brute force method to solve max subarray problem
def find_maximum_subarray_brute(A):

    """

    Return a tuple (i,j) where A[i:j] is the maximum subarray.

    time complexity = O(n^2)

    """

    A = asarray(A)
    if len(A) == 0:
        return None
    if len(A) == 1:
        return (0, 0)
    max = -sys.maxsize-1
    sum = 0
    start = 0
    end = 0
    for i in range(0, A.size):
        sum = 0
        for j in range(i, A.size):
            sum = sum + A[j]
            if sum > max:
                max = sum
                start = i
                end = j
    return (start, end)
#==============================================================


# The maximum crossing subarray method for solving the max subarray problem
def find_maximum_crossing_subarray(A, low, mid, high):

    """

    Find the maximum subarray that crosses mid

    Return a tuple ((i, j), sum) where sum is the maximum subarray of A[i:j].

    """

    left_max = -sys.maxsize-1
    right_max = -sys.maxsize-1
    sum = 0
    i = mid
    j = mid+1
    while i >= low:
        sum = sum + A[i]
        if sum > left_max:
            left_max = sum
            cross_low = i
        i = i-1
    sum = 0
    while j <= high:
        sum = sum+A[j]
        if sum > right_max:
            right_max = sum
            cross_high = j
        j = j+1
    return ((cross_low, cross_high), left_max+right_max)


# The recursive method to solve max subarray problem
def find_maximum_subarray_recursive_helper(A, low=0, high=-1):

    """

    Return a tuple ((i, j), sum) where sum is the maximum subarray of A[i:j].


    """

    if low == high:
        return ((low, high), A[low])
    else:
        mid = int((low + high) / 2)
        l_index, l_max = find_maximum_subarray_recursive_helper(A, low, mid)
        r_index, r_max = find_maximum_subarray_recursive_helper(A, mid+1, high)
        c_index, c_max = find_maximum_crossing_subarray(A, low, mid, high)
        if l_max >= r_max and l_max >= c_max:
            return (l_index, l_max)
        elif r_max >= l_max and r_max >= c_max:
            return (r_index, r_max)
        else:
            return (c_index, c_max)


# The recursive method to solve max subarray problem
def find_maximum_subarray_recursive(A):

    """

    Return a tuple (i,j) where A[i:j] is the maximum subarray.


    """
    A = asarray(A)
    if len(A) == 0:
        return None
    return find_maximum_subarray_recursive_helper(A, 0, len(A) - 1)[0]
#==============================================================


# The iterative method to solve max subarray problem
def find_maximum_subarray_iterative(A):

    """

    Return a tuple (i,j) where A[i:j] is the maximum subarray.

    """
    A = asarray(A)
    if len(A) == 0:
        return None
    start_of_max = 0
    start = 0
    end = 0
    max_computing = A[0]
    max_till_now = A[0]
    for i in range(1, A.size):
        if A[i] > (max_computing + A[i]):
            max_computing = A[i]
            start = i
        else:
            max_computing = A[i] + max_computing
        if max_computing > max_till_now:
            max_till_now = max_computing
            start_of_max = start
            end = i
    return (start_of_max, end)
#=================================================================


def square_matrix_multiply(A, B):

    """

    Return the product AB of matrix multiplication.

    """

    A = asarray(A)

    B = asarray(B)

    assert A.shape == B.shape
    if A.size == 0:
        return 0
    assert A.shape == A.T.shape
    A = asarray(A)
    B = asarray(B)
    assert A.shape == B.shape
    assert A.shape == A.T.shape
    num_rows, num_cols = A.shape
    C = np.zeros(shape=(num_rows, num_cols))
    for i in range(0, num_rows):
        for j in range(0, num_rows):
            for k in range(0, num_rows):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    return C.astype(int)
#==============================================================


def square_matrix_multiply_strassens(A, B):

    """

    Return the product AB of matrix multiplication.

    Assume len(A) is a power of 2

    """

    A = asarray(A)

    B = asarray(B)

    assert A.shape == B.shape
    if A.size == 0:
        return 0
    assert A.shape == A.T.shape

    assert (len(A) & (len(A) - 1)) == 0, "A is not a power of 2"

    n = A.shape[0]
    if n == 1:
        return A * B
    else:
        A11 = A[:int(n/2), :int(n/2)]
        A12 = A[:int(n/2), int(n/2):]
        A21 = A[int(n/2):, :int(n/2)]
        A22 = A[int(n/2):, int(n/2):]
        B11 = B[:int(n/2), :int(n/2)]
        B12 = B[:int(n/2), int(n/2):]
        B21 = B[int(n/2):, :int(n/2)]
        B22 = B[int(n/2):, int(n/2):]

        S1 = B12 - B22
        S2 = A11 + A12
        S3 = A21 + A22
        S4 = B21 - B11
        S5 = A11 + A22
        S6 = B11 + B22
        S7 = A12 - A22
        S8 = B21 + B22
        S9 = A11 - A21
        S10 = B11 + B12

        P1 = square_matrix_multiply_strassens(A11, S1)
        P2 = square_matrix_multiply_strassens(S2, B22)
        P3 = square_matrix_multiply_strassens(S3, B11)
        P4 = square_matrix_multiply_strassens(A22, S4)
        P5 = square_matrix_multiply_strassens(S5, S6)
        P6 = square_matrix_multiply_strassens(S7, S8)
        P7 = square_matrix_multiply_strassens(S9, S10)

        C11 = P5 + P4 - P2 + P6
        C12 = P1 + P2
        C21 = P3 + P4
        C22 = P5 + P1 - P3 - P7

        C = np.zeros(shape=(n, n))
        C[:int(n/2), :int(n/2)] = C11
        C[:int(n/2), int(n/2):] = C12
        C[int(n/2):, :int(n/2)] = C21
        C[int(n/2):, int(n/2):] = C22

        return C.astype(int)
#==============================================================


def test():
    #print "STOCK PRICE CHANGES:"
    print(STOCK_PRICE_CHANGES)
    res1 = find_maximum_subarray_brute(STOCK_PRICE_CHANGES)
    res2 = find_maximum_subarray_recursive(STOCK_PRICE_CHANGES)
    res3 = find_maximum_subarray_iterative(STOCK_PRICE_CHANGES)
    #print "Bruteforce approach: %s" % (res1, )
    #print "Recursive approach: %s" % (res2, )
    #print "Iterative approach: %s" % (res3, )
    #print "Matrix 1:"
    print(asarray(B))
    #print "Matrix 2:"
    print(asarray(D))
    #print "Matrix 3:"
    print(asarray(E))
    #print "Matrix 4:"
    print(asarray(F))
    res4 = square_matrix_multiply(E, F)
    res5 = square_matrix_multiply_strassens(B, D)
    #print("Square Matrix Multiplication(Mat1*Mat2):")
    print(res4)
    #print("Square Matrix Multiplication using Strassens(Mat3*Mat4):")
    print(res5)


if __name__ == '__main__':

    test()
#==============================================================
