import timeit

from numpy import array
from scipy.sparse import coo_matrix, dok_matrix, lil_matrix, csr_matrix, csc_matrix

A = array([[1, 0, 0, 1, 0, 1], [0, 0, 2, 2, 1, 2], [1, 0, 0, 1, 2, 0]])
print(A)

S = coo_matrix(A)
print(S)

print(S.tocsr()[:, 3])

B = S.todense()
print(B)

times = 10000
timeit.timeit(lambda: dok_matrix(B), number=times) / times
timeit.timeit(lambda: lil_matrix(B), number=times) / times
timeit.timeit(lambda: csr_matrix(B), number=times) / times
timeit.timeit(lambda: csc_matrix(B), number=times) / times
