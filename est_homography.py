
import numpy as np
import pdb

def est_homography(x, y, X, Y):
  N = x.size
  A = np.zeros([2 * N, 9])

  i = 0
  while i < N:
    a = np.array([x[i], y[i], 1]).reshape(-1, 3)
    c = np.array([[X[i]], [Y[i]]])
    d = - c * a

    A[2 * i, 0 : 3], A[2 * i + 1, 3 : 6]= a, a
    A[2 * i : 2 * i + 2, 6 : ] = d

    i += 1

  # compute the solution of A
  U, s, V = np.linalg.svd(A, full_matrices=True)
  h = V[8, :]
  H = h.reshape(3, 3)

  return H
