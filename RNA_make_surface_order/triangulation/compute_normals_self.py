import numpy as np
from numpy.matlib import repmat
"""
compute_normal.py: Compute the normals of a closed shape.
This file is part of RNA_NET, based on previous matlab code.
"""

###

###
from default_config.global_vars import epsilon as eps


def compute_normal_self(vertex, face):
    print(vertex.shape)
    print("===============================")
    print(face.shape)
    print("===============================")
    print(face[1, :])
    print("===============================")
    vertex = vertex.T
    face = face.T
    print(face)
    print("===============================")
    print(face[1, :])
    nface = np.size(face, 1)
    nvert = np.size(vertex, 1)
    print("===============================")
    print(vertex)
    print("===============================")
    print(vertex[:, face[1, :]])
    print("===============================")




    normalf = crossp(
        vertex[:, face[1, :]] - vertex[:, face[0, :]],
        vertex[:, face[2, :]] - vertex[:, face[0, :]],
    )




def crossp(x, y):

    # x and y are (m,3) dimensional
    z = np.zeros((x.shape))
    z[0, :] = np.multiply(x[1, :], y[2, :]) - np.multiply(x[2, :], y[1, :])
    z[1, :] = np.multiply(x[2, :], y[0, :]) - np.multiply(x[0, :], y[2, :])
    z[2, :] = np.multiply(x[0, :], y[1, :]) - np.multiply(x[1, :], y[0, :])
    return z