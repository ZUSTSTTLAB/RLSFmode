from Bio.PDB import *
import numpy as np
from sklearn.neighbors import KDTree

"""
computeCharges.py: Wrapper function to compute charges of atoms in the surface
This file is part of RNA-MgSIF.
"""

def computeCharges(names):
    charge = []
    # for i in set(names):
    #     # count函数某一个字符在列表中的出现次数
    #     print(f"{i}出现{names.count(i)}次")
    for name in names:
        # order = name.split("_")[2]
        # charge_ = name.split("_")[5]
        # ss = order + "_" + charge_
        # means.append(ss)
        charge.append(float(name.split("_")[5]))



    return np.array(charge)

def assignChargesToNewMesh(new_vertices, old_vertices, old_charges, seeder_opts):
    dataset = old_vertices
    testset = new_vertices
    new_charges = np.zeros(len(new_vertices))
    if seeder_opts["feature_interpolation"]:
        num_inter = 4  # Number of interpolation features
        # Assign k old vertices to each new vertex.
        kdt = KDTree(dataset)
        dists, result = kdt.query(testset, k=num_inter)
        # Square the distances (as in the original pyflann)
        dists = np.square(dists)
        # The size of result is the same as new_vertices
        for vi_new in range(len(result)):
            vi_old = result[vi_new]
            dist_old = dists[vi_new]
            # If one vertex is right on top, ignore the rest.
            if dist_old[0] == 0.0:
                new_charges[vi_new] = old_charges[vi_old[0]]
                continue

            total_dist = np.sum(1 / dist_old)
            for i in range(num_inter):
                new_charges[vi_new] += (
                        old_charges[vi_old[i]] * (1 / dist_old[i]) / total_dist
                )
    else:
        # Assign k old vertices to each new vertex.
        kdt = KDTree(dataset)
        dists, result = kdt.query(testset)
        new_charges = old_charges[result]
    return new_charges







