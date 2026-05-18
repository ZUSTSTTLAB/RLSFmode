import numpy as np

def compute_theta(vertexes, vertex_normal, Mg_coords):
    mg_normals = []
    for i in range(vertexes.shape[0]):
        nx = Mg_coords[0] - vertexes[i][0]
        ny = Mg_coords[1] - vertexes[i][1]
        nz = Mg_coords[2] - vertexes[i][2]
        mg_normals.append([nx, ny, nz])
    mg_normals = np.array(mg_normals)
    mg_normals = mg_normals.reshape(vertexes.shape[0], 3)

    theta = []
    for i in range(vertexes.shape[0]):
        vertic_normal = vertex_normal[i]
        mg_normal = mg_normals[i]
        data_M = np.sqrt(np.sum(vertic_normal * vertic_normal))
        data_N = np.sqrt(np.sum(mg_normal * mg_normal))
        cos_theta = np.sum(vertic_normal * mg_normal) / (data_M * data_N)
        theta.append(np.degrees(np.arccos(cos_theta)))

    theta = np.array(theta) / 180
    theta = theta.reshape(vertexes.shape[0], 1)

    return theta