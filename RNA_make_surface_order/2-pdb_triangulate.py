import os
from default_config.rna_opts import rna_opts
import pymesh
from sklearn.neighbors import KDTree
from RNA_make_surface_order.triangulation.fixmesh import fix_mesh
from RNA_make_surface_order.triangulation.computeMSMS import computeMSMS
from RNA_make_surface_order.triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from RNA_make_surface_order.triangulation.compute_normal import compute_normal
from input_output.save_ply import save_ply
from RNA_make_surface_order.triangulation.computer_theta import compute_theta
import numpy as np
import warnings

warnings.filterwarnings("ignore")



tmp_dir = ""
rna_pdb_dir = ""
pdbqt_to_pdb_dir = ""









# pdb_read_file()  #转化pdb的R链

pdb_ids = []
pdb_filename = []
DiSion = {}
RNA_pdb_names = os.listdir(rna_pdb_dir)

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

for i, d in enumerate(RNA_pdb_names):
    d = d.split(".")
    d = d[0]
    pdb_ids.append(d)


faile_RNA = []

number = 0
for pdb_id in pdb_ids:
    pdb_filename = rna_pdb_dir + pdb_id + ".pdb"
    #DiSion[pdb_id] = DISion_decoyrna(pdb_filename)  # 的距 得到质心与最近原子离
    number = number + 1
    if number >= 1:
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if not os.path.exists(pdbqt_to_pdb_dir):
            os.makedirs(pdbqt_to_pdb_dir)
        tmp_dir+"/"+pdb_id+".pdbqt"


        #protonate(pdb_filename, protonated_file)    执行质子化操作

        #pdbqt_filename = protonated_file  protonated_file改为下一行赋值
        protonated_file = tmp_dir+pdb_id+".pdbqt"



        pqbqt_to_pdb_filename = pdbqt_to_pdb_dir+"/"+pdb_id+".pdb"
        # pdbqt_to_pdb(pdbqt_filename, pqbqt_to_pdb_filename)  #  质子化转PDB文件

        vertices1, faces1, normals1, names1, areas1, Centroid_coords = computeMSMS(protonated_file, protonate=True, Centroid=None, pdb_id=pdb_id)

        # try:
        #     vertices1, faces1, normals1, names1, areas1, Centroid_coords = computeMSMS(protonated_file, protonate=True, Centroid=None, pdb_id=pdb_id)
        # except:
        #     faile_RNA.append(pdb_id)
        #     set_trace()



        mesh = pymesh.form_mesh(vertices1, faces1)
        #mesh1 = pymesh.form_mesh(v3, f3)
        #out_Ply_files = rna_opts["462D_Ply"] + pdb_id + "/" + pdb_id
        out_Ply_files = "/media/xia/757283ec-3725-4b25-b458-8286d61ecf6a/spring_river/xiao_fen_zi/ply21/" + pdb_id + "/" + pdb_id
        #out_Ply_files = "/media/wangkagn/757283ec-3725-4b25-b458-8286d61ecf6a/spring_river/pdb/ply_11/" + pdb_id + "/" + pdb_id
        #calculate_msms_pdb = pdbqt_to_pdb_dir + "/" + pdb_id
        #out_ply_file = rna_opts["462D_Ply"] + pdb_id
        #out_ply_file = "/media/wangkagn/757283ec-3725-4b25-b458-8286d61ecf6a/spring_river/pdb/ply_11/" + pdb_id
        out_ply_file = "/media/xia/757283ec-3725-4b25-b458-8286d61ecf6a/spring_river/xiao_fen_zi/ply21/" + pdb_id
        if not os.path.exists(out_ply_file):
            os.makedirs(out_ply_file)


        # Fix the mesh
        regular_mesh = fix_mesh(mesh, rna_opts['mesh_res'])
        #full_regular_mesh = mesh1
        #v3 = full_regular_mesh.vertices
        #vertices1 = regular_mesh.vertices
        centroid_coords_new = centroid_coords
        centroid_coords = np.array(centroid_coords).reshape(1, -1)
        kdt = KDTree(centroid_coords)
        dist, r = kdt.query(regular_mesh.vertices)  # 各顶点到质心的距离
        #np.save(out_Ply_files + ".npy", dist.T[0])
        dists = normalization(dist).T[0]  # 归一化

        charge = computeCharges(names1)             #顶点电荷值

        vertex_charges = assignChargesToNewMesh(regular_mesh.vertices, vertices1, charge, rna_opts)

        assert (len(dist) == len(regular_mesh.vertices))
        value_interacte = sum(dist) / len(dist)
        iface = np.zeros(len(regular_mesh.vertices))
        iface_v = np.where(dist <= value_interacte)[0]  # 顶点距离小于5A设置为1
        iface[iface_v] = 1.0



        # Compute the normals
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
        # print(regular_mesh.vertices.shape)
        vertex_theta = compute_theta(regular_mesh.vertices, vertex_normal, centroid_coords_new)




        save_ply(out_Ply_files + ".ply", regular_mesh.vertices, \
                 regular_mesh.faces, normals=vertex_normal, iface=iface, charges=vertex_charges,dists=dists,vertex_theta=vertex_theta)
#np.save("faile_neg_RNA.npy", faile_RNA)

        #save_ply(out_Ply_files + ".ply", regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal, charges=vertex_charges)

np.save("faile_neg_RNA.npy", faile_RNA)








#




