import sys
import time
import os
import numpy as np
from IPython.core.debugger import set_trace

from default_config.rna_opts import rna_opts
import warnings
warnings.filterwarnings("ignore")


# load training data (From many files)
from rna_modules.read_data_from_surface import read_data_from_surface

from sklearn import metrics


params = rna_opts['site']
# params['Decoy_rna_data'] = rna_opts['ply_chain_dir']
root_dir = ""
root_dir = ""
rna_pdb_dir = ""
RNA_pdb_names = os.listdir(rna_pdb_dir)
pdb_ids = []
for i, d in enumerate(RNA_pdb_names):
    d = d.split(".")
    d = d[0]
    pdb_ids.append(d)
faile_RNA = []
number = 0
for pdb_id in pdb_ids:
    #number = number + 1
    #if number >= 20752:
        # print(pdb_id)
        all_list_desc = []
        all_list_coords = []
        all_list_shape_idx = []
        all_list_name = []
        idx_positives = []


        my_precomp_dir = "" + pdb_id + '/'

        if not os.path.exists(my_precomp_dir):
            os.makedirs(my_precomp_dir)

        # Read directly from the ply file
        fields = pdb_id
        ply_file = {}
        in_ply_file = "" + pdb_id + "/" + pdb_id + ".ply"

        
        
        
        
        ply_file['p1'] = in_ply_file.format(fields[0:])

        pids = ['p1']
        #pids = ['p1']
        #print(ply_file['p1'])
        # Compute shape
        rho = {}
        neigh_indices = {}
        mask = {}
        input_feat = {}
        theta = {}
        iface_lables = {}
        verts = {}

        pid = 'p1'
        # Compute the angular and radial coordinates--------rho, theta, neigh_indices, mask---
        input_feat[pid], rho[pid], theta[pid], mask[pid], neigh_indices[pid], iface_lables[pid], verts[pid] = read_data_from_surface(ply_file[pid], params)


        # for pid in pids:
        #     try:
        #        input_feat[pid], rho[pid], theta[pid], mask[pid], neigh_indices[pid], \
        #        iface_lables[pid], verts[pid] = read_data_from_surface(ply_file[pid], params)
        #        input_feat[pid], rho[pid], theta[pid], mask[pid], neigh_indices[pid], \
        #        iface_lables[pid], verts[pid] = read_data_from_surface(root_dir + ply_file[pid], params)
        #     except:
        #          print(pdb_id)
        #          continue
        #          set_trace()
         # Save data only if everything went well
        for pid in pids:
            np.save(my_precomp_dir + pid + '_rho_wrt_center', rho[pid])
            np.save(my_precomp_dir + pid + '_theta_wrt_center', theta[pid])
            np.save(my_precomp_dir + pid + '_input_feat', input_feat[pid])
            np.save(my_precomp_dir + pid + '_mask', mask[pid])
            np.save(my_precomp_dir + pid + '_list_indices', neigh_indices[pid])
            np.save(my_precomp_dir + pid + '_iface_labels', iface_lables[pid])

            # Save x,y,z
            # np.save(my_precomp_dir + pid + '_X.npy', verts[pid][0:])
            # np.save(my_precomp_dir + pid + '_Y.npy', verts[pid][:1])
            # np.save(my_precomp_dir + pid + '_Z.npy', verts[pid][:2])
np.save("fiale_RNA.npy", faile_RNA)

