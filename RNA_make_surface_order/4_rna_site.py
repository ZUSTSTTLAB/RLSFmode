import time
import os
import numpy as np
import pandas as pd
import sys
import importlib
from rna_modules.train_rna_site import run_rna_site
from default_config.rna_opts import rna_opts, custom_params
import pymesh
import tensorflow as tf
from IPython.core.debugger import set_trace
import random
import warnings

warnings.filterwarnings("ignore")


# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


rna_pdb_dir = ""
params = rna_opts["site"]
data_dir = ""
in_ply = ""
save_dir = ""
RNA_pdb_names = os.listdir(rna_pdb_dir)
pdb_ids = []
for i, d in enumerate(RNA_pdb_names):
    d = d.split(".")
    d = d[0]
    pdb_ids.append(d)



# ppi_pair_id\s = ["1b23"]

# custom_params_file = custom_params
# for key in custom_params_file:
#     print("seting {} to {}".format(key, custom_params_file[key]))
#     params[key] = custom_params_file[key]
#     if key not in ("feat_mask", "n_conv_layers"):
#         if not os.path.exists(custom_params_file[key]):
#             os.makedirs(custom_params_file[key])

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Build the neural network model
from rna_modules.RNA_site import RNA_site

learning_obj = RNA_site(
    12,#params["max_distance"],
    n_thetas=14,   #9
    n_rhos=8,   #6
    n_rotations=8,   #4
    idx_gpu="/gpu:0",
    feat_mask= [1.0, 1.0, 1.0, 1.0, 1.0], #params["feat_mask"],
    n_conv_layers=1, #params["n_conv_layers"],
)


idx_count = 0
pids = ["p1"]
patches_number = 0
number = 0
for pdb_id in pdb_ids:
    number = number + 1
    if number >= 1:
        in_dir =  data_dir + pdb_id + '/'
        for pid in pids:
            try:
                rho_wrt_center = np.load(in_dir + pid + "_rho_wrt_center.npy")
            except:
                print(number)
                print(pdb_id)
                print("File not found: {}".format(in_dir + pid + "_rho_wrt_center.npy"))
                set_trace()
            theta_wrt_center = np.load(in_dir + pid + "_theta_wrt_center.npy")
            input_feat = np.load(in_dir + pid + "_input_feat.npy")
            input_feat = mask_input_feat(input_feat, params["feat_mask"])
            mask = np.load(in_dir + pid + "_mask.npy")
            indices = np.load(in_dir + pid + "_list_indices.npy", encoding="latin1", allow_pickle=True)
            labels = np.zeros(len(mask))
            # print("Total number of patches: {}\n".format(len(mask)))

            ply_file = in_ply + pdb_id + "/" + pdb_id + '.ply'
            #ply_file = '../data_prepartion/' + in_ply + pdb_id + "/" + pdb_id + '.ply'
            mymesh = pymesh.load_mesh(ply_file)
            touch_vertex = mymesh.get_attribute("vertex_iface")
            torch_index = np.where(touch_vertex > 0)
            # torch_index = np.array(np.random.choice(torch_index[0], int(len(torch_index[0])/3), replace=False))
            print(touch_vertex)



            tic = time.time()
            tf.compat.v1.reset_default_graph()   # 清空重置计算图，加快卷积运行速度
            global_desc_copy = run_rna_site(
                params,
                learning_obj,
                rho_wrt_center,
                theta_wrt_center,
                input_feat[:, :, (0, 1, 2, 3, 4)],  # input_feat[:, :, :3],
                mask,
                indices,
                torch_index,
            )



            touch_desc = global_desc_copy[0]
            desc_0 =touch_desc[:, :112]
            desc_1 = touch_desc[:, 112:224]
            desc_2 = touch_desc[:, 224:336]
            desc_3 = touch_desc[:, 336:448]
            desc_4 = touch_desc[:, 448:]

            desc_0 = tf.convert_to_tensor(desc_0)
            desc_0 = tf.matmul(
                tf.transpose(desc_0), desc_0
            ) / tf.cast(tf.shape(desc_0)[0], tf.float32)
            desc_0 = tf.compat.v1.Session().run(desc_0)

            desc_1 = tf.convert_to_tensor(desc_1)
            desc_1 = tf.matmul(
                tf.transpose(desc_1), desc_1
            ) / tf.cast(tf.shape(desc_1)[0], tf.float32)
            desc_1 = tf.compat.v1.Session().run(desc_1)

            desc_2 = tf.convert_to_tensor(desc_2)
            desc_2 = tf.matmul(
                tf.transpose(desc_2), desc_2
            ) / tf.cast(tf.shape(desc_2)[0], tf.float32)
            desc_2 = tf.compat.v1.Session().run(desc_2)

            desc_3 = tf.convert_to_tensor(desc_3)
            desc_3 = tf.matmul(
                tf.transpose(desc_3), desc_3
            ) / tf.cast(tf.shape(desc_3)[0], tf.float32)
            desc_3 = tf.compat.v1.Session().run(desc_3)

            desc_4 = tf.convert_to_tensor(desc_4)
            desc_4 = tf.matmul(
                tf.transpose(desc_4), desc_4
            ) / tf.cast(tf.shape(desc_4)[0], tf.float32)
            desc_4 = tf.compat.v1.Session().run(desc_4)

            desc = np.stack([desc_0, desc_1, desc_2, desc_3, desc_4], axis=2)

            toc = time.time()
            print("Total number of patches: {}\n".format(len(mask)))
            # print(
            #     "Total number of patches for which desc were computed: {}\n".format(len(global_desc_copy[0]))
            # )
            patches_number = patches_number + len(torch_index[0])

            print("GPU time (real time, not actual GPU time): {:.3f}s".format(toc - tic))
            # print(desc.shape)
            # print(desc[4])
    # break
            np.save(
                save_dir + pdb_id + ".npy",
                desc,
            )
print("contact_patch=", patches_number)
    #         print(desc)
    #         print(desc.shape)
    # break





