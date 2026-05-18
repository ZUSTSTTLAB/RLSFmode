import time
import os
from sklearn import metrics
import numpy as np
from IPython.core.debugger import set_trace
from sklearn.metrics import accuracy_score, roc_auc_score

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


def pad_indices(indices, max_verts):
    padded_ix = np.zeros((len(indices), max_verts), dtype=int)
    for patch_ix in range(len(indices)):
        padded_ix[patch_ix] = np.concatenate(
            [indices[patch_ix], [patch_ix] * (max_verts - len(indices[patch_ix]))]
        )
    return padded_ix


# Run rna fingerprint
def run_rna_fingerprint(
    params, learning_obj, rho_wrt_center, theta_wrt_center, input_feat, mask, indices, torch_index
):
    indices = pad_indices(indices, mask.shape[1])
    mask = np.expand_dims(mask, 2)
    feed_dict = {
        learning_obj.rho_coords: rho_wrt_center[torch_index],
        learning_obj.theta_coords: theta_wrt_center[torch_index],
        learning_obj.input_feat: input_feat[torch_index],
        learning_obj.mask: mask[torch_index],
        learning_obj.indices_tensor: indices[torch_index],
    }
    global_desc_copy = learning_obj.session.run([learning_obj.global_desc_copy], feed_dict=feed_dict)
    return global_desc_copy

